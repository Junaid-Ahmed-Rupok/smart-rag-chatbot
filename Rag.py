"""
rag.py — RAG pipeline using Ollama (local, free, no API key).
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from settings import SESSION_KEYS, cfg

log = logging.getLogger(__name__)

# ── Loader registry ───────────────────────────────────────────────────────────

_LOADERS: dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}

# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Local RAG pipeline backed by Ollama + FAISS.

    Raises exceptions on failure — never calls Streamlit directly.
    The caller (app.py) is responsible for surfacing errors to the user.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name    = model_name
        self.embeddings:   OllamaEmbeddings | None = None
        self.llm:          Ollama | None            = None
        self.vector_store: FAISS | None             = None
        self.chain:        ConversationalRetrievalChain | None = None
        self._processed:   set[str] = set()

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_models(self) -> None:
        """Lazy-init embeddings and LLM. Safe to call multiple times."""
        if self.embeddings is None:
            self.embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=cfg.ollama_host,
            )
            log.debug("Embeddings initialised: %s", self.model_name)

        if self.llm is None:
            self.llm = Ollama(
                model=self.model_name,
                base_url=cfg.ollama_host,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
            )
            log.debug("LLM initialised: %s (temp=%.2f)", self.model_name, cfg.temperature)

    # ── Document processing ───────────────────────────────────────────────────

    def process_documents(self, files: list) -> int:
        """
        Load, chunk, and index uploaded files.

        Args:
            files: Streamlit UploadedFile objects.

        Returns:
            Total number of chunks indexed.

        Raises:
            ValueError: If no chunks could be extracted.
            Exception:  Propagates loader/FAISS errors for the caller to handle.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.chunk_separators,
        )

        all_chunks = []

        for file in files:
            if file.name in self._processed:
                log.debug("Skipping already-indexed file: %s", file.name)
                continue

            suffix = Path(file.name).suffix.lower()
            loader_cls = _LOADERS.get(suffix)

            if loader_cls is None:
                log.warning("Unsupported file type, skipping: %s", file.name)
                continue

            tmp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(file.read())
                    tmp_path = Path(tmp.name)

                documents = loader_cls(str(tmp_path)).load()

                for doc in documents:
                    doc.metadata["source"] = file.name

                chunks = splitter.split_documents(documents)
                all_chunks.extend(chunks)
                self._processed.add(file.name)
                log.info("Loaded %s → %d chunks", file.name, len(chunks))

            except Exception:
                log.exception("Failed to process file: %s", file.name)
                raise

            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

        if not all_chunks:
            raise ValueError(
                "No content could be extracted from the uploaded files. "
                "Check that the files are not empty or password-protected."
            )

        self.init_models()

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        else:
            self.vector_store.add_documents(all_chunks)

        # Invalidate the chain so it's rebuilt with the updated retriever
        self.chain = None
        log.info("Vector store updated. Total chunks: %d", len(all_chunks))

        return len(all_chunks)

    # ── Retrieval chain ───────────────────────────────────────────────────────

    def _get_chain(self) -> ConversationalRetrievalChain:
        """
        Return the cached chain, or build it if it doesn't exist yet.
        Memory is preserved across calls — only reset when the vector
        store is rebuilt (i.e. new documents are indexed).
        """
        if self.chain is not None:
            return self.chain

        if self.vector_store is None:
            raise ValueError("Index is empty. Process documents before asking questions.")

        self.init_models()

        retriever = self.vector_store.as_retriever(
            search_type=cfg.search_type,
            search_kwargs={
                "k":      cfg.retrieval_k,
                "fetch_k": cfg.retrieval_fetch_k,
                "lambda_mult": cfg.mmr_lambda,
            },
        )

        memory = ConversationBufferWindowMemory(
            k=cfg.memory_length,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=cfg.debug,
        )
        log.debug("Conversation chain built (memory_k=%d)", cfg.memory_length)
        return self.chain

    # ── Inference ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict[str, Any]:
        """
        Run a question through the RAG chain.

        Returns:
            {"answer": str, "sources": list[str]}

        Raises:
            ValueError: If no documents are indexed.
            Exception:  Propagates LLM/retriever errors.
        """
        chain  = self._get_chain()
        result = chain.invoke({"question": question})

        # Deduplicate sources while preserving order
        seen: set[str]   = set()
        sources: list[str] = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        log.info(
            "Question answered. Sources: %s | Answer length: %d chars",
            sources,
            len(result["answer"]),
        )

        return {"answer": result["answer"], "sources": sources}

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset the pipeline. Called by 'Reset all' in the UI."""
        self.vector_store = None
        self.chain        = None
        self._processed   = set()
        log.info("RAG pipeline cleared")


# ── Streamlit session helpers ─────────────────────────────────────────────────

def init_rag() -> None:
    """Ensure RAG session keys exist. Call once at app startup."""
    st.session_state.setdefault(SESSION_KEYS["rag_pipeline"], None)


def get_rag_pipeline(model_name: str) -> RAGPipeline:
    """
    Return the cached pipeline for this session, or create a new one.
    If the model changes, the old pipeline is discarded.
    """
    current: RAGPipeline | None = st.session_state.get(SESSION_KEYS["rag_pipeline"])

    if current is None or current.model_name != model_name:
        log.info("Creating new RAG pipeline for model: %s", model_name)
        st.session_state[SESSION_KEYS["rag_pipeline"]] = RAGPipeline(model_name)

    return st.session_state[SESSION_KEYS["rag_pipeline"]]
