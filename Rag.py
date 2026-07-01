"""
Rag.py — RAG pipeline.
"""

import logging
import shutil
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

import Session
from Config import SESSION_KEYS, cfg

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
    RAG pipeline backed by FAISS, with a swappable chat-model provider.
    """

    def __init__(self, chat_model: str) -> None:
        self.chat_model:    str = chat_model
        self.provider:      str = cfg.llm_provider

        self.embeddings:   HuggingFaceEmbeddings | None = None
        self.llm:          ChatGroq | Ollama | None     = None
        self.vector_store: FAISS | None                 = None
        self.chain:        ConversationalRetrievalChain | None = None
        self._processed:   set[str] = set()

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def set_chat_model(self, chat_model: str) -> None:
        if chat_model == self.chat_model:
            return

        log.info("Switching chat model: %s -> %s", self.chat_model, chat_model)
        self.chat_model = chat_model
        self.llm = None
        self.chain = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_models(self) -> None:
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=cfg.local_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            log.debug("Embeddings initialised (local): %s", cfg.local_embedding_model)

        if self.llm is None:
            if self.provider == "groq":
                self.llm = ChatGroq(
                    model=self.chat_model,
                    api_key=cfg.groq_api_key,
                    temperature=cfg.temperature,
                    model_kwargs={"top_p": cfg.top_p},
                )
                log.debug("LLM initialised: groq/%s (temp=%.2f)", self.chat_model, cfg.temperature)
            else:
                self.llm = Ollama(
                    model=self.chat_model,
                    base_url=cfg.ollama_host,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    repeat_penalty=cfg.repeat_penalty,
                )
                log.debug("LLM initialised: ollama/%s (temp=%.2f)", self.chat_model, cfg.temperature)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the current FAISS index to disk at the fixed path."""
        if self.vector_store is None:
            return
        try:
            path = cfg.vector_store_path
            path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(path))
            log.info("Vector store persisted to %s", path)
        except Exception:
            log.exception("Failed to persist vector store")

    def load(self) -> bool:
        """Load a previously persisted FAISS index from the fixed path."""
        path = cfg.vector_store_path
        if not (path / "index.faiss").exists():
            return False
        try:
            self.init_models()
            self.vector_store = FAISS.load_local(
                str(path), self.embeddings, allow_dangerous_deserialization=True
            )
            log.info("Vector store loaded from %s", path)
            return True
        except Exception:
            log.exception("Failed to load persisted vector store")
            return False

    # ── Document processing ───────────────────────────────────────────────────

    def process_documents(self, files: list) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.chunk_separators,
        )

        all_chunks = []
        failed: list[str] = []

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
                log.info("Loaded %s -> %d chunks", file.name, len(chunks))

            except Exception:
                log.exception("Failed to process file: %s", file.name)
                failed.append(file.name)
                continue

            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

        if not all_chunks:
            raise ValueError(
                "No content could be extracted from the uploaded files. "
                "Check that the files are not empty, corrupted, or password-protected."
            )

        self.init_models()

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        else:
            self.vector_store.add_documents(all_chunks)

        self.chain = None
        log.info("Vector store updated. New chunks: %d", len(all_chunks))

        if failed:
            log.warning("Indexing completed with %d failed file(s): %s", len(failed), failed)

        # Persist to disk
        self.save()

        return len(all_chunks)

    # ── Retrieval chain ───────────────────────────────────────────────────────

    def _get_chain(self) -> ConversationalRetrievalChain:
        if self.chain is not None:
            return self.chain

        if self.vector_store is None:
            raise ValueError("Index is empty. Process documents before asking questions.")

        self.init_models()

        search_kwargs: dict[str, Any] = {"k": cfg.retrieval_k}
        if cfg.search_type == "mmr":
            search_kwargs["fetch_k"] = cfg.retrieval_fetch_k
            search_kwargs["lambda_mult"] = cfg.mmr_lambda

        retriever = self.vector_store.as_retriever(
            search_type=cfg.search_type,
            search_kwargs=search_kwargs,
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
        log.debug("Conversation chain built (model=%s, memory_k=%d)", self.chat_model, cfg.memory_length)
        return self.chain

    # ── Inference ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict[str, Any]:
        chain  = self._get_chain()
        result = chain.invoke({"question": question})

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

    def ask_direct(self, question: str) -> str:
        self.init_models()
        result = self.llm.invoke(question)
        return result.content if hasattr(result, "content") else result

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.vector_store = None
        self.chain        = None
        self._processed   = set()
        log.info("RAG pipeline cleared")


# ── Streamlit session helpers ─────────────────────────────────────────────────

def init_rag() -> None:
    st.session_state.setdefault(SESSION_KEYS["rag_pipeline"], None)


def get_rag_pipeline(chat_model: str) -> RAGPipeline:
    current: RAGPipeline | None = st.session_state.get(SESSION_KEYS["rag_pipeline"])

    if current is None:
        log.info("Creating new RAG pipeline (provider=%s, chat_model=%s, embedding_model=%s)", cfg.llm_provider, chat_model, cfg.local_embedding_model)
        current = RAGPipeline(chat_model)
        current.load()  # <-- No session_id argument
        st.session_state[SESSION_KEYS["rag_pipeline"]] = current
    else:
        current.set_chat_model(chat_model)

    return current


def delete_persisted_store() -> None:
    """Delete the on-disk FAISS index from the fixed path."""
    path = cfg.vector_store_path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        log.info("Deleted persisted vector store: %s", path)
