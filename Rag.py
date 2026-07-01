"""
Rag.py — RAG pipeline.

Chat model: Groq's hosted API (fast, free-tier, no local install — this is
what makes the app a one-command `streamlit run` for end users) or, if
configured, a local Ollama model.

Embeddings: always a small local sentence-transformers model. Groq has no
embeddings endpoint, so this is the one piece that always runs on-device —
but it ships as a normal pip dependency, no separate server required.

Persistence: the FAISS index is optionally saved to disk under
cfg.vector_store_path, namespaced by Session.session_id() so concurrent
users on a shared deployment never read or write each other's documents.
Note that on ephemeral hosts (e.g. Streamlit Cloud), this survives only
for the lifetime of the running container — it is not durable storage.
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
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
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

    Design note — embeddings vs. chat model are intentionally decoupled:
    embeddings always use a fixed local sentence-transformers model
    (`cfg.local_embedding_model`), while the chat model comes from
    whichever provider is configured (`cfg.llm_provider`: "groq" or
    "ollama") and can change freely via `set_chat_model()`. This means
    switching the active chat model mid-session never invalidates the
    vector store — only the LLM and the conversation chain are rebuilt.
    Embeddings being provider-independent also means a model switch
    (or even a provider switch) never re-triggers re-embedding.

    Raises exceptions on failure — never calls Streamlit directly.
    The caller (Sidebar.py / Chat.py) is responsible for surfacing
    errors to the user.
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
        """
        Swap the active chat model in place.

        Only the LLM and the (LLM-bound) conversation chain are
        invalidated — embeddings and the FAISS index are left untouched,
        so previously indexed documents remain queryable immediately
        with the new model.
        """
        if chat_model == self.chat_model:
            return

        log.info("Switching chat model: %s -> %s", self.chat_model, chat_model)
        self.chat_model = chat_model
        self.llm = None     # lazily rebuilt by init_models()
        self.chain = None   # chain is bound to self.llm, must be rebuilt

    # ── Initialisation ────────────────────────────────────────────────────────

    def init_models(self) -> None:
        """Lazy-init embeddings and LLM. Safe to call multiple times."""
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

    @staticmethod
    def _store_dir(session_id: str) -> Path:
        """On-disk location for one session's FAISS index. Namespaced by
        session_id so concurrent users never share (or leak into) an index."""
        return Path(cfg.vector_store_path) / session_id

    def save(self, session_id: str) -> None:
        """Persist the current FAISS index to disk. No-op if nothing is
        indexed yet. Failures are logged, not raised — persistence is a
        nice-to-have, it should never break an otherwise-successful upload."""
        if self.vector_store is None:
            return
        try:
            path = self._store_dir(session_id)
            path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(path))
            log.info("Vector store persisted to %s", path)
        except Exception:
            log.exception("Failed to persist vector store for session %s", session_id)

    def load(self, session_id: str) -> bool:
        """Load a previously persisted FAISS index for this session, if one
        exists. Returns True if a store was loaded, False otherwise."""
        path = self._store_dir(session_id)
        if not (path / "index.faiss").exists():
            return False
        try:
            self.init_models()
            # Safe here because we only ever load indexes this same app
            # previously wrote to this same namespaced path — never
            # arbitrary user-supplied files.
            self.vector_store = FAISS.load_local(
                str(path), self.embeddings, allow_dangerous_deserialization=True
            )
            log.info("Vector store loaded from %s", path)
            return True
        except Exception:
            log.exception("Failed to load persisted vector store for session %s", session_id)
            return False

    # ── Document processing ───────────────────────────────────────────────────

    def process_documents(self, files: list) -> int:
        """
        Load, chunk, and index uploaded files.

        Args:
            files: Streamlit UploadedFile objects.

        Returns:
            Total number of chunks indexed in this call.

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
                # Don't let one bad file (corrupted/encrypted/etc.) abort
                # the whole batch — log it, skip it, keep going, and tell
                # the caller which files failed once the batch is done.
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

        # Invalidate the chain so it's rebuilt with the updated retriever.
        self.chain = None
        log.info("Vector store updated. New chunks: %d", len(all_chunks))

        if failed:
            log.warning("Indexing completed with %d failed file(s): %s", len(failed), failed)

        # Persist to disk so this session's index survives a page refresh
        # within the same running container.
        self.save(Session.session_id())

        return len(all_chunks)

    # ── Retrieval chain ───────────────────────────────────────────────────────

    def _get_chain(self) -> ConversationalRetrievalChain:
        """
        Return the cached chain, or build it if it doesn't exist yet.
        Memory is preserved across calls — only reset when the chain
        itself is rebuilt (new documents indexed, or chat model changed).
        """
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

        # Deduplicate sources while preserving order.
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
        """
        Bare chat-model call, bypassing retrieval — used when no documents
        are indexed yet. Normalizes the return type across providers:
        Ollama's legacy LLM interface returns a plain str, while ChatGroq
        (a chat model) returns an AIMessage whose text lives in `.content`.
        """
        self.init_models()
        result = self.llm.invoke(question)
        return result.content if hasattr(result, "content") else result

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


def get_rag_pipeline(chat_model: str) -> RAGPipeline:
    """
    Return the cached pipeline for this session, creating one on first
    use. If the user switches chat models, the *same* pipeline instance
    is reused with its chat model swapped in place — embeddings, the
    FAISS index, and indexed-file bookkeeping are preserved, so a model
    switch no longer silently discards indexed documents.

    On first creation, attempts to load a previously persisted index for
    this session_id (e.g. after a page refresh within the same running
    container).
    """
    current: RAGPipeline | None = st.session_state.get(SESSION_KEYS["rag_pipeline"])

    if current is None:
        log.info("Creating new RAG pipeline (provider=%s, chat_model=%s, embedding_model=%s)", cfg.llm_provider, chat_model, cfg.local_embedding_model)
        current = RAGPipeline(chat_model)
        current.load(Session.session_id())
        st.session_state[SESSION_KEYS["rag_pipeline"]] = current
    else:
        current.set_chat_model(chat_model)

    return current


def delete_persisted_store(session_id: str) -> None:
    """Delete the on-disk FAISS index for a given session, if any. Called
    from Sidebar.py's 'Reset all' so a full reset also clears disk state —
    kept as a module-level function (rather than inside Session.py) to
    avoid a circular import between Session.py and Rag.py."""
    path = Path(cfg.vector_store_path) / session_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        log.info("Deleted persisted vector store: %s", path)
