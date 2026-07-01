"""
Rag.py — RAG pipeline.
"""

import logging
import shutil
import tempfile
import time
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

# Marker file dropped in a session's store folder every time it's touched.
# Its mtime is what the janitor uses to decide a session is stale.
_ACTIVITY_MARKER = ".last_active"

# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    RAG pipeline backed by FAISS, with a swappable chat-model provider.
    Persists to a folder scoped to a single browser session (see
    Config.session_store_path) so uploaded documents never leak into,
    or survive into, a different chat.
    """

    def __init__(self, chat_model: str, session_id: str) -> None:
        self.chat_model:    str = chat_model
        self.session_id:    str = session_id
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

    def _store_path(self) -> Path:
        return cfg.session_store_path(self.session_id)

    def save(self) -> None:
        """Persist the current FAISS index to this session's own folder."""
        if self.vector_store is None:
            return
        try:
            path = self._store_path()
            path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(path))
            _touch_marker(path)
            log.info("Vector store persisted to %s", path)
        except Exception:
            log.exception("Failed to persist vector store")

    def load(self) -> bool:
        """Load this session's previously persisted FAISS index, if any."""
        path = self._store_path()
        if not (path / "index.faiss").exists():
            return False
        try:
            self.init_models()
            self.vector_store = FAISS.load_local(
                str(path), self.embeddings, allow_dangerous_deserialization=True
            )
            _touch_marker(path)
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
                # The RAW uploaded file on disk (temp copy) is always wiped
                # immediately after we've extracted its text — regardless
                # of chat lifetime. Only the derived FAISS index sticks
                # around, scoped to this session, until it expires/resets.
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

        # Persist to this session's own folder (also refreshes activity marker)
        self.save()

        return len(all_chunks)

    # ── Retrieval chain ───────────────────────────────────────────────────────

    def _get_chain(self) -> ConversationalRetrievalChain:
        if self.chain is not None:
            return self.chain

        self.init_models()

        memory = ConversationBufferWindowMemory(
            k=cfg.memory_length,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        search_kwargs = {"k": cfg.retrieval_k}
        if cfg.search_type == "mmr":
            search_kwargs["fetch_k"] = cfg.retrieval_fetch_k
            search_kwargs["lambda_mult"] = cfg.mmr_lambda

        retriever = self.vector_store.as_retriever(
            search_type=cfg.search_type,
            search_kwargs=search_kwargs,
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
        )
        return self.chain

    def ask(self, question: str) -> dict[str, Any]:
        chain = self._get_chain()
        result = chain.invoke({"question": question})

        sources = sorted({
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        })

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
    session_id = Session.get_session_id()
    current: RAGPipeline | None = st.session_state.get(SESSION_KEYS["rag_pipeline"])

    if current is None:
        log.info(
            "Creating new RAG pipeline (session=%s, provider=%s, chat_model=%s, embedding_model=%s)",
            session_id, cfg.llm_provider, chat_model, cfg.local_embedding_model,
        )
        current = RAGPipeline(chat_model, session_id)
        current.load()
        st.session_state[SESSION_KEYS["rag_pipeline"]] = current
    else:
        current.set_chat_model(chat_model)

    return current


def delete_persisted_store(session_id: str) -> None:
    """Delete the on-disk FAISS index for one specific session."""
    path = cfg.session_store_path(session_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        log.info("Deleted persisted vector store for session %s: %s", session_id, path)


def touch_session_activity(session_id: str) -> None:
    """
    Refresh the "still alive" marker for a session's store folder, if it
    exists. Call this once per app rerun so an ongoing chat never gets
    swept up by the stale-session janitor while it's still in use.
    """
    path = cfg.session_store_path(session_id)
    if path.exists():
        _touch_marker(path)


def cleanup_stale_sessions() -> int:
    """
    Janitor: deletes any session's persisted vector store once it has
    gone longer than SESSION_TTL_MINUTES without activity — this is what
    makes an uploaded PDF/DOCX vanish once its chat is actually over,
    since Streamlit has no reliable "tab closed" signal to hook into.
    Safe to call on every app boot; cheap when there's nothing stale.
    """
    root = cfg.vector_store_root
    if not root.exists():
        return 0

    ttl_seconds = cfg.session_ttl_minutes * 60
    now = time.time()
    removed = 0

    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        marker = session_dir / _ACTIVITY_MARKER
        # Fall back to the directory's own mtime if the marker is missing
        # (e.g. upgraded from an older version of the store).
        last_active = marker.stat().st_mtime if marker.exists() else session_dir.stat().st_mtime

        if now - last_active > ttl_seconds:
            shutil.rmtree(session_dir, ignore_errors=True)
            removed += 1
            log.info("Janitor removed stale session store: %s", session_dir)

    if removed:
        log.info("Janitor cleanup complete: %d stale session(s) removed", removed)

    return removed


def _touch_marker(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / _ACTIVITY_MARKER).touch(exist_ok=True)
