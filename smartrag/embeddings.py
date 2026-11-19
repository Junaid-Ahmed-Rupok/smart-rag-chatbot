"""One shared, lazily loaded embedding model for the whole server process.

Loading the model takes a while (and memory), so it is loaded once, in the background
as soon as the app starts, and shared by every visitor. If it cannot be loaded (no
network, no memory) the app keeps working with keyword search and says so.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)
RETRY_AFTER_SECONDS = 300


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


_lock = threading.Lock()
_state: dict = {"embedder": None, "failed_at": 0.0, "loading": False}


def _load(model_name: str) -> Embedder | None:
    with _lock:
        if _state["embedder"] is not None:
            return _state["embedder"]
        if time.time() - _state["failed_at"] < RETRY_AFTER_SECONDS:
            return None
        _state["loading"] = True
        try:
            _state["embedder"] = SentenceTransformerEmbedder(model_name)
            log.info("Embedding model ready: %s", model_name)
        except Exception:
            _state["failed_at"] = time.time()
            log.exception("Embedding model unavailable; falling back to keyword search")
        finally:
            _state["loading"] = False
        return _state["embedder"]


def get_embedder(model_name: str) -> Embedder | None:
    """Return the shared embedder, waiting for it if it is still loading. None if unavailable."""
    return _load(model_name)


def warm_up(model_name: str) -> None:
    """Start loading the model in the background (safe to call on every page run)."""
    if _state["embedder"] is not None or _state["loading"]:
        return
    if time.time() - _state["failed_at"] < RETRY_AFTER_SECONDS:
        return
    threading.Thread(target=_load, args=(model_name,), daemon=True, name="embedder-warmup").start()


def status() -> str:
    if _state["embedder"] is not None:
        return "ready"
    if _state["loading"]:
        return "loading"
    return "unavailable" if _state["failed_at"] else "idle"


def set_embedder(embedder: Embedder | None) -> None:
    """Install an embedder directly (used by tests)."""
    with _lock:
        _state["embedder"] = embedder
        _state["failed_at"] = 0.0
        _state["loading"] = False
