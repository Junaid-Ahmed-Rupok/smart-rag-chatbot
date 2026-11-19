import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartrag import embeddings  # noqa: E402
from smartrag.bm25 import tokenize  # noqa: E402
from smartrag.llm import LLMError  # noqa: E402
from smartrag.settings import load_settings  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar words give similar vectors."""
    DIM = 128

    def embed(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, hash(w) % self.DIM] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class BrokenEmbedder:
    def embed(self, texts):
        raise RuntimeError("boom")


class FakeLLM:
    """Scripted model: replies with `reply` in small pieces and records every call."""
    available = True
    provider = "fake"

    def __init__(self, reply="Chunk overlap keeps context between chunks [1].", fail=None, condense="what is chunk overlap"):
        self.reply, self.fail, self.condense = reply, fail, condense
        self.calls: list[list[dict]] = []
        self.last_model = "fake-model"

    def list_models(self):
        return ["fake-model", "other-model"]

    def stream(self, messages, model, temperature=0.2, max_tokens=900):
        self.calls.append(messages)
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
        if self.fail:
            raise self.fail

    def complete(self, messages, model, temperature=0.0, max_tokens=200):
        self.calls.append(messages)
        return self.condense


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    return load_settings()


@pytest.fixture(autouse=True)
def _reset_embedder_state():
    embeddings.set_embedder(None)
    yield
    embeddings.set_embedder(None)


SAMPLE_MD = (
    b"# Chunking\nChunk overlap keeps context between neighbouring chunks so no fact is cut in half.\n\n"
    b"# Retrieval\nHybrid search mixes keyword search and semantic search using reciprocal rank fusion.\n\n"
    b"# Limits\nScanned PDFs need optical character recognition, which this tool does not do."
)
