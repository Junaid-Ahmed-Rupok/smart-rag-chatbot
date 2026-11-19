"""The knowledge base: chunks plus hybrid search (semantic FAISS + BM25 keywords, fused with RRF)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .bm25 import BM25, tokenize
from .chunking import Chunk, chunk_documents
from .documents import IngestError, LoadedFile
from .embeddings import Embedder

log = logging.getLogger(__name__)

RRF_K = 60
Progress = Callable[[str, float], None]

_OVERVIEW = re.compile(
    r"\b(summar(y|ize|ise|ies)|overview|key (points|takeaways|ideas)|main (points|ideas|topics)|tl;?dr|"
    r"what (is|are) (this|these|the) (document|documents|file|files|paper|text|report)s? (about|on)|gist)\b",
    re.I,
)


def is_overview_question(text: str) -> bool:
    return bool(_OVERVIEW.search(text))


@dataclass
class FileInfo:
    name: str
    sha256: str
    pages: int
    chunks: int
    characters: int


@dataclass
class Hit:
    chunk: Chunk
    score: float                 # fused rank score (higher is better)
    coverage: float              # share of the question's keywords found in this chunk (0 to 1)
    dense: float | None = None   # cosine similarity when semantic search is available


class KnowledgeBase:
    def __init__(self, embedder_factory: Callable[[], Embedder | None] | None = None,
                 chunk_size: int = 900, chunk_overlap: int = 150, max_chunks: int = 6000):
        self._factory = embedder_factory
        self._embedder: Embedder | None = None
        self._embedder_tried = False
        self.chunk_size, self.chunk_overlap, self.max_chunks = chunk_size, chunk_overlap, max_chunks
        self.files: dict[str, FileInfo] = {}
        self.chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._token_sets: list[set[str]] = []
        self._vecs: np.ndarray | None = None
        self._faiss = None
        self._bm25: BM25 | None = None

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def semantic(self) -> bool:
        return self._vecs is not None and self._embedder is not None

    @property
    def mode(self) -> str:
        return "hybrid" if self.semantic else "keyword"

    def __len__(self) -> int:
        return len(self.chunks)

    def has_documents(self) -> bool:
        return bool(self.files)

    # ── building ─────────────────────────────────────────────────────────────
    def _get_embedder(self) -> Embedder | None:
        if self._embedder is None and not self._embedder_tried and self._factory is not None:
            self._embedder_tried = True
            self._embedder = self._factory()
        return self._embedder

    def add(self, loaded: LoadedFile, progress: Progress | None = None) -> FileInfo:
        """Index one file. Same name and same content is skipped; same name, new content replaces."""
        existing = self.files.get(loaded.name)
        if existing and existing.sha256 == loaded.sha256:
            return existing
        if any(f.sha256 == loaded.sha256 for f in self.files.values()):
            raise IngestError(f"{loaded.name} has the same content as a file that is already indexed.")
        if existing:
            self.remove(loaded.name)

        new_chunks = chunk_documents(loaded.docs, self.chunk_size, self.chunk_overlap)
        if not new_chunks:
            raise IngestError(f"{loaded.name} has no readable text.")
        if len(self.chunks) + len(new_chunks) > self.max_chunks:
            raise IngestError(
                f"Adding {loaded.name} would pass the limit of {self.max_chunks:,} passages. "
                "Remove a document first."
            )

        vecs = None
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                parts, texts = [], [c.text for c in new_chunks]
                step = 64
                for i in range(0, len(texts), step):
                    parts.append(embedder.embed(texts[i:i + step]))
                    if progress:
                        progress(f"Reading {loaded.name}", min(1.0, (i + step) / len(texts)))
                vecs = np.vstack(parts).astype(np.float32)
            except Exception:
                log.exception("Embedding failed; continuing with keyword search only")
                self._drop_semantic()
        if vecs is not None and self._vecs is None and self.chunks:
            # Earlier files were indexed without vectors, so semantic search would be partial. Stay keyword-only.
            vecs = None
        base = len(self.chunks)
        self.chunks.extend(new_chunks)
        if vecs is not None and self._embedder is not None:
            self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])
        elif self._vecs is not None:
            self._drop_semantic()
        info = FileInfo(loaded.name, loaded.sha256, loaded.pages, len(new_chunks), loaded.characters)
        self.files[loaded.name] = info
        self._rebuild(start=base)
        return info

    def _drop_semantic(self) -> None:
        self._vecs, self._faiss, self._embedder = None, None, None

    def remove(self, name: str) -> None:
        if name not in self.files:
            return
        keep = [i for i, c in enumerate(self.chunks) if c.source != name]
        self.chunks = [self.chunks[i] for i in keep]
        if self._vecs is not None:
            self._vecs = self._vecs[keep] if keep else None
        self._tokens, self._token_sets = [], []
        del self.files[name]
        self._rebuild(start=0)

    def clear(self) -> None:
        self.files.clear()
        self.chunks.clear()
        self._tokens, self._token_sets = [], []
        self._vecs, self._faiss, self._bm25 = None, None, None

    def _rebuild(self, start: int) -> None:
        for c in self.chunks[start:]:
            toks = tokenize(c.text)
            self._tokens.append(toks)
            self._token_sets.append(set(toks))
        self._bm25 = BM25(self._tokens) if self._tokens else None
        self._faiss = None
        if self._vecs is not None and len(self._vecs):
            import faiss

            index = faiss.IndexFlatIP(self._vecs.shape[1])      # cosine, because vectors are normalised
            index.add(np.ascontiguousarray(self._vecs))
            self._faiss = index

    # ── search ───────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[Hit]:
        n = len(self.chunks)
        if not n or self._bm25 is None:
            return []
        qtoks = tokenize(query)
        qset = set(qtoks)
        pool = min(n, max(20, k * 4))

        lex = self._bm25.scores(qtoks)
        lex_order = [int(i) for i in np.argsort(-lex)[:pool] if lex[i] > 0]

        dense_order: list[int] = []
        qvec = None
        if self._faiss is not None and self._embedder is not None:
            try:
                qvec = self._embedder.embed([query])
                _, ids = self._faiss.search(np.ascontiguousarray(qvec), pool)
                dense_order = [int(i) for i in ids[0] if i >= 0]
            except Exception:
                log.exception("Semantic query failed; using keywords for this question")
                qvec = None

        fused: dict[int, float] = {}
        for order in (lex_order, dense_order):
            for rank, idx in enumerate(order):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        hits = []
        for idx, score in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
            cov = len(qset & self._token_sets[idx]) / len(qset) if qset else 0.0
            dense = float(self._vecs[idx] @ qvec[0]) if qvec is not None and self._vecs is not None else None
            hits.append(Hit(self.chunks[idx], score, cov, dense))
        return hits

    def overview(self, k: int = 8) -> list[Hit]:
        """An even spread of passages across all documents, for 'summarize' style questions."""
        if not self.chunks:
            return []
        by_file: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            by_file.setdefault(c.source, []).append(i)
        per_file = max(1, k // max(1, len(by_file)))
        picked: list[int] = []
        for idxs in by_file.values():
            if len(idxs) <= per_file:
                picked.extend(idxs)
            else:
                step = (len(idxs) - 1) / (per_file - 1) if per_file > 1 else 0
                picked.extend(idxs[round(j * step)] for j in range(per_file))
        picked = sorted(set(picked))[:k]
        return [Hit(self.chunks[i], 0.0, 1.0, None) for i in picked]

    @staticmethod
    def is_relevant(hits: list[Hit]) -> bool:
        """Heuristic guard against answering from passages that do not actually match the question."""
        for h in hits:
            if h.coverage >= 0.25 or (h.dense is not None and h.dense >= 0.28):
                return True
        return False
