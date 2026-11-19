"""A small, dependency-free BM25 keyword ranker (works for English and for Bengali text)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_STOP = frozenset(
    "a an and are as at be been but by for from had has have he her his i if in into is it its of on or she "
    "that the their them they this to was we were what when where which who whom why will with you your do "
    "does did how can could should would about not no than then so such these those there here me my our us "
    "tell please give show explain describe list".split()
)
_SPLIT = re.compile(r"[\s.,;:!?()\[\]{}\"'“”‘’<>/\\|—–\-_=+*#@^~`।]+")


def _stem(tok: str) -> str:
    if not tok.isascii() or len(tok) < 5:
        return tok
    for suffix in ("ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = [t for t in _SPLIT.split(text.lower()) if t]
    if not keep_stop:
        toks = [t for t in toks if t not in _STOP]
    return [_stem(t) for t in toks]


class BM25:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(tokenized_docs)
        self.lengths = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avg = float(self.lengths.mean()) if self.n else 0.0
        self.tf = [Counter(d) for d in tokenized_docs]
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n or not query_tokens or self.avg == 0:
            return out
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out
