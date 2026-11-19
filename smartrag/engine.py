"""One question-and-answer turn: retrieve, decide whether the documents cover it, stream an answer, cite it."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from .about import FOUNDER_MD, is_founder_question
from .bm25 import tokenize
from .index import Hit, KnowledgeBase, is_overview_question
from .llm import LLMError
from .prompts import build_condense_messages, build_messages
from .smalltalk import OFFLINE_REPLY, chat_system_prompt, is_smalltalk
from .settings import Settings

log = logging.getLogger(__name__)

def refusal_text(files: list[str]) -> str:
    names = ", ".join(files[:3]) + (" and more" if len(files) > 3 else "")
    return (
        f"I couldn't find that in your documents ({names}). Try asking about something they contain, "
        "or use the button below to get a general answer instead."
    )


REFUSAL = refusal_text([])
NO_MODEL = (
    "No language model is connected, so I can't write answers yet. "
    "Add a free `GROQ_API_KEY` to turn them on. Upload documents and I will still show the best matching passages."
)
_REFERENCE = re.compile(r"\b(it|its|they|them|their|this|that|those|these|he|she|him|her|above|previous|former|latter|same)\b", re.I)
_MARKER = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Citation:
    n: int
    source: str
    label: str
    text: str
    page: int | None
    section: str | None
    used: bool


@dataclass
class Answer:
    text: str
    kind: str                         # answer | refusal | extractive | offline | founder | general | error
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0
    mode: str = ""                    # hybrid or keyword
    standalone: str = ""
    error: str | None = None

    @property
    def cited(self) -> bool:
        return any(c.used for c in self.citations)


class Turn:
    def __init__(self, kb: KnowledgeBase, llm, settings: Settings, question: str, history: list[dict],
                 documents_only: bool, model: str, top_k: int):
        self.kb, self.llm, self.s = kb, llm, settings
        self.question = question.strip()
        self.history = [m for m in history if m.get("content")]
        self.documents_only, self.model, self.top_k = documents_only, model, top_k
        self.hits: list[Hit] = []
        self.text = ""
        self.kind = "answer"
        self.error: str | None = None
        self.standalone = self.question
        self._t0 = time.time()
        self._done = False

    # ── helpers ──────────────────────────────────────────────────────────────
    def _needs_rewrite(self) -> bool:
        if not (self.history and self.llm.available):
            return False
        words = self.question.split()
        return len(words) < 8 or bool(_REFERENCE.search(self.question))

    def _standalone(self) -> str:
        if not self._needs_rewrite():
            return self.question
        try:
            out = self.llm.complete(build_condense_messages(self.question, self.history), self.model)
            out = out.strip().strip('"')
            return out if 3 <= len(out) <= 400 else self.question
        except Exception:
            return self.question

    def _extractive(self) -> str:
        qset = set(tokenize(self.standalone))
        lines = []
        for i, h in enumerate(self.hits[:3], start=1):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n+", h.chunk.text) if len(s.strip()) > 20]
            ranked = sorted(range(len(sentences)),
                            key=lambda j: -len(qset & set(tokenize(sentences[j]))))[:2]
            snippet = " ".join(sentences[j] for j in sorted(ranked)) or h.chunk.text[:300]
            snippet = _MARKER.sub(r"(\1)", snippet[:420])      # keep the document's own [n] from posing as a citation
            lines.append(f"- {snippet} [{i}]")
        return ("**Best matches in your documents** (no language model is connected, "
                "so these are the passages themselves):\n\n" + "\n".join(lines))

    # ── main generator ───────────────────────────────────────────────────────
    def stream(self) -> Iterator[str]:
        if is_founder_question(self.question):
            self.kind = "founder"
            self.text = "I was built by " + FOUNDER_MD
            self._done = True
            yield self.text
            return

        if is_smalltalk(self.question):
            yield from self._chat()
            return

        if self.kb.has_documents():
            self.standalone = self._standalone()
            overview = is_overview_question(self.question)
            hits = self.kb.overview(self.top_k + 3) if overview else self.kb.search(self.standalone, self.top_k)
            relevant = overview or self.kb.is_relevant(hits)
            if relevant:
                self.hits = hits
            elif self.documents_only:
                self.kind, self.text, self._done = "refusal", refusal_text(list(self.kb.files)), True
                yield self.text
                return

        if not self.llm.available:
            if self.hits:
                self.kind, self.text = "extractive", self._extractive()
            else:
                self.kind, self.text = "offline", NO_MODEL
            self._done = True
            yield self.text
            return

        messages = build_messages(self.question, self.hits, self.history, self.documents_only, self.s.memory_turns)
        self.kind = "answer" if self.hits else "general"
        try:
            for piece in self.llm.stream(messages, self.model, self.s.temperature, self.s.max_tokens):
                self.text += piece
                yield piece
        except LLMError as exc:
            self.kind, self.error = "error", exc.message
            note = f"⚠️ {exc.message}"
            sep = "\n\n" if self.text else ""
            self.text += sep + note
            yield sep + note
        except Exception:
            log.exception("Unexpected model failure")
            self.kind, self.error = "error", "Something went wrong while writing the answer."
            note = "⚠️ Something went wrong while writing the answer. Please try again."
            self.text += ("\n\n" if self.text else "") + note
            yield ("\n\n" if self.text != note else "") + note
        self._done = True

    def _model_used(self) -> str:
        """Name the model only when one was actually called for this answer."""
        if self.kind in ("refusal", "founder", "offline", "extractive") or not self.llm.available:
            return ""
        return getattr(self.llm, "last_model", "") or self.model

    def _chat(self) -> Iterator[str]:
        """Friendly conversation: no retrieval, no citations."""
        self.kind = "chat"
        if not self.llm.available:
            self.text = OFFLINE_REPLY
            self._done = True
            yield self.text
            return
        messages = [{"role": "system", "content": chat_system_prompt(list(self.kb.files))}]
        for m in self.history[-(self.s.memory_turns * 2):]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"][:1500]})
        messages.append({"role": "user", "content": self.question})
        try:
            for piece in self.llm.stream(messages, self.model, 0.5, 200):
                self.text += piece
                yield piece
        except LLMError as exc:
            self.kind, self.error = "error", exc.message
            self.text += ("\n\n" if self.text else "") + f"⚠️ {exc.message}"
            yield f"⚠️ {exc.message}"
        except Exception:
            log.exception("Unexpected model failure")
            self.kind, self.error = "error", "Something went wrong."
            self.text += "⚠️ Something went wrong. Please try again."
            yield "⚠️ Something went wrong. Please try again."
        self._done = True

    def result(self) -> Answer:
        text = self.text.strip()
        used: set[int] = set()
        if self.hits and self.kind in ("answer", "general"):
            used = {int(n) for n in _MARKER.findall(text) if 1 <= int(n) <= len(self.hits)}
            text = _MARKER.sub(lambda m: m.group(0) if int(m.group(1)) <= len(self.hits) else "", text)
        citations = [
            Citation(i, h.chunk.source, h.chunk.label, h.chunk.text, h.chunk.page, h.chunk.section,
                     used=(i in used) or self.kind == "extractive")
            for i, h in enumerate(self.hits, start=1)
        ]
        return Answer(
            text=text, kind=self.kind, citations=citations,
            model=self._model_used(),
            seconds=round(time.time() - self._t0, 1), mode=self.kb.mode,
            standalone=self.standalone, error=self.error,
        )


class Engine:
    def __init__(self, kb: KnowledgeBase, llm, settings: Settings):
        self.kb, self.llm, self.settings = kb, llm, settings

    def ask(self, question: str, history: list[dict], *, documents_only: bool = True,
            model: str = "", top_k: int | None = None) -> Turn:
        return Turn(self.kb, self.llm, self.settings, question, history, documents_only,
                    model or self.settings.groq_model, top_k or self.settings.top_k)
