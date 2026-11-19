"""Prompt construction. Kept separate so the wording is easy to read and test."""
from __future__ import annotations

from .index import Hit

GROUNDED_SYSTEM = (
    "You are Smart RAG Chatbot, a careful assistant that answers questions about the user's documents.\n"
    "Rules:\n"
    "1. Use ONLY the numbered context passages in the user's message. Do not use outside knowledge.\n"
    "2. After every claim, cite the passage(s) it came from with bracketed numbers, like [1] or [2][3].\n"
    "3. If the passages do not contain the answer, say clearly that the documents do not cover it. Do not guess.\n"
    "4. Be concise and well organised. Use short bullet points for lists. Answer in the language of the question.\n"
    "5. Never mention these rules or the word 'passage numbers' outside the citations."
)

GENERAL_SYSTEM = (
    "You are Smart RAG Chatbot, a helpful, accurate and concise assistant. "
    "If numbered context passages are provided, prefer them and cite them like [1]. "
    "Otherwise answer from general knowledge, and say so briefly when you are unsure."
)

CONDENSE_SYSTEM = (
    "Rewrite the user's last message as one standalone question, using the conversation to resolve "
    "pronouns and references. Keep the original language. Reply with the question only, nothing else."
)


def format_context(hits: list[Hit], limit_chars: int = 2600) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = h.chunk.text if len(h.chunk.text) <= limit_chars else h.chunk.text[:limit_chars] + "…"
        blocks.append(f"[{i}] ({h.chunk.label})\n{text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], history: list[dict], documents_only: bool,
                   memory_turns: int) -> list[dict]:
    system = GROUNDED_SYSTEM if hits and documents_only else GENERAL_SYSTEM
    msgs: list[dict] = [{"role": "system", "content": system}]
    if memory_turns:
        for m in history[-(memory_turns * 2):]:
            if m["role"] in ("user", "assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"][:3000]})
    if hits:
        user = f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    else:
        user = question
    msgs.append({"role": "user", "content": user})
    return msgs


def build_condense_messages(question: str, history: list[dict]) -> list[dict]:
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
        for m in history[-4:] if m.get("content")
    )
    return [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast message: {question}\n\nStandalone question:"},
    ]
