"""Greetings, thanks and 'what can you do?' deserve a friendly reply, not a 'not in your documents' refusal."""
from __future__ import annotations

import re

_PATTERNS = re.compile(
    r"\b(hi+|hello+|hey+|hola|yo|sup|salam|salaam|assalamu\s*alaikum|assalamualaikum|good\s+(morning|afternoon|evening|night)|"
    r"namaste|how\s+are\s+(you|u)|how\s+do\s+you\s+do|how'?s\s+it\s+going|what'?s\s+up|thanks|thank\s+you|thx|ty|"
    r"bye|goodbye|see\s+you|who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"can\s+you\s+help(\s+me)?|help(\s+me)?|how\s+(do|does)\s+(this|it)\s+work|how\s+do\s+i\s+use\s+(this|you|it))\b"
    r"|সালাম|আসসালামু\s*আলাইকুম|হাই|হ্যালো|ধন্যবাদ|কেমন\s*আছ",
    re.I,
)
_FILLER = frozenset(
    "bro buddy dude man friend dear sir mam madam there please pls today you u are is a the doing going it me to this "
    "that so very much a lot all good great nice ok okay cool how what who do does can i my your".split()
)

OFFLINE_REPLY = (
    "Hello! I'm **Smart RAG Chatbot**. Upload a PDF, Word, Markdown, text or CSV file in the sidebar and ask me "
    "about it. I'll answer with citations to the exact passage. (I'm in offline mode right now, so I show matching "
    "passages instead of writing answers. Add a free `GROQ_API_KEY` to turn AI answers on.)"
)

CHAT_SYSTEM = (
    "You are Smart RAG Chatbot, a friendly assistant that answers questions about documents the user uploads and "
    "cites the exact passage. Reply to the user's message warmly and briefly, in their language, in one to three "
    "sentences. If they ask what you can do, explain that they can upload PDF, Word, Markdown, text or CSV files and "
    "ask questions about them. {docs}Do not invent facts about the documents."
)


def is_smalltalk(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 90 or not _PATTERNS.search(t):
        return False
    rest = _PATTERNS.sub(" ", t.lower())
    words = [w for w in re.findall(r"\w+", rest) if w not in _FILLER]
    return len(words) <= 1                    # anything with real content is a question, not small talk


def chat_system_prompt(files: list[str]) -> str:
    docs = f"The user has uploaded: {', '.join(files)}. " if files else "No documents are uploaded yet. "
    return CHAT_SYSTEM.format(docs=docs)
