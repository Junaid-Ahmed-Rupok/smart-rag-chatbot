"""Turn a conversation into a Markdown transcript with its sources."""
from __future__ import annotations

from datetime import datetime


def chat_to_markdown(messages: list[dict], files: list[str], app_name: str = "Smart RAG Chatbot") -> str:
    out = [f"# {app_name} transcript", f"*Exported {datetime.now():%Y-%m-%d %H:%M}*", ""]
    if files:
        out += ["**Documents:** " + ", ".join(files), ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        out += [f"## {who}", "", m["content"], ""]
        used = [c for c in m.get("citations", []) if c.get("used")]
        if used:
            out.append("**Sources**")
            for c in used:
                snippet = " ".join(c["text"].split())[:240]
                out.append(f"- [{c['n']}] {c['label']}: \"{snippet}…\"")
            out.append("")
    return "\n".join(out)
