"""Split Documents into overlapping chunks that remember where they came from."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    section: str | None = None

    @property
    def label(self) -> str:
        """Short human-readable location, e.g. 'report.pdf, p. 3'."""
        if self.page:
            return f"{self.source}, p. {self.page}"
        if self.section:
            return f"{self.source}, {self.section}"
        return self.source


def chunk_documents(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "। ", " ", ""],   # "।" is the Bengali full stop
    )
    chunks: list[Chunk] = []
    for doc in docs:
        parts = splitter.split_text(doc.page_content)
        for part in parts:
            part = part.strip()
            if len(part) < 25 and len(parts) > 1:
                continue                       # drop stray fragments, keep a lone short chunk
            if part:
                chunks.append(Chunk(
                    text=part,
                    source=doc.metadata.get("source", "document"),
                    page=doc.metadata.get("page"),
                    section=doc.metadata.get("section"),
                ))
    return chunks
