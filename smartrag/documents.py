"""Read uploaded files into page- or section-aware Documents, with clear errors."""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv")


class IngestError(Exception):
    """A file could not be read. The message is safe to show to the user."""


@dataclass
class LoadedFile:
    name: str
    sha256: str
    size_bytes: int
    docs: list[Document]          # one per page / section / row-block

    @property
    def pages(self) -> int:
        nums = {d.metadata.get("page") for d in self.docs if d.metadata.get("page")}
        return len(nums)

    @property
    def characters(self) -> int:
        return sum(len(d.page_content) for d in self.docs)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # re-join words hyphenated across lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("latin-1", errors="replace")


def _sections(text: str, name: str, heading_re: re.Pattern) -> list[Document]:
    """Split text at headings so citations can say which section an answer came from."""
    docs: list[Document] = []
    title, buf = None, []

    def flush():
        body = _clean("\n".join(buf))
        if body:
            meta = {"source": name}
            if title:
                meta["section"] = title
            docs.append(Document(page_content=body, metadata=meta))

    for line in text.split("\n"):
        m = heading_re.match(line)
        if m:
            flush()
            title, buf = m.group(1).strip()[:90], [line]
        else:
            buf.append(line)
    flush()
    return docs


def _load_pdf(name: str, data: bytes) -> list[Document]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise IngestError(f"{name} is password-protected. Remove the password and upload it again.")
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = _clean(page.extract_text() or "")
            except Exception:
                text = ""
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "page": i}))
    except IngestError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError, RecursionError):
        raise IngestError(f"{name} could not be read. It may be damaged.") from None
    if sum(len(d.page_content) for d in docs) < 20:
        raise IngestError(
            f"{name} has no selectable text. It looks like a scanned image, and scanned PDFs "
            "need OCR, which this app does not do yet."
        )
    return docs


def _load_docx(name: str, data: bytes) -> list[Document]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception:
        raise IngestError(f"{name} could not be read. Is it a valid .docx file?") from None
    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue
        style = (para.style.name or "") if para.style is not None else ""
        lines.append(f"# {text}" if style.lower().startswith(("heading", "title")) else text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    docs = _sections("\n".join(lines), name, re.compile(r"^#\s+(.+)$"))
    if not docs:
        raise IngestError(f"{name} contains no text.")
    return docs


def _load_text(name: str, data: bytes, markdown: bool) -> list[Document]:
    text = _decode(data)
    if not text.strip():
        raise IngestError(f"{name} is empty.")
    pattern = re.compile(r"^#{1,4}\s+(.+?)\s*#*\s*$") if markdown else re.compile(r"^\s*#\s+(.+)$")
    docs = _sections(text, name, pattern) if markdown else [Document(page_content=_clean(text), metadata={"source": name})]
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise IngestError(f"{name} contains no text.")
    return docs


def _load_csv(name: str, data: bytes) -> list[Document]:
    text = _decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise IngestError(f"{name} has no data rows.")
    header, body = rows[0], rows[1:]
    docs, block = [], 25
    for start in range(0, len(body), block):
        part = body[start:start + block]
        lines = ["; ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(header, r) if v.strip()) for r in part]
        docs.append(Document(
            page_content="\n".join(lines),
            metadata={"source": name, "section": f"rows {start + 1} to {start + len(part)}"},
        ))
    return docs


def load_file(name: str, data: bytes, max_bytes: int) -> LoadedFile:
    """Turn raw upload bytes into Documents. Raises IngestError with a friendly message."""
    safe_name = PurePath(name).name or "document"
    ext = PurePath(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"{safe_name}: unsupported file type. Use PDF, DOCX, TXT, MD or CSV.")
    if not data:
        raise IngestError(f"{safe_name} is empty.")
    if len(data) > max_bytes:
        raise IngestError(f"{safe_name} is larger than the {max_bytes // (1024 * 1024)} MB limit.")

    if ext == ".pdf":
        docs = _load_pdf(safe_name, data)
    elif ext == ".docx":
        docs = _load_docx(safe_name, data)
    elif ext == ".csv":
        docs = _load_csv(safe_name, data)
    else:
        docs = _load_text(safe_name, data, markdown=(ext == ".md"))
    return LoadedFile(safe_name, hashlib.sha256(data).hexdigest(), len(data), docs)
