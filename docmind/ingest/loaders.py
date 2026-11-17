"""Turn files (txt, md, pdf) into ``Document`` objects."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docmind.models import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _doc_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}::{text[:2000]}".encode()).hexdigest()[:12]


def load_text(text: str, title: str, source: str = "inline") -> Document:
    text = text.strip()
    if not text:
        raise ValueError("Document text is empty")
    return Document(id=_doc_id(source, text), title=title, source=source, text=text)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily: pypdf is only needed for PDFs

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def load_file(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix} (supported: {sorted(SUPPORTED_SUFFIXES)})")
    if path.suffix.lower() == ".pdf":
        text = _read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return load_text(text, title=path.stem.replace("_", " ").replace("-", " ").title(), source=str(path))


def load_bytes(data: bytes, filename: str) -> Document:
    """Load an uploaded file held in memory (used by the API)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((p.extract_text() or "").strip() for p in reader.pages)
    elif suffix in SUPPORTED_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return load_text(text, title=Path(filename).stem, source=filename)
