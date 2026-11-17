"""Recursive character chunking with overlap.

Splits on the most "semantic" separator available (paragraph -> line -> sentence -> word)
so chunks respect document structure, then adds overlap so facts straddling a boundary
are still retrievable.
"""

from __future__ import annotations

from docmind.models import Chunk, Document

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) > size:
            pieces.extend(_split(part, size, rest))
            current = ""
        else:
            current = part
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    pieces = [p.strip() for p in _split(text.strip(), size, SEPARATORS) if p.strip()]
    if overlap == 0 or len(pieces) <= 1:
        return pieces
    # Prepend the tail of the previous piece so boundary facts appear in both chunks.
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:], strict=False):
        tail = prev[-overlap:]
        cut = tail.find(" ")
        if cut != -1:
            tail = tail[cut + 1 :]
        out.append((tail + " " + cur).strip())
    return out


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    return [
        Chunk(id=f"{doc.id}-{i}", doc_id=doc.id, doc_title=doc.title, text=t, index=i)
        for i, t in enumerate(chunk_text(doc.text, size, overlap))
    ]
