"""Text cleanup and intelligent chunking utilities."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

from src.models import PDFPage, TextChunk


def clean_text(text: str) -> str:
    """Remove common PDF extraction artifacts while preserving paragraphs."""
    if not text:
        return ""

    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\x00": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _semantic_units(text: str) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    units: List[str] = []
    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    for paragraph in paragraphs:
        parts = [part.strip() for part in sentence_pattern.split(paragraph) if part.strip()]
        units.extend(parts or [paragraph])

    return units


def _window_words(words: List[str], chunk_size: int, overlap: int) -> Iterable[str]:
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            yield " ".join(window)
        if start + chunk_size >= len(words):
            break


def _chunk_id(document_id: str, page_number: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}-p{page_number}-c{chunk_index}-{digest}"


def chunk_pages(
    pages: Iterable[PDFPage],
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[TextChunk]:
    """Create sentence-aware, page-traceable chunks."""
    chunk_size = max(100, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks: List[TextChunk] = []

    for page in pages:
        units = _semantic_units(page.text)
        current_words: List[str] = []
        page_chunk_index = 0

        def emit_current() -> None:
            nonlocal current_words, page_chunk_index
            if not current_words:
                return
            text = " ".join(current_words).strip()
            if not text:
                current_words = []
                return
            page_chunk_index += 1
            chunks.append(
                TextChunk(
                    id=_chunk_id(page.document_id, page.page_number, page_chunk_index, text),
                    document_id=page.document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_index=page_chunk_index,
                    text=text,
                )
            )
            current_words = current_words[-overlap:] if overlap else []

        for unit in units:
            words = unit.split()
            if not words:
                continue

            if len(words) > chunk_size:
                emit_current()
                for window in _window_words(words, chunk_size, overlap):
                    page_chunk_index += 1
                    chunks.append(
                        TextChunk(
                            id=_chunk_id(
                                page.document_id, page.page_number, page_chunk_index, window
                            ),
                            document_id=page.document_id,
                            document_name=page.document_name,
                            page_number=page.page_number,
                            chunk_index=page_chunk_index,
                            text=window,
                        )
                    )
                current_words = []
                continue

            if current_words and len(current_words) + len(words) > chunk_size:
                emit_current()

            current_words.extend(words)

        emit_current()

    return chunks

