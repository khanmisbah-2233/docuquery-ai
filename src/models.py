"""Shared data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class UploadedPDF:
    name: str
    content: bytes


@dataclass(frozen=True)
class PDFPage:
    document_id: str
    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_id": self.id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "word_count": len(self.text.split()),
        }


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

    @property
    def document_name(self) -> str:
        return str(self.metadata.get("document_name", "Unknown document"))

    @property
    def page_number(self) -> int:
        try:
            return int(self.metadata.get("page_number", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", self.id))


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    is_supported: bool
    confidence_score: float
    query: str
    sources: List[RetrievalResult] = field(default_factory=list)


@dataclass(frozen=True)
class IngestionReport:
    namespace: str
    index_name: str
    embedding_model: str
    documents: Dict[str, str]
    page_count: int
    chunk_count: int

