"""Pinecone vector database integration."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

from src.config import AppConfig
from src.exceptions import PineconeConfigurationError, PineconeOperationError
from src.models import RetrievalResult, TextChunk


class PineconeVectorStore:
    """Create, populate, and query a Pinecone cosine-similarity index."""

    def __init__(self, config: AppConfig) -> None:
        if not config.pinecone_api_key:
            raise PineconeConfigurationError("PINECONE_API_KEY is not configured.")

        self.config = config
        self.index_name = config.pinecone_index_name
        self.metric = "cosine"

        try:
            from pinecone import Pinecone

            self.pc = Pinecone(api_key=config.pinecone_api_key)
        except Exception as exc:
            raise PineconeOperationError("Could not initialize Pinecone client.") from exc

    def ensure_index(self, dimension: int) -> None:
        """Create the Pinecone index when it does not already exist."""
        try:
            from pinecone import ServerlessSpec

            if not self._index_exists():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.pinecone_cloud,
                        region=self.config.pinecone_region,
                    ),
                    deletion_protection="disabled",
                )
                self._wait_until_ready()
            else:
                self._validate_existing_index(dimension)
        except PineconeOperationError:
            raise
        except Exception as exc:
            raise PineconeOperationError("Pinecone index creation/check failed.") from exc

    def upsert_chunks(
        self,
        chunks: List[TextChunk],
        vectors: List[List[float]],
        namespace: str,
        batch_size: int = 100,
    ) -> int:
        if len(chunks) != len(vectors):
            raise PineconeOperationError("Chunk count and vector count do not match.")

        index = self._index()
        upserted = 0
        try:
            for start in range(0, len(chunks), batch_size):
                batch = []
                for chunk, vector in zip(
                    chunks[start : start + batch_size], vectors[start : start + batch_size]
                ):
                    batch.append(
                        {
                            "id": chunk.id,
                            "values": vector,
                            "metadata": chunk.to_metadata(),
                        }
                    )
                if batch:
                    index.upsert(vectors=batch, namespace=namespace)
                    upserted += len(batch)
        except Exception as exc:
            raise PineconeOperationError("Vector upsert to Pinecone failed.") from exc

        return upserted

    def query(
        self,
        vector: List[float],
        namespace: str,
        top_k: int,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[RetrievalResult]:
        index = self._index()
        try:
            response = index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=metadata_filter or None,
            )
        except Exception as exc:
            raise PineconeOperationError("Pinecone query failed.") from exc

        return self._parse_matches(response)

    def describe_stats(self) -> Dict[str, object]:
        try:
            return dict(self._index().describe_index_stats())
        except Exception as exc:
            raise PineconeOperationError("Could not read Pinecone index stats.") from exc

    def _index(self):
        try:
            return self.pc.Index(self.index_name)
        except Exception as exc:
            raise PineconeOperationError("Could not connect to the Pinecone index.") from exc

    def _index_exists(self) -> bool:
        if hasattr(self.pc, "has_index"):
            return bool(self.pc.has_index(self.index_name))

        indexes = self.pc.list_indexes()
        if hasattr(indexes, "names"):
            return self.index_name in indexes.names()
        return self.index_name in [item.get("name") for item in indexes]

    def _wait_until_ready(self, timeout_seconds: int = 120) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            description = self.pc.describe_index(self.index_name)
            status = self._read_attr(description, "status", {})
            ready = self._read_attr(status, "ready", False)
            if ready:
                return
            time.sleep(2)
        raise PineconeOperationError(
            f"Pinecone index `{self.index_name}` was not ready after {timeout_seconds}s."
        )

    def _validate_existing_index(self, dimension: int) -> None:
        description = self.pc.describe_index(self.index_name)
        existing_dimension = self._read_attr(description, "dimension", None)
        existing_metric = self._read_attr(description, "metric", None)

        if existing_dimension and int(existing_dimension) != int(dimension):
            raise PineconeOperationError(
                f"Existing Pinecone index dimension is {existing_dimension}, "
                f"but embedding model dimension is {dimension}. Use another index name."
            )
        if existing_metric and str(existing_metric).lower() != self.metric:
            raise PineconeOperationError(
                f"Existing Pinecone index metric is `{existing_metric}`, not cosine."
            )

    @staticmethod
    def _parse_matches(response: object) -> List[RetrievalResult]:
        matches = PineconeVectorStore._read_attr(response, "matches", [])
        parsed: List[RetrievalResult] = []

        for match in matches or []:
            metadata = PineconeVectorStore._read_attr(match, "metadata", {}) or {}
            match_id = str(PineconeVectorStore._read_attr(match, "id", ""))
            score = float(PineconeVectorStore._read_attr(match, "score", 0.0) or 0.0)
            text = str(metadata.get("text", ""))
            parsed.append(
                RetrievalResult(
                    id=match_id,
                    score=score,
                    text=text,
                    metadata=dict(metadata),
                )
            )

        return parsed

    @staticmethod
    def _read_attr(value: object, name: str, default: object = None) -> object:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)


def build_metadata_filter(
    document_ids: Optional[Iterable[str]] = None,
    page_number: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    filters: Dict[str, object] = {}

    ids = [doc_id for doc_id in document_ids or [] if doc_id]
    if ids:
        filters["document_id"] = {"$in": ids}
    if page_number is not None and page_number > 0:
        filters["page_number"] = {"$eq": int(page_number)}

    return filters or None

