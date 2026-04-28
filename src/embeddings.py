"""Embedding generation using Sentence Transformers."""

from __future__ import annotations

from functools import cached_property
from typing import Iterable, List

from src.exceptions import EmbeddingModelError


class EmbeddingService:
    """Generate normalized dense embeddings for chunks and queries."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        try:
            return SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model `{self.model_name}`."
            ) from exc

    @cached_property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if not dimension:
            sample_vector = self.embed_query("dimension probe")
            dimension = len(sample_vector)
        return int(dimension)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = [text for text in texts if text and text.strip()]
        if not text_list:
            return []

        try:
            vectors = self.model.encode(
                text_list,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingModelError("Embedding generation failed.") from exc

        return [vector.astype(float).tolist() for vector in vectors]

    def embed_query(self, query: str) -> List[float]:
        vectors = self.embed_texts([query])
        if not vectors:
            raise EmbeddingModelError("Cannot embed an empty query.")
        return vectors[0]

