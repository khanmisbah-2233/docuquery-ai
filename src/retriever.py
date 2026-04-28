"""Semantic retrieval over Pinecone results."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from src.embeddings import EmbeddingService
from src.exceptions import AmbiguousQueryError, EmptyQueryError
from src.models import RetrievalResult
from src.pinecone_store import PineconeVectorStore, build_metadata_filter


class Retriever:
    """Embed a query, search Pinecone, and apply score filtering."""

    _GENERIC_TERMS = {
        "a",
        "an",
        "and",
        "answer",
        "are",
        "define",
        "describe",
        "explain",
        "for",
        "from",
        "give",
        "in",
        "is",
        "it",
        "me",
        "of",
        "please",
        "tell",
        "that",
        "the",
        "this",
        "topic",
        "what",
        "when",
        "where",
        "why",
        "how",
    }

    _QUERY_PREFIXES = [
        "definition of",
        "what is",
        "explain",
        "describe",
        "key points about",
    ]

    def __init__(self, embeddings: EmbeddingService, vector_store: PineconeVectorStore) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35,
        document_ids: Optional[Iterable[str]] = None,
        page_number: Optional[int] = None,
    ) -> List[RetrievalResult]:
        clean_query = query.strip()
        if not clean_query:
            raise EmptyQueryError("Please enter a question before searching.")
        if self._is_ambiguous(clean_query):
            raise AmbiguousQueryError(
                "Please include the exact topic or term from the PDF. "
                "Example: `Define Data Heterogeneity`."
            )

        metadata_filter = build_metadata_filter(document_ids, page_number)
        candidate_k = max(top_k * 4, 20)
        raw_results = self._retrieve_candidates(
            query=clean_query,
            namespace=namespace,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        threshold = min(max(similarity_threshold, 0.0), 1.0)
        filtered = [result for result in raw_results if result.score >= threshold]
        ranked = self._rank_results(clean_query, filtered)
        return ranked[:top_k]

    def _retrieve_candidates(
        self,
        query: str,
        namespace: str,
        top_k: int,
        metadata_filter: Optional[dict],
    ) -> List[RetrievalResult]:
        results_by_id = {}
        for variant in self._query_variants(query):
            query_vector = self.embeddings.embed_query(variant)
            results = self.vector_store.query(
                vector=query_vector,
                namespace=namespace,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            for result in results:
                current = results_by_id.get(result.id)
                if current is None or result.score > current.score:
                    results_by_id[result.id] = result

        return list(results_by_id.values())

    @classmethod
    def _is_ambiguous(cls, query: str) -> bool:
        return len(cls._meaningful_terms(query)) == 0

    @classmethod
    def _query_variants(cls, query: str) -> List[str]:
        topic = " ".join(cls._meaningful_terms(query))
        variants = [query]
        if topic:
            variants.append(topic)
            variants.extend(f"{prefix} {topic}" for prefix in cls._QUERY_PREFIXES)

        unique_variants = []
        seen = set()
        for variant in variants:
            normalized = variant.strip().lower()
            if normalized and normalized not in seen:
                unique_variants.append(variant.strip())
                seen.add(normalized)
        return unique_variants

    @classmethod
    def _rank_results(
        cls, query: str, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        query_terms = set(cls._meaningful_terms(query))

        def rank_score(result: RetrievalResult) -> float:
            text_terms = set(cls._meaningful_terms(result.text))
            if not query_terms:
                overlap = 0.0
            else:
                overlap = len(query_terms & text_terms) / len(query_terms)
            heading_boost = 0.04 if cls._looks_like_heading_match(query, result.text) else 0.0
            return result.score + (overlap * 0.08) + heading_boost

        return sorted(results, key=rank_score, reverse=True)

    @classmethod
    def _meaningful_terms(cls, value: str) -> List[str]:
        terms = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", value.lower())
        return [term for term in terms if term not in cls._GENERIC_TERMS]

    @classmethod
    def _looks_like_heading_match(cls, query: str, text: str) -> bool:
        topic = " ".join(cls._meaningful_terms(query))
        if not topic:
            return False
        first_words = " ".join(text.split()[:20]).lower()
        return topic in first_words
