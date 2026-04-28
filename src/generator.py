"""Grounded answer generation using an LLM API."""

from __future__ import annotations

from typing import List, Tuple

from src.config import AppConfig
from src.exceptions import LLMGenerationError
from src.models import GeneratedAnswer, RetrievalResult


FALLBACK_ANSWER = "The answer is not available in the provided document."


ANSWER_MODE_DETAILED = "Detailed explanation"
ANSWER_MODE_EXACT = "Exact PDF text"


class AnswerGenerator:
    """Generate answers strictly from retrieved PDF context."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate(
        self,
        query: str,
        sources: List[RetrievalResult],
        answer_mode: str = ANSWER_MODE_DETAILED,
    ) -> GeneratedAnswer:
        if not sources:
            return GeneratedAnswer(
                answer=FALLBACK_ANSWER,
                is_supported=False,
                confidence_score=0.0,
                query=query,
                sources=[],
            )

        if answer_mode == ANSWER_MODE_EXACT:
            return self._exact_pdf_text_answer(query, sources)

        if self.config.llm_provider == "offline":
            return self._offline_answer(query, sources)

        answer = self._call_llm(query, sources).strip()
        if not answer:
            answer = FALLBACK_ANSWER

        is_supported = answer.strip().lower() != FALLBACK_ANSWER.lower()
        confidence = max(source.score for source in sources) if is_supported else 0.0
        return GeneratedAnswer(
            answer=answer,
            is_supported=is_supported,
            confidence_score=round(float(confidence), 4),
            query=query,
            sources=sources,
        )

    def _call_llm(self, query: str, sources: List[RetrievalResult]) -> str:
        api_key, base_url = self._resolve_provider()
        if not api_key:
            raise LLMGenerationError(
                f"Missing API key for `{self.config.llm_provider}` LLM provider."
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=self.config.llm_model,
                temperature=0,
                max_tokens=1100,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional PDF research assistant. "
                            "Answer only from the provided PDF context. "
                            "If the context does not contain the answer, reply exactly: "
                            f"{FALLBACK_ANSWER} "
                            "Do not use outside knowledge. "
                            "When the answer is supported, cover every relevant point found "
                            "in the context, explain definitions clearly, and cite page "
                            "numbers like (p. 3). If multiple context chunks contain useful "
                            "details, combine them into one coherent answer without inventing "
                            "missing details."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(query, sources),
                    },
                ],
            )
        except Exception as exc:
            raise LLMGenerationError("LLM response generation failed.") from exc

        return response.choices[0].message.content or ""

    def _resolve_provider(self) -> Tuple[str, str | None]:
        provider = self.config.llm_provider.lower()
        if provider == "groq":
            return self.config.groq_api_key or "", "https://api.groq.com/openai/v1"
        if provider == "openai":
            return self.config.openai_api_key or "", None
        if provider == "custom":
            return self.config.llm_api_key or "", self.config.llm_base_url
        raise LLMGenerationError(
            "LLM_PROVIDER must be one of: groq, openai, custom, offline."
        )

    @staticmethod
    def _build_prompt(query: str, sources: List[RetrievalResult]) -> str:
        context_blocks = []
        for index, source in enumerate(sources, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Document: {source.document_name}",
                        f"Page: {source.page_number}",
                        f"Similarity: {source.score:.4f}",
                        f"Text: {source.text}",
                    ]
                )
            )

        joined_context = "\n\n".join(context_blocks)
        return (
            f"Question: {query.strip()}\n\n"
            "Context:\n"
            f"{joined_context}\n\n"
            "Answer using only the context above. If the question asks about a topic, "
            "include the definition, main explanation, causes/effects, examples, and "
            "limitations only when those details are present in the context."
        )

    @staticmethod
    def _offline_answer(query: str, sources: List[RetrievalResult]) -> GeneratedAnswer:
        best_source = max(sources, key=lambda source: source.score)
        excerpt = best_source.text[:900].strip()
        answer = (
            "LLM provider is set to offline, so no generative API call was made. "
            f"The most relevant document excerpt is from page {best_source.page_number}: "
            f"{excerpt}"
        )
        return GeneratedAnswer(
            answer=answer,
            is_supported=True,
            confidence_score=round(float(best_source.score), 4),
            query=query,
            sources=sources,
        )

    @staticmethod
    def _exact_pdf_text_answer(
        query: str, sources: List[RetrievalResult]
    ) -> GeneratedAnswer:
        excerpts = []
        for index, source in enumerate(sources, start=1):
            excerpts.append(
                "\n".join(
                    [
                        f"Excerpt {index} - {source.document_name}, page {source.page_number}:",
                        source.text.strip(),
                    ]
                )
            )

        confidence = max(source.score for source in sources)
        return GeneratedAnswer(
            answer="\n\n".join(excerpts),
            is_supported=True,
            confidence_score=round(float(confidence), 4),
            query=query,
            sources=sources,
        )
