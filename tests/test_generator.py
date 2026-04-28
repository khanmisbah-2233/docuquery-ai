from src.config import get_config
from src.generator import ANSWER_MODE_EXACT, AnswerGenerator
from src.models import RetrievalResult


def test_exact_pdf_text_answer_uses_source_text_without_llm():
    source = RetrievalResult(
        id="chunk-1",
        score=0.91,
        text="Data Heterogeneity: Devices collect data in a non-identically distributed manner.",
        metadata={
            "document_name": "paper.pdf",
            "page_number": 2,
            "chunk_id": "chunk-1",
        },
    )

    answer = AnswerGenerator(get_config()).generate(
        query="Define Data Heterogeneity",
        sources=[source],
        answer_mode=ANSWER_MODE_EXACT,
    )

    assert answer.is_supported
    assert "paper.pdf, page 2" in answer.answer
    assert source.text in answer.answer

