from src.models import PDFPage
from src.text_processor import chunk_pages, clean_text


def test_clean_text_removes_line_artifacts():
    raw = "Retrieval-\nAugmented\nGeneration   systems\n\nuse context."
    assert clean_text(raw) == "RetrievalAugmented Generation systems\n\nuse context."


def test_chunk_pages_keeps_page_metadata():
    page = PDFPage(
        document_id="doc123",
        document_name="sample.pdf",
        page_number=7,
        text="Sentence one. Sentence two. Sentence three.",
    )

    chunks = chunk_pages([page], chunk_size=100, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].document_name == "sample.pdf"
    assert chunks[0].page_number == 7
    assert chunks[0].id.startswith("doc123-p7-c1")

