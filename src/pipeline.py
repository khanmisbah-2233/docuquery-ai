"""High-level orchestration for ingestion and question answering."""

from __future__ import annotations

from typing import Iterable, List, Optional

from src.config import AppConfig
from src.embeddings import EmbeddingService
from src.generator import AnswerGenerator
from src.logger import QueryLogger
from src.models import GeneratedAnswer, IngestionReport, PDFPage, UploadedPDF
from src.pdf_loader import PDFLoader
from src.pinecone_store import PineconeVectorStore
from src.retriever import Retriever
from src.text_processor import chunk_pages


class RAGPipeline:
    """End-to-end PDF ingestion, retrieval, and grounded generation."""

    def __init__(
        self,
        config: AppConfig,
        embeddings: EmbeddingService,
        vector_store: PineconeVectorStore,
    ) -> None:
        self.config = config
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.loader = PDFLoader()
        self.retriever = Retriever(embeddings, vector_store)
        self.generator = AnswerGenerator(config)
        self.query_logger = QueryLogger(config.query_log_path)

    def ingest(self, uploaded_pdfs: Iterable[UploadedPDF], namespace: str) -> IngestionReport:
        pages: List[PDFPage] = []
        documents = {}

        for uploaded_pdf in uploaded_pdfs:
            loaded_pages = self.loader.load(uploaded_pdf)
            pages.extend(loaded_pages)
            if loaded_pages:
                documents[loaded_pages[0].document_id] = uploaded_pdf.name

        chunks = chunk_pages(
            pages,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        vectors = self.embeddings.embed_texts(chunk.text for chunk in chunks)
        self.vector_store.ensure_index(self.embeddings.dimension)
        self.vector_store.upsert_chunks(chunks, vectors, namespace=namespace)

        return IngestionReport(
            namespace=namespace,
            index_name=self.config.pinecone_index_name,
            embedding_model=self.config.embedding_model_name,
            documents=documents,
            page_count=len(pages),
            chunk_count=len(chunks),
        )

    def answer(
        self,
        query: str,
        namespace: str,
        top_k: int,
        similarity_threshold: float,
        document_ids: Optional[Iterable[str]] = None,
        page_number: Optional[int] = None,
        answer_mode: str = "Detailed explanation",
    ) -> GeneratedAnswer:
        self.vector_store.ensure_index(self.embeddings.dimension)
        sources = self.retriever.retrieve(
            query=query,
            namespace=namespace,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
            page_number=page_number,
        )
        answer = self.generator.generate(
            query=query,
            sources=sources,
            answer_mode=answer_mode,
        )
        self.query_logger.log(
            namespace=namespace,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            answer=answer,
        )
        return answer
