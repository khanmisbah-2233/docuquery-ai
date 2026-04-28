"""Streamlit interface for the DocuQuery AI RAG system."""

from __future__ import annotations

from dataclasses import replace
from textwrap import shorten

import streamlit as st

from src.config import MAX_UPLOAD_SIZE_MB, get_config, sanitize_namespace
from src.embeddings import EmbeddingService
from src.exceptions import DocuQueryError
from src.models import GeneratedAnswer, IngestionReport, UploadedPDF
from src.pinecone_store import PineconeVectorStore
from src.pipeline import RAGPipeline


st.set_page_config(
    page_title="DocuQuery AI",
    page_icon="DQ",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_embedding_service(model_name: str) -> EmbeddingService:
    return EmbeddingService(model_name)


def build_pipeline(config) -> RAGPipeline:
    embeddings = get_embedding_service(config.embedding_model_name)
    vector_store = PineconeVectorStore(config)
    return RAGPipeline(config=config, embeddings=embeddings, vector_store=vector_store)


def initialize_state() -> None:
    st.session_state.setdefault("query_history", [])
    st.session_state.setdefault("last_answer", None)
    st.session_state.setdefault("last_report", None)
    st.session_state.setdefault("indexed_documents", {})


def render_sources(answer: GeneratedAnswer) -> None:
    if not answer.sources:
        st.info("No source chunks passed the selected similarity threshold.")
        return

    for position, source in enumerate(answer.sources, start=1):
        title = (
            f"Source {position}: {source.document_name} | "
            f"Page {source.page_number} | Score {source.score:.4f}"
        )
        with st.expander(title, expanded=position == 1):
            st.caption(f"Chunk ID: {source.chunk_id}")
            st.write(source.text)


def render_history() -> None:
    history = st.session_state.get("query_history", [])
    if not history:
        st.info("No questions asked in this session yet.")
        return

    for item in reversed(history[-10:]):
        with st.expander(item["query"], expanded=False):
            st.write(item["answer"])
            st.caption(
                f"Namespace: {item['namespace']} | "
                f"Confidence: {item['confidence']:.4f} | "
                f"Sources: {item['source_count']}"
            )


def main() -> None:
    initialize_state()
    base_config = get_config()

    st.title("DocuQuery AI")

    with st.sidebar:
        st.header("Configuration")
        namespace = sanitize_namespace(
            st.text_input("Pinecone namespace", value=base_config.pinecone_namespace)
        )
        chunk_size = st.slider("Chunk size (words)", 200, 1200, base_config.chunk_size, 50)
        chunk_overlap = st.slider(
            "Chunk overlap (words)",
            0,
            min(300, chunk_size // 2),
            min(base_config.chunk_overlap, min(300, chunk_size // 2)),
            10,
        )
        top_k = st.slider("Top-k retrieval", 1, 20, base_config.top_k)
        threshold = st.slider(
            "Similarity threshold",
            0.0,
            1.0,
            float(base_config.similarity_threshold),
            0.01,
        )
        page_filter_enabled = st.checkbox("Filter by page number")
        page_number = None
        if page_filter_enabled:
            page_number = st.number_input("Page number", min_value=1, step=1, value=1)

        provider = base_config.llm_provider
        model = base_config.llm_model

    config = replace(
        base_config,
        pinecone_namespace=namespace,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        similarity_threshold=threshold,
        llm_provider=provider,
        llm_model=model.strip() or base_config.llm_model,
    )

    ingest_tab, ask_tab, history_tab, ops_tab = st.tabs(
        ["Ingest PDFs", "Ask Questions", "Query History", "System Status"]
    )

    with ingest_tab:
        st.subheader("Upload and Index")
        uploaded_files = st.file_uploader(
            f"Upload one or more PDF files up to {MAX_UPLOAD_SIZE_MB} MB each",
            type=["pdf"],
            accept_multiple_files=True,
        )
        st.caption(
            "Each chunk is stored in Pinecone with document name, page number, chunk ID, "
            "document ID, and chunk text metadata."
        )

        if st.button("Index documents in Pinecone", type="primary", disabled=not uploaded_files):
            try:
                pdfs = [
                    UploadedPDF(name=file.name, content=file.getvalue())
                    for file in uploaded_files
                ]
                with st.spinner("Extracting text, embedding chunks, and upserting to Pinecone..."):
                    report = build_pipeline(config).ingest(pdfs, namespace=namespace)
                st.session_state["last_report"] = report
                st.session_state["indexed_documents"].update(report.documents)
                st.success(
                    f"Indexed {report.chunk_count} chunks from {report.page_count} pages "
                    f"into namespace `{report.namespace}`."
                )
            except DocuQueryError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected ingestion error: {exc}")

        report: IngestionReport | None = st.session_state.get("last_report")
        if report:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documents", len(report.documents))
            c2.metric("Pages", report.page_count)
            c3.metric("Chunks", report.chunk_count)
            c4.metric("Namespace", report.namespace)
            st.write("Indexed documents")
            st.dataframe(
                [{"document_id": doc_id, "document_name": name} for doc_id, name in report.documents.items()],
                use_container_width=True,
                hide_index=True,
            )

    with ask_tab:
        st.subheader("Ask From Indexed PDF Content")
        indexed_documents = st.session_state.get("indexed_documents", {})
        selected_document_names = []
        if indexed_documents:
            name_by_id = {doc_id: name for doc_id, name in indexed_documents.items()}
            selected_document_names = st.multiselect(
                "Optional document filter",
                options=list(name_by_id.values()),
                default=list(name_by_id.values()),
            )
            selected_document_ids = [
                doc_id for doc_id, name in name_by_id.items() if name in selected_document_names
            ]
        else:
            selected_document_ids = []
            st.info(
                "No documents were indexed in this browser session. You can still query an "
                "existing Pinecone namespace if it already contains compatible vectors."
            )

        query = st.text_area(
            "Question",
            placeholder="Example: Define Data Heterogeneity and explain its key points.",
            height=110,
        )

        if st.button("Answer from PDF", type="primary"):
            try:
                with st.spinner("Retrieving context and generating an answer..."):
                    answer = build_pipeline(config).answer(
                        query=query,
                        namespace=namespace,
                        top_k=top_k,
                        similarity_threshold=threshold,
                        document_ids=selected_document_ids or None,
                        page_number=int(page_number) if page_number else None,
                    )
                st.session_state["last_answer"] = answer
                st.session_state["query_history"].append(
                    {
                        "query": query.strip(),
                        "answer": answer.answer,
                        "confidence": answer.confidence_score,
                        "source_count": len(answer.sources),
                        "namespace": namespace,
                    }
                )
            except DocuQueryError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected query error: {exc}")

        answer: GeneratedAnswer | None = st.session_state.get("last_answer")
        if answer:
            st.markdown("#### Answer")
            st.write(answer.answer)
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{answer.confidence_score:.4f}")
            c2.metric("Sources Used", len(answer.sources))
            c3.metric("Grounded", "Yes" if answer.is_supported else "No")
            st.markdown("#### Source Attribution")
            render_sources(answer)

    with history_tab:
        st.subheader("Session Query History")
        render_history()

    with ops_tab:
        st.subheader("System Status")
        st.write(
            "The application creates a Pinecone cosine index when needed, uses a namespace "
            "for each document collection, stores metadata on every vector, filters by "
            "document/page when requested, and logs user queries to `logs/query_log.csv`."
        )
        if st.button("Show Pinecone index stats"):
            try:
                stats = build_pipeline(config).vector_store.describe_stats()
                st.json(stats)
            except DocuQueryError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not fetch stats: {exc}")

        last_answer: GeneratedAnswer | None = st.session_state.get("last_answer")
        if last_answer and last_answer.sources:
            st.write("Current best source excerpt")
            best = max(last_answer.sources, key=lambda source: source.score)
            st.info(shorten(best.text, width=500, placeholder="..."))


if __name__ == "__main__":
    main()
