# DocuQuery AI

Professional intermediate Retrieval-Augmented Generation (RAG) system for PDF question answering using Python, Streamlit, Sentence Transformers, Pinecone, and an OpenAI-compatible LLM API such as Groq or OpenAI.

The system accepts PDFs, extracts text, chunks page-traceable content, embeds chunks, stores vectors in Pinecone namespaces, retrieves relevant context with cosine similarity, and generates answers strictly from the retrieved document context.

## Requirement Coverage

| Assignment requirement | Implementation |
| --- | --- |
| PDF upload | Streamlit multi-file uploader in `app.py`, limited to 20 MB by `.streamlit/config.toml` and runtime validation |
| Text extraction | `src/pdf_loader.py` uses `pypdf.PdfReader` |
| Formatting cleanup | `src/text_processor.py::clean_text` removes common PDF artifacts |
| Intelligent chunking | Sentence/paragraph-aware chunking with overlap in `src/text_processor.py` |
| Embeddings | `src/embeddings.py` uses Sentence Transformers, default `sentence-transformers/all-MiniLM-L6-v2` |
| Pinecone index creation | `src/pinecone_store.py::ensure_index` creates a serverless cosine index |
| Namespace usage | Namespace is user-configurable in the sidebar and passed to upsert/query |
| Upserting vectors | `src/pinecone_store.py::upsert_chunks` batches chunk vectors |
| Querying vectors | `src/pinecone_store.py::query` searches by embedded query vector |
| Metadata management | Document name, document ID, page number, chunk ID, chunk index, text, and word count are stored |
| Top-k retrieval | Adjustable sidebar slider |
| Similarity threshold | Adjustable sidebar slider |
| Strict grounded answers | `src/generator.py` uses a strict prompt and fallback answer |
| Insufficient context fallback | Returns `The answer is not available in the provided document.` |
| Source attribution | UI displays page number, excerpt, similarity score, document name, and chunk ID |
| Error handling | Invalid PDF, empty PDF text, empty query, Pinecone config/connection, embedding, and LLM failures |
| Modular architecture | Separate modules for loader, processor, embeddings, Pinecone store, retriever, generator, logger, and pipeline |
| Environment variables | `.env.example` and `src/config.py` |

## Mandatory Enhancements Implemented

This project implements more than the required three enhancements:

- Multi-document support
- Query history in Streamlit session state
- Adjustable chunk size
- Adjustable chunk overlap
- Adjustable top-k retrieval
- Metadata filtering by page number
- Optional filtering by indexed document
- Confidence score display
- CSV logging of user queries in `logs/query_log.csv`

## Project Structure

```text
docuquery-ai/
  app.py
  requirements.txt
  .env.example
  .streamlit/config.toml
  src/
    config.py
    embeddings.py
    exceptions.py
    generator.py
    logger.py
    models.py
    pdf_loader.py
    pinecone_store.py
    pipeline.py
    retriever.py
    text_processor.py
  docs/
    architecture.md
    demo_script.md
  reports/
    technical_report.md
  tests/
    test_text_processor.py
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create the environment file.

```bash
copy .env.example .env
```

4. Fill in the required keys in `.env`.

```text
PINECONE_API_KEY=...
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

For OpenAI instead of Groq:

```text
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload one or more PDFs, index them, then ask questions from the indexed content.

## Streamlit Cloud Secrets

Do not upload `.env` or real API keys to GitHub. For Streamlit Cloud, open your app settings and add these values in **Secrets**:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "docuquery-rag"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_NAMESPACE = "course-assignment"

LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

The app reads local `.env` values first and then Streamlit secrets when deployed.

## Pinecone Configuration

The default Pinecone configuration is:

```text
PINECONE_INDEX_NAME=docuquery-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=course-assignment
```

The index is created automatically using cosine similarity and the embedding dimension reported by the configured Sentence Transformer model. If an index with the same name already exists, the application verifies that its dimension and metric match the embedding model.

## Hallucination Prevention

The system uses three safeguards:

- Retrieval threshold: chunks below the selected similarity score are rejected.
- No-context fallback: if no chunks pass the threshold, the exact required fallback message is returned.
- Strict LLM prompt: the generator instructs the model to answer only from retrieved context and to return the fallback message if the answer is absent.

## Source Attribution

Every answer displays:

- Document name
- Page number
- Chunk ID
- Relevant excerpt
- Similarity score

## Testing

Run syntax checks:

```bash
python -m compileall app.py src tests
```

Run unit tests:

```bash
pytest
```

## Deliverables

- Source code: this repository
- Architecture diagram: `docs/architecture.md`
- Technical report: `reports/technical_report.md`
- Demo video guide: `docs/demo_script.md`
