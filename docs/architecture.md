# Architecture Diagram

```mermaid
flowchart TD
    A[PDF Upload<br/>Streamlit UI] --> B[PDF Validation<br/>20 MB, PDF type]
    B --> C[Text Extraction<br/>pypdf page text]
    C --> D[Text Cleaning<br/>formatting artifact removal]
    D --> E[Intelligent Chunking<br/>sentence and paragraph aware]
    E --> F[Embedding Generation<br/>Sentence Transformer]
    F --> G[Pinecone Index Creation<br/>serverless cosine index]
    G --> H[Pinecone Namespace Upsert<br/>vectors plus metadata]
    H --> I[(Pinecone Vector DB)]

    J[User Query] --> K[Query Validation]
    K --> L[Query Embedding]
    L --> M[Semantic Retrieval<br/>top-k plus threshold]
    M --> N[Metadata Filters<br/>document/page]
    N --> I
    I --> O[Retrieved Context<br/>page, excerpt, score]
    O --> P[Strict LLM Prompt<br/>context-only answer]
    P --> Q[Answer or Required Fallback]
    Q --> R[Source Attribution<br/>page, excerpt, score]
    R --> S[Query History and CSV Log]
```

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI, user controls, PDF upload, answer display |
| `src/config.py` | Environment variables and runtime settings |
| `src/pdf_loader.py` | PDF validation and page-level text extraction |
| `src/text_processor.py` | Text cleanup and chunk generation |
| `src/embeddings.py` | Sentence Transformer model loading and embedding generation |
| `src/pinecone_store.py` | Pinecone index creation, namespace upsert, query, metadata filtering |
| `src/retriever.py` | Query embedding, semantic retrieval, threshold filtering |
| `src/generator.py` | Strict context-only LLM answer generation |
| `src/logger.py` | User query logging |
| `src/pipeline.py` | End-to-end orchestration |

## Data Flow

1. The user uploads one or more PDF files through the Streamlit interface.
2. Each PDF is validated for file type, non-empty content, and size up to 20 MB.
3. Text is extracted page by page with `pypdf`.
4. Extracted text is cleaned to remove common PDF line break and spacing artifacts.
5. Text is chunked with page-level traceability.
6. Chunks are embedded with a Sentence Transformer model.
7. The application creates a Pinecone index when necessary with cosine similarity.
8. Chunk vectors are upserted to the selected namespace with metadata.
9. User queries are embedded and searched in Pinecone with top-k retrieval.
10. Optional document and page filters are applied through Pinecone metadata filters.
11. Retrieved chunks below the selected similarity threshold are removed.
12. The LLM receives only the retrieved context and returns a grounded answer or the fallback message.
13. The UI displays the answer and traceable source references.

