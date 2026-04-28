# Technical Report: DocuQuery AI

## 1. Project Overview

DocuQuery AI is an intermediate Retrieval-Augmented Generation system designed for question answering over PDF documents. The application accepts PDF uploads, extracts and cleans text, chunks the content, generates dense vector embeddings, stores vectors in Pinecone, retrieves semantically relevant chunks for a user query, and generates an answer strictly from the retrieved context.

The primary goal is to reduce hallucination by grounding each answer in document text and by displaying traceable evidence, including page number, relevant excerpt, similarity score, document name, and chunk ID. If the retrieved context is insufficient, the system returns the required fallback response:

```text
The answer is not available in the provided document.
```

## 2. System Architecture

The system follows the required RAG pipeline:

```text
PDF Upload
  -> Text Extraction
  -> Text Chunking
  -> Embedding Generation
  -> Pinecone Vector Indexing
  -> Semantic Retrieval
  -> LLM Response Generation
  -> Answer with Source Reference
```

The codebase separates responsibilities into modules:

| Component | File | Purpose |
| --- | --- | --- |
| Interface | `app.py` | Streamlit upload, retrieval controls, answers, sources, history |
| Configuration | `src/config.py` | Environment variables, defaults, upload size |
| PDF loader | `src/pdf_loader.py` | PDF validation and page-level extraction |
| Text processor | `src/text_processor.py` | Cleanup and intelligent chunking |
| Embeddings | `src/embeddings.py` | Sentence Transformer embeddings |
| Vector store | `src/pinecone_store.py` | Pinecone index, namespace, upsert, query, metadata |
| Retrieval | `src/retriever.py` | Query embedding and threshold filtering |
| Generation | `src/generator.py` | Strict context-only LLM answer generation |
| Logging | `src/logger.py` | CSV query logs |
| Pipeline | `src/pipeline.py` | End-to-end orchestration |

This modular design improves maintainability and makes each stage testable independently.

## 3. Design Decisions

### Streamlit Interface

Streamlit was selected because it supports rapid development of a usable interface for upload, configuration, question answering, and evidence display. The UI includes controls for namespace, chunk size, chunk overlap, top-k retrieval, similarity threshold, LLM provider, model name, document filtering, and page filtering.

### Page-Level Traceability

PDFs are extracted page by page. Chunks retain the original page number so every answer can show source pages. This is essential for traceable answers and improves evaluation because users can verify the retrieved evidence directly.

### Chunking Strategy

The chunker is paragraph and sentence aware. It attempts to preserve semantic units before falling back to word windows for long passages. Chunk size and overlap are adjustable from the UI. The default chunk size is 500 words with 80 words of overlap, balancing context completeness with retrieval precision.

### Metadata Strategy

Each Pinecone vector stores:

- `document_id`
- `document_name`
- `page_number`
- `chunk_id`
- `chunk_index`
- `text`
- `word_count`

This metadata supports source attribution, document filtering, page filtering, and user-facing evidence.

## 4. Embedding Model

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is efficient for an intermediate RAG assignment because it generates compact dense embeddings with good semantic similarity performance and reasonable local resource requirements. The application reads the embedding dimension from the loaded model and creates the Pinecone index with the same dimension, preventing dimension mismatch errors.

The embedding service normalizes embeddings. Pinecone is configured with cosine similarity, making the vector scores suitable for semantic threshold filtering.

## 5. Pinecone Configuration

The default Pinecone settings are:

```text
PINECONE_INDEX_NAME=docuquery-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=course-assignment
```

The vector store layer demonstrates the required Pinecone operations:

- Index creation with `ServerlessSpec`
- Cosine metric configuration
- Namespace-based upsert and query
- Batched vector upserts
- Metadata management
- Metadata filters for document and page
- Index statistics display

If the index already exists, the application checks that its dimension and metric match the current embedding model. If they do not match, the user is asked to use a different index name. This prevents corrupting an existing index with incompatible vectors.

## 6. Retrieval and Confidence Scoring

When a user asks a question, the query is embedded using the same embedding model as the document chunks. Pinecone returns top-k matches from the selected namespace. Results below the selected similarity threshold are removed before answer generation.

The displayed confidence score is the highest similarity score among the chunks used for the answer. This is a practical confidence signal for an intermediate system. It does not claim mathematical certainty, but it helps users understand whether the answer is strongly or weakly grounded.

## 7. Hallucination Prevention

The system prevents hallucination through layered safeguards:

1. Retrieval filtering removes low-similarity chunks before generation.
2. If no chunks pass the threshold, the answer generator returns the required fallback immediately.
3. The LLM prompt explicitly states that the model must answer only from the provided context.
4. The prompt instructs the model to return the exact fallback message when the context is insufficient.
5. Retrieved excerpts and page references are displayed so users can audit the answer.

These safeguards do not guarantee perfect factuality in every possible case, but they significantly reduce unsupported answers and make the answer traceable.

## 8. Error Handling

The project handles the required failure cases:

- Invalid PDF file type
- Empty uploaded file
- PDF larger than 20 MB
- Encrypted or unreadable PDF
- PDF with no extractable text
- Empty user query
- Missing Pinecone API key
- Pinecone index creation or query failures
- Embedding model loading failures
- LLM API failures

Expected errors are surfaced as user-friendly Streamlit messages.

## 9. Intermediate Enhancements

The implementation includes these mandatory enhancement options:

| Enhancement | Status |
| --- | --- |
| Multi-document support | Implemented |
| Query history | Implemented |
| Adjustable chunk size from UI | Implemented |
| Adjustable top-k retrieval | Implemented |
| Metadata filtering by page | Implemented |
| Confidence scoring display | Implemented |
| Logging user queries | Implemented |

The application also includes adjustable chunk overlap and optional document-level filtering.

## 10. Challenges Faced

### Balancing Chunk Size and Retrieval Quality

Small chunks improve precision but may omit necessary context. Large chunks improve context completeness but can reduce retrieval accuracy. The solution exposes chunk size and overlap in the UI so users can tune retrieval behavior for different PDFs.

### Maintaining Source Traceability

Chunks must remain tied to page numbers even after text cleanup and splitting. The loader extracts page-level text first, then the chunker creates chunks within each page instead of mixing pages together.

### Avoiding Pinecone Dimension Mismatch

Pinecone indexes require a fixed vector dimension. If a user changes embedding models, the dimension can change. The vector store checks the existing index dimension before upserting and raises a clear error if the dimensions differ.

### Preventing Unsupported Answers

LLMs can answer from prior knowledge if not constrained. The generator uses a strict prompt, low temperature, retrieval thresholding, and the required fallback response to reduce hallucinations.

## 11. Performance Analysis

Performance depends on PDF size, chunk size, embedding model speed, and Pinecone network latency.

Expected behavior for typical course documents:

- Text extraction is usually fast for text-based PDFs.
- Embedding time increases with chunk count.
- Upsert time depends on Pinecone network latency and batch size.
- Query latency is generally low because only one query vector is searched.
- LLM generation latency depends on the selected provider and model.

The application uses batched Pinecone upserts with a default batch size of 100 vectors. This is more efficient than upserting each vector individually. Smaller chunk sizes produce more vectors and higher ingestion time, while larger chunks reduce vector count but may reduce retrieval precision.

## 12. Conclusion

DocuQuery AI satisfies the assignment requirements for an intermediate RAG system using Pinecone. It implements the full PDF-to-answer pipeline, demonstrates Pinecone index creation and namespace usage, stores page-level metadata, supports semantic retrieval with adjustable controls, generates context-grounded answers through an LLM API, and displays traceable source references.

The project is structured professionally for extension. Future improvements could add OCR for scanned PDFs, reranking, evaluation datasets, authentication, and deployment packaging.

