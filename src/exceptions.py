"""Project-specific exceptions with user-safe messages."""


class DocuQueryError(Exception):
    """Base exception for expected application errors."""


class InvalidPDFError(DocuQueryError):
    """Raised when an uploaded file cannot be read as a valid PDF."""


class EmptyDocumentError(DocuQueryError):
    """Raised when a PDF has no extractable text."""


class EmptyQueryError(DocuQueryError):
    """Raised when a user submits a blank query."""


class AmbiguousQueryError(DocuQueryError):
    """Raised when a query is too vague for reliable retrieval."""


class PineconeConfigurationError(DocuQueryError):
    """Raised when Pinecone credentials or settings are missing."""


class PineconeOperationError(DocuQueryError):
    """Raised when a Pinecone operation fails."""


class EmbeddingModelError(DocuQueryError):
    """Raised when embeddings cannot be generated."""


class LLMGenerationError(DocuQueryError):
    """Raised when answer generation fails."""
