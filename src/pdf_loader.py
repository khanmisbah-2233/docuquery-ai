"""PDF validation and text extraction."""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
from src.exceptions import EmptyDocumentError, InvalidPDFError
from src.models import PDFPage, UploadedPDF
from src.text_processor import clean_text


class PDFLoader:
    """Load PDF bytes into page-level text records."""

    def load(self, uploaded_pdf: UploadedPDF) -> List[PDFPage]:
        self._validate(uploaded_pdf)
        document_id = hashlib.sha256(uploaded_pdf.content).hexdigest()[:16]

        try:
            reader = PdfReader(BytesIO(uploaded_pdf.content))
            if reader.is_encrypted:
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise InvalidPDFError(
                        f"{uploaded_pdf.name} is encrypted and cannot be opened."
                    )
        except PdfReadError as exc:
            raise InvalidPDFError(f"{uploaded_pdf.name} is not a readable PDF.") from exc
        except Exception as exc:
            raise InvalidPDFError(f"{uploaded_pdf.name} could not be processed.") from exc

        pages: List[PDFPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception:
                text = ""
            if text:
                pages.append(
                    PDFPage(
                        document_id=document_id,
                        document_name=uploaded_pdf.name,
                        page_number=index,
                        text=text,
                    )
                )

        if not pages:
            raise EmptyDocumentError(
                f"{uploaded_pdf.name} has no extractable text. Scanned PDFs require OCR first."
            )

        return pages

    @staticmethod
    def _validate(uploaded_pdf: UploadedPDF) -> None:
        if not uploaded_pdf.name.lower().endswith(".pdf"):
            raise InvalidPDFError(f"{uploaded_pdf.name} must be a PDF file.")
        if not uploaded_pdf.content:
            raise InvalidPDFError(f"{uploaded_pdf.name} is empty.")
        if len(uploaded_pdf.content) > MAX_UPLOAD_SIZE_BYTES:
            raise InvalidPDFError(
                f"{uploaded_pdf.name} exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit."
            )

