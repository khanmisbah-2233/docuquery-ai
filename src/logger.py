"""CSV logging for user queries and retrieval outcomes."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import GeneratedAnswer


class QueryLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.fieldnames = [
            "timestamp_utc",
            "namespace",
            "query",
            "top_k",
            "similarity_threshold",
            "result_count",
            "best_score",
            "answer_supported",
            "document_pages",
        ]

    def log(
        self,
        namespace: str,
        query: str,
        top_k: int,
        similarity_threshold: float,
        answer: GeneratedAnswer,
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.log_path.exists()
        best_score: Optional[float] = None
        if answer.sources:
            best_score = max(source.score for source in answer.sources)

        pages = "; ".join(
            f"{source.document_name}:p{source.page_number}" for source in answer.sources
        )
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "namespace": namespace,
            "query": query.strip(),
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "result_count": len(answer.sources),
            "best_score": "" if best_score is None else round(best_score, 4),
            "answer_supported": answer.is_supported,
            "document_pages": pages,
        }

        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

