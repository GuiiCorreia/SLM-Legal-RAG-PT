"""
Data schemas for both datasets.
Using dataclasses for lightweight, typed representation without ORM overhead.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal


RelevanceLabel = Literal["r", "pr", "i"]

# Graded relevance weights for nDCG computation
# r=1.0, pr=0.5, i=0.0  — consistent with standard IR evaluation practice
RELEVANCE_WEIGHTS: dict[RelevanceLabel, float] = {"r": 1.0, "pr": 0.5, "i": 0.0}


@dataclass
class Bill:
    """
    One row from bills_dataset.csv.
    Join key: name (e.g., "PL 3650/2021") — matches FeedbackItem.bill_id
    """
    code: str              # Internal numeric ID (e.g., "2307087")
    sig_tipo: str          # Type: PL, PEC, PLP, PDC, INC
    name: str              # Human-readable ID (e.g., "PL 3650/2021") — JOIN KEY
    txt_ementa: str        # Official summary — used for dense retrieval
    em_tramitacao: str     # "Sim"/"Não" — active/archived
    situacao: str          # Current status
    text: str              # Full document text
    preprocessed_tokens: list[str] = field(default_factory=list)  # Parsed text_preprocessed

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "Bill":
        tokens: list[str] = []
        raw_pre = row.get("text_preprocessed", "")
        if raw_pre and raw_pre.startswith("["):
            try:
                tokens = ast.literal_eval(raw_pre)
            except (ValueError, SyntaxError):
                tokens = []

        return cls(
            code=row.get("code", ""),
            sig_tipo=row.get("sig_tipo", ""),
            name=row.get("name", ""),
            txt_ementa=row.get("txt_ementa", ""),
            em_tramitacao=row.get("em_tramitacao", ""),
            situacao=row.get("situacao", ""),
            text=row.get("text", ""),
            preprocessed_tokens=tokens,
        )

    def retrieval_text(self, field: str = "preprocessed") -> str | list[str]:
        """Return the appropriate text representation for retrieval."""
        if field == "preprocessed":
            return self.preprocessed_tokens
        elif field == "ementa":
            return self.txt_ementa
        elif field == "full":
            return self.txt_ementa + " " + self.text[:2000]
        return self.txt_ementa


@dataclass
class FeedbackItem:
    """One document judgment within a query's user_feedback list."""
    bill_id: str             # e.g., "PL 3650/2021" — matches Bill.name
    label: RelevanceLabel    # "r", "pr", or "i"
    score: float             # Original system score
    score_normalized: float  # 0–1 normalized score

    @property
    def relevance_weight(self) -> float:
        return RELEVANCE_WEIGHTS[self.label]

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackItem":
        return cls(
            bill_id=str(d.get("id", "")),
            label=d.get("class", "i"),
            score=float(d.get("score", 0.0)),
            score_normalized=float(d.get("score_normalized", 0.0)),
        )


@dataclass
class Query:
    """
    One row from relevance_feedback_dataset.csv.
    Contains the query text and all human relevance judgments.
    """
    query_id: str                        # Original CSV "id" field
    text: str                            # Query text in natural language
    feedback: list[FeedbackItem]         # Up to 12 judgments
    extra_results: list[str]             # Additional relevant bills (ground truth extra)
    date_created: str
    num_doc_feedback: int

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "Query":
        feedback_raw = row.get("user_feedback", "[]")
        feedback_items: list[FeedbackItem] = []
        if feedback_raw and feedback_raw.startswith("["):
            try:
                parsed = ast.literal_eval(feedback_raw)
                feedback_items = [FeedbackItem.from_dict(d) for d in parsed]
            except (ValueError, SyntaxError):
                pass

        extra_raw = row.get("extra_results", "[]")
        extra: list[str] = []
        if extra_raw and extra_raw.startswith("["):
            try:
                extra = ast.literal_eval(extra_raw)
            except (ValueError, SyntaxError):
                pass

        return cls(
            query_id=str(row.get("id", "")),
            text=row.get("query", ""),
            feedback=feedback_items,
            extra_results=extra,
            date_created=row.get("date_created", ""),
            num_doc_feedback=int(row.get("num_doc_feedback", 0)),
        )

    def graded_relevance(self) -> dict[str, float]:
        """
        Returns {bill_id: relevance_weight} for nDCG computation.
        Includes both primary judgments and extra_results (weight 1.0).
        """
        grades: dict[str, float] = {}
        for item in self.feedback:
            grades[item.bill_id] = item.relevance_weight
        for bill_id in self.extra_results:
            if bill_id not in grades:
                grades[bill_id] = 1.0  # extra_results are implicitly relevant
        return grades

    @property
    def has_relevant(self) -> bool:
        """True if query has at least one relevant document."""
        return any(f.label in ("r", "pr") for f in self.feedback) or bool(self.extra_results)
