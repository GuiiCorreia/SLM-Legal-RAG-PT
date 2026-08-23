"""
Dataset loader for bills_dataset.csv and relevance_feedback_dataset.csv.

Key insight from dataset analysis:
- bills_dataset.csv: join key is 'name' column (e.g., "PL 3650/2021")
- relevance_feedback_dataset.csv: references bills by 'id' field inside user_feedback JSON
  (e.g., {'id': 'PL 3650/2021', 'class': 'r', ...})
- Both files use CSV with embedded newlines → need proper csv.reader quoting
"""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

from tqdm import tqdm

from .schema import Bill, Query

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates both datasets, reporting join integrity statistics.

    Usage:
        loader = DataLoader()
        bills, queries = loader.load_all()
        # or separately:
        bills = loader.load_bills()
        queries = loader.load_queries()
    """

    def __init__(
        self,
        bills_path: str | Path | None = None,
        feedback_path: str | Path | None = None,
    ) -> None:
        from mestrado.config import data as cfg
        self.bills_path = Path(bills_path or cfg.bills_path)
        self.feedback_path = Path(feedback_path or cfg.feedback_path)
        self._bills_by_name: dict[str, Bill] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(
        self,
        sample_size: int | None = None,
        seed: int = 42,
    ) -> tuple[dict[str, Bill], list[Query]]:
        """
        Load both datasets. Returns (bills_by_name, queries).
        bills_by_name: {bill.name -> Bill} for O(1) lookup during evaluation.
        """
        bills_by_name = self.load_bills()
        queries = self.load_queries(sample_size=sample_size, seed=seed)
        self._report_join_integrity(bills_by_name, queries)
        return bills_by_name, queries

    def load_bills(self) -> dict[str, Bill]:
        """
        Parse bills_dataset.csv → {name: Bill}.
        Handles embedded newlines via csv.reader with QUOTE_MINIMAL.
        """
        if self._bills_by_name is not None:
            return self._bills_by_name

        logger.info(f"Loading bills from {self.bills_path} ...")
        bills: dict[str, Bill] = {}

        # Increase CSV field size limit for large text fields (full legislative text)
        csv.field_size_limit(10000000)  # 10MB limit per field

        with open(self.bills_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading bills", unit=" bills"):
                bill = Bill.from_csv_row(row)
                if bill.name:
                    bills[bill.name] = bill

        logger.info(f"Loaded {len(bills):,} bills.")
        self._bills_by_name = bills
        return bills

    def load_queries(
        self,
        sample_size: int | None = None,
        seed: int = 42,
    ) -> list[Query]:
        """
        Parse relevance_feedback_dataset.csv → list[Query].
        Optionally sample a subset for SLM experiments (stratified by query length).
        """
        logger.info(f"Loading queries from {self.feedback_path} ...")
        queries: list[Query] = []

        with open(self.feedback_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = Query.from_csv_row(row)
                if q.text and q.has_relevant:
                    queries.append(q)

        logger.info(f"Loaded {len(queries)} queries with at least one relevant document.")

        if sample_size is not None and sample_size < len(queries):
            queries = self._stratified_sample(queries, sample_size, seed)
            logger.info(f"Sampled {len(queries)} queries (stratified, seed={seed}).")

        return queries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stratified_sample(
        self,
        queries: list[Query],
        n: int,
        seed: int,
    ) -> list[Query]:
        """
        Stratified sample by query length (short/medium/long) to maintain
        distribution representativeness — avoids pure random bias.
        Validates with KS-test in EvaluationSuite later.
        """
        rng = random.Random(seed)

        def bucket(q: Query) -> str:
            words = len(q.text.split())
            if words <= 8:
                return "short"
            elif words <= 18:
                return "medium"
            return "long"

        buckets: dict[str, list[Query]] = {"short": [], "medium": [], "long": []}
        for q in queries:
            buckets[bucket(q)].append(q)

        result: list[Query] = []
        per_bucket = n // 3
        remainder = n % 3

        for i, (_, bucket_queries) in enumerate(buckets.items()):
            k = per_bucket + (1 if i < remainder else 0)
            k = min(k, len(bucket_queries))
            result.extend(rng.sample(bucket_queries, k))

        rng.shuffle(result)
        return result[:n]

    def _report_join_integrity(
        self,
        bills_by_name: dict[str, Bill],
        queries: list[Query],
    ) -> None:
        """
        Checks what fraction of referenced bills exist in the corpus.
        This is limitation L-D2 in 20_limitacoes_ameacas_validade.md.
        """
        total_refs = 0
        missing_refs = 0
        missing_examples: list[str] = []

        for q in queries:
            for item in q.feedback:
                total_refs += 1
                if item.bill_id not in bills_by_name:
                    missing_refs += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(item.bill_id)
            for bill_id in q.extra_results:
                total_refs += 1
                if bill_id not in bills_by_name:
                    missing_refs += 1

        pct = 100 * missing_refs / total_refs if total_refs > 0 else 0
        logger.info(
            f"Join integrity: {total_refs} total references, "
            f"{missing_refs} missing ({pct:.1f}%). "
            f"Examples: {missing_examples[:3]}"
        )
        if pct > 10:
            logger.warning(
                f"High join miss rate ({pct:.1f}%) — see limitation L-D2. "
                "Metrics computed only on bills present in corpus."
            )

    def get_valid_bill_names(self) -> set[str]:
        """Anti-hallucination filter set — pattern from findRelevantMemories.ts validFilenames."""
        if self._bills_by_name is None:
            self.load_bills()
        return set(self._bills_by_name.keys())  # type: ignore[union-attr]

    def load_ulysses(
        self,
        sample_size: int | None = None,
        seed: int = 42,
    ) -> tuple[dict[str, Bill], list[Query]]:
        """Alias for load_all() — returns (bills_by_name, queries) for Ulysses-RFCorpus."""
        return self.load_all(sample_size=sample_size, seed=seed)

    def load_juristcu(self) -> tuple[dict[str, Bill], list[Query]]:
        """Load JurisTCU via ufca-llms/juris-tcu (JUÁ benchmark version, EXCERTO-only).

        Uses the same corpus construction as Pereira et al. (2026) — EXCERTO field only,
        intentionally excluding ENUNCIADO to ensure fair BM25/Dense comparison with JUÁ.
        """
        from mestrado.data.beir_loader import BEIRLoader
        return BEIRLoader("juris-tcu").load()


def load_ulysses(
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, Bill], list[Query]]:
    """Module-level convenience function — load Ulysses-RFCorpus."""
    return DataLoader().load_all(sample_size=sample_size, seed=seed)
