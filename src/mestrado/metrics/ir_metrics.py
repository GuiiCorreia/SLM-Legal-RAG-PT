"""
Information Retrieval evaluation metrics for BillsRAG-BR benchmark.

Implements: nDCG@K, MRR, P@K, Recall@K, MAP, Average Precision.
All metrics support graded relevance: r=1.0, pr=0.5, i=0.0.

Statistical testing: Wilcoxon signed-rank test with Bonferroni correction
for pairwise system comparisons (H1–H5 hypothesis testing).

References:
- Manning et al. (2008) — Introduction to Information Retrieval
- Järvelin & Kekäläinen (2002) — Cumulated gain-based evaluation
- Robertson (2010) — On the history of evaluation in IR
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

from mestrado.data.schema import Query
from mestrado.retrieval.bm25 import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.
    DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)
    """
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(
    retrieved: list[str],
    graded_relevance: dict[str, float],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    nDCG@k = DCG@k / IDCG@k

    Args:
        retrieved: Ordered list of bill names from the retrieval system
        graded_relevance: {bill_name: relevance_score} ground truth
        k: cutoff

    Returns:
        float in [0, 1]; 1.0 = perfect ranking
    """
    retrieved_rels = [graded_relevance.get(name, 0.0) for name in retrieved[:k]]
    ideal_rels = sorted(graded_relevance.values(), reverse=True)

    dcg = dcg_at_k(retrieved_rels, k)
    idcg = dcg_at_k(ideal_rels, k)

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: list[str], graded_relevance: dict[str, float]) -> float:
    """
    Mean Reciprocal Rank (single query).
    MRR = 1 / rank_of_first_relevant
    Counts both r and pr as relevant (weight > 0).
    """
    for rank, name in enumerate(retrieved, start=1):
        if graded_relevance.get(name, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved: list[str],
    graded_relevance: dict[str, float],
    k: int,
    threshold: float = 0.0,
) -> float:
    """
    Precision@k — fraction of top-k results that are relevant.
    threshold: minimum relevance score to count as relevant (default >0 = r or pr).
    """
    hits = sum(1 for name in retrieved[:k] if graded_relevance.get(name, 0.0) > threshold)
    return hits / k


def recall_at_k(
    retrieved: list[str],
    graded_relevance: dict[str, float],
    k: int,
    threshold: float = 0.0,
) -> float:
    """
    Recall@k — fraction of relevant documents found in top-k.
    """
    relevant_total = sum(1 for v in graded_relevance.values() if v > threshold)
    if relevant_total == 0:
        return 0.0
    hits = sum(1 for name in retrieved[:k] if graded_relevance.get(name, 0.0) > threshold)
    return hits / relevant_total


def average_precision(
    retrieved: list[str],
    graded_relevance: dict[str, float],
    threshold: float = 0.0,
) -> float:
    """
    Average Precision (for MAP computation).
    AP = (1/R) * Σ P@k * rel(k)
    """
    relevant_total = sum(1 for v in graded_relevance.values() if v > threshold)
    if relevant_total == 0:
        return 0.0

    hits = 0
    ap = 0.0
    for rank, name in enumerate(retrieved, start=1):
        if graded_relevance.get(name, 0.0) > threshold:
            hits += 1
            ap += hits / rank

    return ap / relevant_total


# ---------------------------------------------------------------------------
# Per-query result container
# ---------------------------------------------------------------------------

@dataclass
class QueryMetrics:
    query_id: str
    query_text: str
    ndcg_5: float
    ndcg_10: float
    mrr: float
    p5: float
    recall_12: float
    ap: float
    num_retrieved: int
    num_relevant_in_corpus: int

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text[:80],
            "ndcg@5": round(self.ndcg_5, 4),
            "ndcg@10": round(self.ndcg_10, 4),
            "mrr": round(self.mrr, 4),
            "p@5": round(self.p5, 4),
            "recall@12": round(self.recall_12, 4),
            "ap": round(self.ap, 4),
            "num_retrieved": self.num_retrieved,
            "num_relevant_in_corpus": self.num_relevant_in_corpus,
        }


# ---------------------------------------------------------------------------
# Evaluation Suite — runs metrics over all queries
# ---------------------------------------------------------------------------

@dataclass
class SystemResult:
    """Aggregate results for one system/experiment."""
    system_name: str
    mean_ndcg_5: float
    mean_ndcg_10: float
    mean_mrr: float
    mean_p5: float
    mean_recall_12: float
    map_score: float
    num_queries: int
    per_query: list[QueryMetrics] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # cost, latency, etc.

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "nDCG@5": round(self.mean_ndcg_5, 4),
            "nDCG@10": round(self.mean_ndcg_10, 4),
            "MRR": round(self.mean_mrr, 4),
            "P@5": round(self.mean_p5, 4),
            "Recall@12": round(self.mean_recall_12, 4),
            "MAP": round(self.map_score, 4),
            "n_queries": self.num_queries,
            **self.extra,
        }

    def ndcg10_array(self) -> list[float]:
        return [q.ndcg_10 for q in self.per_query]

    def mrr_array(self) -> list[float]:
        return [q.mrr for q in self.per_query]


class EvaluationSuite:
    """
    Evaluates a retrieval system over all queries in the benchmark.

    Usage:
        suite = EvaluationSuite(queries)
        results = suite.evaluate(
            system_name="BM25",
            retrieve_fn=lambda q: bm25.retrieve(q.text, top_k=12)
        )
        suite.print_results([bm25_results, dense_results])
        suite.significance_test(bm25_results, dense_results)
    """

    K_VALUES = [5, 10, 12]

    def __init__(self, queries: list[Query]) -> None:
        self.queries = queries

    def evaluate(
        self,
        system_name: str,
        retrieve_fn: Callable[[Query], list[RetrievalResult]],
        extra: dict | None = None,
    ) -> SystemResult:
        """
        Evaluate a retrieval function over all queries.

        retrieve_fn: takes a Query, returns list[RetrievalResult] ordered by rank.
        extra: dict with cost/latency info to attach to results.
        """
        per_query: list[QueryMetrics] = []

        for query in self.queries:
            results = retrieve_fn(query)
            retrieved_names = [r.bill.name for r in results]
            grades = query.graded_relevance()

            qm = QueryMetrics(
                query_id=query.query_id,
                query_text=query.text,
                ndcg_5=ndcg_at_k(retrieved_names, grades, k=5),
                ndcg_10=ndcg_at_k(retrieved_names, grades, k=10),
                mrr=mrr(retrieved_names, grades),
                p5=precision_at_k(retrieved_names, grades, k=5),
                recall_12=recall_at_k(retrieved_names, grades, k=12),
                ap=average_precision(retrieved_names, grades),
                num_retrieved=len(results),
                num_relevant_in_corpus=sum(1 for v in grades.values() if v > 0),
            )
            per_query.append(qm)

        return SystemResult(
            system_name=system_name,
            mean_ndcg_5=float(np.mean([q.ndcg_5 for q in per_query])),
            mean_ndcg_10=float(np.mean([q.ndcg_10 for q in per_query])),
            mean_mrr=float(np.mean([q.mrr for q in per_query])),
            mean_p5=float(np.mean([q.p5 for q in per_query])),
            mean_recall_12=float(np.mean([q.recall_12 for q in per_query])),
            map_score=float(np.mean([q.ap for q in per_query])),
            num_queries=len(per_query),
            per_query=per_query,
            extra=extra or {},
        )

    def significance_test(
        self,
        system_a: SystemResult,
        system_b: SystemResult,
        metric: str = "ndcg10",
        alpha: float = 0.05,
    ) -> dict:
        """
        Wilcoxon signed-rank test for paired comparison.
        H0: system_a and system_b have same nDCG@10 distribution.
        Bonferroni correction: alpha /= number_of_comparisons (apply externally).

        Returns dict with test statistic, p-value, and significance verdict.
        """
        if metric == "ndcg10":
            a_scores = system_a.ndcg10_array()
            b_scores = system_b.ndcg10_array()
        elif metric == "mrr":
            a_scores = system_a.mrr_array()
            b_scores = system_b.mrr_array()
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'ndcg10' or 'mrr'.")

        if len(a_scores) != len(b_scores):
            raise ValueError("Systems must be evaluated on the same queries for paired test.")

        stat, p_value = stats.wilcoxon(a_scores, b_scores, alternative="two-sided")
        delta = float(np.mean(b_scores)) - float(np.mean(a_scores))

        return {
            "system_a": system_a.system_name,
            "system_b": system_b.system_name,
            "metric": metric,
            "mean_a": round(float(np.mean(a_scores)), 4),
            "mean_b": round(float(np.mean(b_scores)), 4),
            "delta": round(delta, 4),
            "delta_pct": round(100 * delta / max(float(np.mean(a_scores)), 1e-9), 2),
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < alpha),
            "annotation": "†" if p_value < 0.05 else ("‡" if p_value < 0.01 else ""),
        }

    def print_results(self, systems: list[SystemResult]) -> None:
        """Pretty-print comparison table (as in Table 7.2 of survey structure)."""
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()
            table = Table(title="BillsRAG-BR Evaluation Results")
            table.add_column("System", style="bold")
            table.add_column("nDCG@5", justify="right")
            table.add_column("nDCG@10", justify="right")
            table.add_column("MRR", justify="right")
            table.add_column("P@5", justify="right")
            table.add_column("Recall@12", justify="right")
            table.add_column("MAP", justify="right")
            table.add_column("N", justify="right")

            for s in systems:
                table.add_row(
                    s.system_name,
                    f"{s.mean_ndcg_5:.4f}",
                    f"{s.mean_ndcg_10:.4f}",
                    f"{s.mean_mrr:.4f}",
                    f"{s.mean_p5:.4f}",
                    f"{s.mean_recall_12:.4f}",
                    f"{s.map_score:.4f}",
                    str(s.num_queries),
                )
            console.print(table)
        except ImportError:
            # Fallback: plain text
            header = f"{'System':<25} {'nDCG@5':>8} {'nDCG@10':>8} {'MRR':>8} {'P@5':>8} {'Recall@12':>10} {'MAP':>8}"
            print(header)
            print("-" * len(header))
            for s in systems:
                print(
                    f"{s.system_name:<25} {s.mean_ndcg_5:>8.4f} {s.mean_ndcg_10:>8.4f} "
                    f"{s.mean_mrr:>8.4f} {s.mean_p5:>8.4f} {s.mean_recall_12:>10.4f} {s.map_score:>8.4f}"
                )

    def save_results(self, systems: list[SystemResult], output_path: Path) -> None:
        """Save results to JSON for further analysis."""
        data = {
            "systems": [s.to_dict() for s in systems],
            "per_query": {
                s.system_name: [q.to_dict() for q in s.per_query]
                for s in systems
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
