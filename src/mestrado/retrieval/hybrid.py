"""
Hybrid Retriever — Experiment 3.
Combines BM25 (sparse) + Dense (semantic) via Reciprocal Rank Fusion (RRF).

RRF formula (Cormack et al., 2009):
    RRF(d) = Σ 1 / (k + rank_i(d))   where k=60 (default, robust across datasets)

Captures complementarity:
- BM25 excels when query terms appear literally in the document
- Dense excels when semantic meaning matches despite different vocabulary
- RRF is robust to score scale differences between the two systems
"""
from __future__ import annotations

import logging
from collections import defaultdict

from mestrado.data.schema import Bill
from mestrado.retrieval.bm25 import BM25Retriever, RetrievalResult
from mestrado.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever via Reciprocal Rank Fusion.

    RRF requires no score calibration — just the rank positions.
    This is the recommended Stage 1 retriever for the LLM reranker.

    Example:
        hybrid = HybridRetriever(bm25, dense)
        results = hybrid.retrieve("cavalos de raça", top_k=50)
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        bill_key: str = "name",
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bill_key = bill_key  # "name" for Ulysses, "code" for JurisTCU

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        """
        Fuse BM25 and Dense rankings via weighted RRF.
        Fetches 2×top_k from each system for better fusion coverage.
        """
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int = 50) -> list[list[RetrievalResult]]:
        """
        Batch RRF fusion: encodes all dense queries at once, runs BM25 in parallel.
        Much faster than calling retrieve() per query.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        fetch_k = top_k * 2

        # Dense: all queries in one batch (amortizes model inference)
        all_dense = self.dense.retrieve_batch(queries, top_k=fetch_k)

        # BM25: parallel threads (no GPU bottleneck, pure CPU)
        all_bm25 = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(self.bm25.retrieve, q, fetch_k): i for i, q in enumerate(queries)}
            for fut in as_completed(futs):
                all_bm25[futs[fut]] = fut.result()

        output = []
        for bm25_results, dense_results in zip(all_bm25, all_dense):
            rrf_scores = self._reciprocal_rank_fusion(bm25_results, dense_results)
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            bill_map: dict[str, Bill] = {}
            for r in bm25_results:
                bill_map[getattr(r.bill, self.bill_key)] = r.bill
            for r in dense_results:
                bill_map[getattr(r.bill, self.bill_key)] = r.bill
            results = []
            for rank, (bill_name, score) in enumerate(sorted_results[:top_k], start=1):
                if bill_name in bill_map:
                    results.append(RetrievalResult(bill=bill_map[bill_name], score=score, rank=rank))
            output.append(results)
        return output

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
    ) -> dict[str, float]:
        """
        Core RRF computation.
        RRF(d) = w_bm25 * 1/(k + rank_bm25(d)) + w_dense * 1/(k + rank_dense(d))
        """
        scores: dict[str, float] = defaultdict(float)

        for result in bm25_results:
            scores[getattr(result.bill, self.bill_key)] += (
                self.bm25_weight / (self.rrf_k + result.rank)
            )

        for result in dense_results:
            scores[getattr(result.bill, self.bill_key)] += (
                self.dense_weight / (self.rrf_k + result.rank)
            )

        return dict(scores)

    @classmethod
    def build(
        cls,
        bills_by_name: dict[str, Bill],
        dense_cache_path=None,
        rrf_k: int = 60,
    ) -> "HybridRetriever":
        """Convenience factory: builds and indexes both retrievers."""
        from mestrado.config import experiment as cfg

        bm25 = BM25Retriever(bills_by_name, k1=cfg.bm25_k1, b=cfg.bm25_b)
        bm25.build_index()

        dense = DenseRetriever(bills_by_name)
        dense.build_index(cache_path=dense_cache_path)

        return cls(bm25=bm25, dense=dense, rrf_k=rrf_k)
