"""
BM25 Retriever — baseline lexical retrieval.

Implements two variants:
- BM25Okapi  (Robertson & Zaragoza, 2009): standard, k1=1.5 b=0.75
- BM25L      (Lv & Zhai, 2011): robust to document-length variation —
             used in the Ulysses-RFCorpus reference paper (Vitório et al., 2024).

Both classes share the same interface so they are drop-in replaceable.

Tokenization design:
  Corpus and query both use _tokenize_text(): NFD unicode normalization,
  accent stripping, lowercase, punctuation removal, min-length 2.
  This symmetric approach avoids the corpus/query mismatch that would arise
  from indexing RSLP-stemmed preprocessed_tokens while tokenizing queries
  with a plain split. bill.text (raw full text) is used as the indexed field
  for both Ulysses and JurisTCU.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from rank_bm25 import BM25Okapi, BM25L

from mestrado.data.schema import Bill

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with its score."""
    bill: Bill
    score: float
    rank: int


class BM25Retriever:
    """
    BM25 retriever over the bills corpus.

    Indexes bill.text (raw full text) using symmetric NFD tokenization for
    both corpus and query — avoids the mismatch that arises from indexing
    RSLP-stemmed tokens while querying with plain split.
    Falls back to txt_ementa if text is empty.

    Example:
        retriever = BM25Retriever(bills_by_name)
        results = retriever.retrieve("cavalos de raça", top_k=50)
    """

    def __init__(
        self,
        bills_by_name: dict[str, Bill],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        from mestrado.config import experiment as cfg
        self.k1 = k1 or cfg.bm25_k1
        self.b = b or cfg.bm25_b
        self._bills: list[Bill] = list(bills_by_name.values())
        self._index: BM25Okapi | None = None
        logger.info(f"BM25Retriever initialized with {len(self._bills):,} bills.")

    def build_index(self) -> "BM25Retriever":
        """Build the BM25 index from the corpus. Call once before retrieve()."""
        logger.info("Building BM25 index ...")
        corpus_tokens = []
        fallback_count = 0
        for bill in self._bills:
            text = bill.text or bill.txt_ementa
            if not text:
                text = bill.txt_ementa
                fallback_count += 1
            corpus_tokens.append(self._tokenize_text(text))

        if fallback_count:
            logger.warning(
                f"{fallback_count} bills used ementa fallback (no text field)."
            )

        self._index = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
        logger.info("BM25 index built.")
        return self

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        """
        Retrieve top_k documents for a query.
        Query is tokenized identically to the corpus (_tokenize_text: NFD + accent
        strip + lowercase + punct removal). The remaining semantic mismatch
        (informal PT-BR queries vs. legislative corpus vocabulary) is the
        retrieval gap central to this thesis — but it is now a vocabulary/semantic
        gap, not an artificial tokenization gap.
        """
        if self._index is None:
            raise RuntimeError("Call build_index() before retrieve().")

        query_tokens = self._tokenize_text(query)
        scores = self._index.get_scores(query_tokens)

        # Pair bills with scores and sort descending
        bill_scores = sorted(
            zip(self._bills, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for rank, (bill, score) in enumerate(bill_scores[:top_k], start=1):
            results.append(RetrievalResult(bill=bill, score=float(score), rank=rank))

        return results

    def retrieve_by_ids(self, query: str, target_ids: set[str], top_k: int = 50) -> list[RetrievalResult]:
        """Retrieve and filter to only documents in target_ids (for evaluation)."""
        results = self.retrieve(query, top_k=top_k * 2)
        filtered = [r for r in results if r.bill.name in target_ids]
        for i, r in enumerate(filtered, start=1):
            r.rank = i
        return filtered[:top_k]

    @staticmethod
    def _tokenize_text(text: str) -> list[str]:
        """
        Symmetric tokenizer used for BOTH corpus documents and queries.

        Steps:
          1. NFD decompose → strip combining characters (removes accents:
             ç→c, ã→a, é→e, etc.) — critical for PT-BR lexical matching.
          2. Lowercase.
          3. Replace non-alphanumeric chars with space.
          4. Split and filter tokens shorter than 2 characters.

        Using the same function for corpus and query ensures tokens that
        appear in queries will match tokens in the index.
        """
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]


class BM25LRetriever(BM25Retriever):
    """
    BM25L retriever — Lv & Zhai (2011).

    BM25L avoids over-penalizing long documents by using a lower-bound on
    the normalized term frequency. This improves recall on legislative corpora
    where acórdãos and bills vary widely in length.

    Used as the primary retriever in Vitório et al. (2024) — the reference
    paper for the Ulysses-RFCorpus dataset.

    Reference: Lv, Y., & Zhai, C. (2011). Lower-Bounding Term Frequency
    Normalization. CIKM 2011.
    """

    def build_index(self) -> "BM25LRetriever":
        """Build the BM25L index from the corpus (same symmetric tokenization as BM25Retriever)."""
        logger.info("Building BM25L index ...")
        corpus_tokens = []
        fallback_count = 0
        for bill in self._bills:
            text = bill.text or bill.txt_ementa
            if not text:
                text = bill.txt_ementa
                fallback_count += 1
            corpus_tokens.append(self._tokenize_text(text))

        if fallback_count:
            logger.warning(f"{fallback_count} bills used ementa fallback.")

        self._index = BM25L(corpus_tokens, k1=self.k1, b=self.b)
        logger.info("BM25L index built.")
        return self
