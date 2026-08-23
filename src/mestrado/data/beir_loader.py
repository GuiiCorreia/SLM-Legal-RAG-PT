"""
BEIRLoader — Generic loader for JUÁ HuggingFace legal datasets (Pereira et al. 2026).

Supports:
  juris-tcu  : ufca-llms/juris-tcu   — 16k TCU docs,  150 queries, 0-3, EXCERTO only
  normas-tcu : ufca-llms/normas-tcu  — 14k TCU norms,  46 queries, 0-2, ASSUNTO+TEXTONORMA
  jua        : ufca-llms/jua         — 17k multi-src, 1714 queries, binary, ENUNCIADO↔EXCERTO
  br-taxqa   : ufca-llms/BR-TaxQA   — 715 tax FAQ docs, 715 queries, 1-2 scale

All datasets are converted to Bill/Query schema for compatibility with BM25Retriever,
DenseRetriever, HybridRetriever, and the RAG generation pipeline.

Field schemas discovered from HuggingFace dataset pages (Pereira et al. 2026):
  juris-tcu corpus : corpus-id (ID), EXCERTO (text). ENUNCIADO excluded in JUÁ version.
  normas-tcu corpus: corpus-id (ID), ASSUNTO (title/summary), TEXTONORMA (full text).
  jua corpus       : corpus-id (ID), _excerto_ (text). Queries: query-id, _enunciado_.
  BR-TaxQA         : corpus_id / query_id (underscore variant), standard text fields.

Usage:
    loader = BEIRLoader("juris-tcu")
    bills, queries = loader.load()

    loader = BEIRLoader("normas-tcu", max_doc_chars=30_000)
    bills, queries = loader.load()
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from mestrado.data.schema import Bill, Query, FeedbackItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset field mapping — discovered from HuggingFace dataset pages
# ---------------------------------------------------------------------------

_FIELD_MAP: dict[str, dict] = {
    "ufca-llms/juris-tcu": {
        "corpus_id":    "corpus-id",
        "corpus_text":  "EXCERTO",
        "corpus_title": None,           # ENUNCIADO intentionally excluded in JUÁ version
        "query_id":     "query-id",
        "query_text":   "text",
        "qrel_qid":     "query-id",
        "qrel_did":     "corpus-id",
        "qrel_score":   "score",
        "qrels_config": "qrels",
        "qrels_split":  "test",
    },
    "ufca-llms/normas-tcu": {
        "corpus_id":    "corpus-id",
        "corpus_text":  "TEXTONORMA",
        "corpus_title": "ASSUNTO",      # missing in ~7k docs; use empty string if absent
        "query_id":     "query-id",
        "query_text":   "text",
        "qrel_qid":     "query-id",
        "qrel_did":     "corpus-id",
        "qrel_score":   "score",
        "qrels_config": "qrels",
        "qrels_split":  "test",
    },
    "ufca-llms/jua": {
        "corpus_id":    "corpus-id",
        "corpus_text":  "_excerto_",
        "corpus_title": None,
        "query_id":     "query-id",
        "query_text":   "_enunciado_",
        "qrel_qid":     "query-id",
        "qrel_did":     "corpus-id",
        "qrel_score":   "score",
        "qrels_config": "default",      # qrels live in "default" config, not "qrels"
        "qrels_split":  "train",
    },
    "ufca-llms/BR-TaxQA": {
        "corpus_id":    "corpus_id",    # underscore variant
        "corpus_text":  "text",
        "corpus_title": "title",
        "query_id":     "query_id",     # underscore variant
        "query_text":   "text",
        "qrel_qid":     "query_id",
        "qrel_did":     "corpus_id",
        "qrel_score":   "score",
        "qrels_config": "default",
        "qrels_split":  "test",
    },
}

# Ordered fallback sequences when the preferred field is absent
_CORPUS_ID_FALLBACKS  = ["corpus-id", "corpus_id", "_id", "id"]
_CORPUS_TEXT_FALLBACKS = ["EXCERTO", "_excerto_", "TEXTONORMA", "text", "TEXT", "content"]
_CORPUS_TITLE_FALLBACKS = ["ASSUNTO", "_enunciado_", "title", "TITLE"]
_QUERY_ID_FALLBACKS   = ["query-id", "query_id", "_id", "id"]
_QUERY_TEXT_FALLBACKS = ["_enunciado_", "text", "TEXT", "query", "QUERY"]
_QREL_QID_FALLBACKS   = ["query-id", "query_id"]
_QREL_DID_FALLBACKS   = ["corpus-id", "corpus_id", "doc_id"]


def _first_present(row: dict, candidates: list[str], default: str = "") -> str:
    for key in candidates:
        val = row.get(key)
        if val is not None:
            return str(val)
    return default


class BEIRLoader:
    """
    Generic loader for JUÁ-benchmark HuggingFace datasets (Pereira et al. 2026).

    Converts non-standard BEIR-like datasets to Bill/Query objects compatible with
    all existing retrievers and the generation pipeline.
    """

    DATASET_IDS: dict[str, str] = {
        "juris-tcu":  "ufca-llms/juris-tcu",
        "normas-tcu": "ufca-llms/normas-tcu",
        "jua":        "ufca-llms/jua",
        "br-taxqa":   "ufca-llms/BR-TaxQA",
    }

    RELEVANCE_MAX: dict[str, float] = {
        "ufca-llms/juris-tcu":  3.0,
        "ufca-llms/normas-tcu": 2.0,
        "ufca-llms/jua":        1.0,
        "ufca-llms/BR-TaxQA":   2.0,
    }

    _METADATA: dict[str, dict] = {
        "ufca-llms/juris-tcu": {
            "sig_tipo":    "TCU-JUA",
            "description": "JurisTCU via JUÁ (EXCERTO-only, 0-3 scale)",
        },
        "ufca-llms/normas-tcu": {
            "sig_tipo":    "NORMA-TCU",
            "description": "NormasTCU — TCU regulatory norms (0-2 scale)",
        },
        "ufca-llms/jua": {
            "sig_tipo":    "JUA",
            "description": "JUÁ aggregate benchmark — ENUNCIADO↔EXCERTO matching (binary)",
        },
        "ufca-llms/BR-TaxQA": {
            "sig_tipo":    "TAX",
            "description": "BR-TaxQA — Brazilian tax law FAQ (1-2 scale)",
        },
    }

    def __init__(
        self,
        dataset_name: str,
        max_doc_chars: int = 50_000,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            dataset_name: Short key ('juris-tcu', 'normas-tcu', 'jua', 'br-taxqa')
                         or full HF ID ('ufca-llms/juris-tcu').
            max_doc_chars: Truncate documents at this character count.
                          normas-tcu has docs up to 3.2 MB — truncation is essential.
            hf_token: HuggingFace token (falls back to HF_TOKEN env var).
            cache_dir: Local cache directory for HuggingFace downloads.
        """
        if dataset_name in self.DATASET_IDS:
            self.hf_id = self.DATASET_IDS[dataset_name]
        elif dataset_name in self.RELEVANCE_MAX:
            self.hf_id = dataset_name
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name!r}. "
                f"Valid short names: {sorted(self.DATASET_IDS)}"
            )

        self.max_doc_chars = max_doc_chars
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir
        self._rel_max  = self.RELEVANCE_MAX[self.hf_id]
        self._sig_tipo = self._METADATA.get(self.hf_id, {}).get("sig_tipo", "BEIR")
        self._fmap     = _FIELD_MAP.get(self.hf_id, {})

        logger.info(
            f"BEIRLoader: {self.hf_id}  rel_max={self._rel_max}  "
            f"max_doc={max_doc_chars}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> tuple[dict[str, Bill], list[Query]]:
        """
        Download and convert dataset to (bills_by_id, queries).

        Returns:
            bills_by_id : dict mapping doc_id → Bill
            queries     : list of Query with graded relevance judgments
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Package 'datasets' not installed. Run: uv add datasets"
            ) from exc

        print(f"\n  📥 Loading {self.hf_id} from HuggingFace...")

        kwargs: dict = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        corpus_ds, queries_ds, qrels_ds = self._load_subsets(load_dataset, kwargs)

        bills    = self._build_bills(corpus_ds)
        qrel_map = self._build_qrel_map(qrels_ds)
        queries  = self._build_queries(queries_ds, qrel_map)

        n_relevant = sum(1 for q in queries if q.has_relevant)
        print(
            f"  ✅ {self.hf_id}: {len(bills):,} docs | {len(queries)} queries "
            f"({n_relevant} with relevant judgments)"
        )
        return bills, queries

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_subsets(self, load_dataset, kwargs: dict):
        """
        Load corpus, queries, and qrels from HuggingFace.

        Each JUÁ dataset has a slightly different split/config structure,
        so we try multiple combinations systematically.
        """
        fmap = self._fmap
        qrels_config = fmap.get("qrels_config", "qrels")
        qrels_split  = fmap.get("qrels_split", "test")

        # Corpus: always in "corpus" config; split may be "train" or "corpus"
        corpus_ds = None
        for cs in ("train", "corpus"):
            try:
                corpus_ds = load_dataset(self.hf_id, "corpus", split=cs, **kwargs)
                logger.debug(f"Corpus loaded: config=corpus split={cs}")
                break
            except Exception:
                continue
        if corpus_ds is None:
            raise RuntimeError(f"Cannot load corpus for {self.hf_id}")

        # Queries: always in "queries" config
        queries_ds = None
        for qs in ("queries", "train"):
            try:
                queries_ds = load_dataset(self.hf_id, "queries", split=qs, **kwargs)
                logger.debug(f"Queries loaded: config=queries split={qs}")
                break
            except Exception:
                continue
        if queries_ds is None:
            raise RuntimeError(f"Cannot load queries for {self.hf_id}")

        # Qrels: config varies per dataset (qrels_config from _FIELD_MAP)
        qrels_ds = None
        for qc, qs in [
            (qrels_config, qrels_split),
            ("qrels", "test"),
            ("qrels", "train"),
            ("default", "train"),
            ("default", "test"),
        ]:
            try:
                qrels_ds = load_dataset(self.hf_id, qc, split=qs, **kwargs)
                logger.debug(f"Qrels loaded: config={qc} split={qs}")
                break
            except Exception:
                continue
        if qrels_ds is None:
            raise RuntimeError(f"Cannot load qrels for {self.hf_id}")

        return corpus_ds, queries_ds, qrels_ds

    # ------------------------------------------------------------------
    # Schema conversion
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        return text[: self.max_doc_chars] if text else ""

    def _build_bills(self, corpus_ds) -> dict[str, Bill]:
        """Convert corpus rows to {doc_id: Bill}."""
        fmap  = self._fmap
        id_key    = fmap.get("corpus_id")
        text_key  = fmap.get("corpus_text")
        title_key = fmap.get("corpus_title")

        bills: dict[str, Bill] = {}
        for row in corpus_ds:
            # Document ID — use mapped field or fallback sequence
            doc_id = (
                str(row[id_key]) if id_key and id_key in row
                else _first_present(row, _CORPUS_ID_FALLBACKS)
            )
            if not doc_id:
                continue

            # Document text
            text = (
                str(row[text_key] or "") if text_key and text_key in row
                else _first_present(row, _CORPUS_TEXT_FALLBACKS)
            )

            # Document title/summary (optional)
            title = ""
            if title_key and title_key in row and row[title_key]:
                title = str(row[title_key])
            elif title_key is None:
                # No title for this dataset (e.g. juris-tcu, jua corpus)
                pass
            else:
                title = _first_present(row, _CORPUS_TITLE_FALLBACKS)

            full_text = self._truncate(
                (title + " " + text).strip() if title else text
            )

            bills[doc_id] = Bill(
                code=doc_id,
                sig_tipo=self._sig_tipo,
                name=doc_id,                            # JOIN KEY — matches qrel corpus-id
                txt_ementa=title[:2000] if title else text[:500],
                em_tramitacao="N/A",
                situacao="N/A",
                text=full_text,
                preprocessed_tokens=[],                 # BM25 tokenizes .text at index time
            )

        logger.info(f"Built {len(bills):,} Bill objects")
        return bills

    def _build_qrel_map(self, qrels_ds) -> dict[str, list[tuple[str, int]]]:
        """Build {query_id: [(doc_id, score)]} from qrels rows."""
        fmap    = self._fmap
        qid_key = fmap.get("qrel_qid")
        did_key = fmap.get("qrel_did")
        sc_key  = fmap.get("qrel_score", "score")

        qrel_map: dict[str, list[tuple[str, int]]] = {}
        for row in qrels_ds:
            qid = (
                str(row[qid_key]) if qid_key and qid_key in row
                else _first_present(row, _QREL_QID_FALLBACKS)
            )
            did = (
                str(row[did_key]) if did_key and did_key in row
                else _first_present(row, _QREL_DID_FALLBACKS)
            )
            score = int(float(row.get(sc_key, 0) or 0))
            if qid and did:
                qrel_map.setdefault(qid, []).append((did, score))

        return qrel_map

    def _score_to_label(self, score: int) -> str:
        """Map numeric relevance score → FeedbackItem label (r / pr / i)."""
        if self._rel_max >= 3.0:
            # 0-3 scale (juris-tcu)
            if score >= 2:  return "r"
            if score == 1:  return "pr"
            return "i"
        elif self._rel_max == 2.0:
            # 0-2 or 1-2 scale (normas-tcu, BR-TaxQA)
            if score >= 2:  return "r"
            if score == 1:  return "pr"
            return "i"
        else:
            # Binary 0/1 (jua)
            return "r" if score >= 1 else "i"

    def _build_queries(
        self,
        queries_ds,
        qrel_map: dict[str, list[tuple[str, int]]],
    ) -> list[Query]:
        """Convert query rows + qrel_map to list[Query] (filtered to has_relevant)."""
        fmap    = self._fmap
        qid_key = fmap.get("query_id")
        txt_key = fmap.get("query_text")

        queries: list[Query] = []
        for row in queries_ds:
            qid = (
                str(row[qid_key]) if qid_key and qid_key in row
                else _first_present(row, _QUERY_ID_FALLBACKS)
            )
            text = (
                str(row[txt_key] or "") if txt_key and txt_key in row
                else _first_present(row, _QUERY_TEXT_FALLBACKS)
            )
            if not qid or not text:
                continue

            feedback_items = [
                FeedbackItem(
                    bill_id=did,
                    label=self._score_to_label(score),
                    score=float(score),
                    score_normalized=float(score) / self._rel_max,
                )
                for did, score in qrel_map.get(qid, [])
            ]

            queries.append(Query(
                query_id=qid,
                text=text,
                feedback=feedback_items,
                extra_results=[],
                date_created="",
                num_doc_feedback=len(feedback_items),
            ))

        return [q for q in queries if q.has_relevant]
