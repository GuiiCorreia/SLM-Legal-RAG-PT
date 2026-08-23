"""
Dense Retriever — Experiment 2.
Sentence-transformers embeddings + FAISS IndexFlatIP for semantic search.

Uses txt_ementa for indexing (shorter, cleaner than full text — avoids noise).
Multilingual model covers PT-BR legislative vocabulary.

GPU usage (RTX 3050 4GB):
  - Automatically uses CUDA if torch was installed with CUDA support.
  - fp16 precision halves VRAM usage (~1.5 GB total for BGE-M3 vs ~3.8 GB fp32).
  - batch_size=32 for corpus encoding (safe for 4 GB), 64 for query batches.
  - Install CUDA torch: uv pip install torch --index-url https://download.pytorch.org/whl/cu121
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mestrado.data.schema import Bill
from mestrado.retrieval.bm25 import RetrievalResult

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Dense retrieval unavailable.")

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


class DenseRetriever:
    """
    Dense retriever using sentence-transformers + FAISS.

    Encodes bill ementas into dense vectors and performs cosine similarity
    search via FAISS IndexFlatIP (inner product on normalized vectors = cosine).

    GPU notes:
      - Uses CUDA automatically when available (install torch with CUDA support).
      - fp16=True halves VRAM: BGE-M3 goes from ~3.8 GB to ~1.5 GB on GPU.
      - batch_size=32 is safe for 4 GB VRAM with BGE-M3 fp16.

    Example:
        retriever = DenseRetriever(bills_by_name)
        retriever.build_index()
        results = retriever.retrieve("cavalos de raça", top_k=50)
    """

    # Conservative batch sizes: corpus encoding uses more VRAM than query encoding
    _BATCH_SIZE_CORPUS = 32   # safe for 4 GB VRAM with BGE-M3 fp16
    _BATCH_SIZE_QUERIES = 64  # queries are shorter, can use larger batch

    def __init__(
        self,
        bills_by_name: dict[str, Bill],
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from mestrado.config import data as cfg
        if not FAISS_AVAILABLE:
            raise ImportError("Install faiss-cpu: uv add faiss-cpu")

        self.model_name = model_name or cfg.embedding_model
        # batch_size arg kept for backwards compatibility; internal defaults are used otherwise
        self._batch_size_override = batch_size
        self._bills: list[Bill] = list(bills_by_name.values())
        self._model: SentenceTransformer | None = None
        self._index: "faiss.IndexFlatIP | None" = None
        logger.info(
            f"DenseRetriever initialized: {len(self._bills):,} bills, "
            f"model={self.model_name}, device={_DEVICE}"
        )

    def _load_model(self) -> SentenceTransformer:
        """Load model onto GPU (if available) with fp16 to save VRAM."""
        model = SentenceTransformer(self.model_name, device=_DEVICE)
        if _DEVICE == "cuda":
            try:
                model = model.half()  # fp16: halves VRAM, negligible accuracy loss
                logger.info(f"Model loaded on GPU ({_DEVICE}) in fp16.")
            except Exception:
                logger.warning("fp16 cast failed, staying in fp32.")
        else:
            logger.info("CUDA not available — running on CPU. "
                        "Install torch with CUDA: uv pip install torch "
                        "--index-url https://download.pytorch.org/whl/cu121")
        return model

    def build_index(self, cache_path: Path | None = None) -> "DenseRetriever":
        """
        Encode all bills and build FAISS index.
        Optionally saves/loads embeddings from cache to avoid re-encoding.
        """
        if cache_path and cache_path.exists():
            logger.info(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            embeddings = data["embeddings"]
            logger.info("Cache loaded.")
        else:
            logger.info(f"Encoding {len(self._bills):,} bills with {self.model_name} on {_DEVICE}...")
            self._model = self._load_model()
            texts = [self._bill_text(b) for b in self._bills]
            batch_size = self._batch_size_override or self._BATCH_SIZE_CORPUS
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # L2 normalize → cosine = inner product
                convert_to_numpy=True,
            )
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({"embeddings": embeddings}, f)
                logger.info(f"Embeddings cached to {cache_path}")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}.")
        return self

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        """Semantic search: encode query → FAISS search → ranked results."""
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int = 50) -> list[list[RetrievalResult]]:
        """Batch semantic search: encode all queries at once → FAISS batch search.

        Much faster than calling retrieve() one query at a time because
        sentence-transformers amortizes tokenization and GPU/CPU inference
        across the full batch.
        """
        if self._index is None:
            raise RuntimeError("Call build_index() before retrieve_batch().")
        if self._model is None:
            self._model = self._load_model()

        batch_size = self._batch_size_override or self._BATCH_SIZE_QUERIES
        q_embs = self._model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        all_scores, all_indices = self._index.search(q_embs, top_k)

        output = []
        for scores, indices in zip(all_scores, all_indices):
            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx < 0:
                    continue
                results.append(
                    RetrievalResult(bill=self._bills[idx], score=float(score), rank=rank)
                )
            output.append(results)
        return output

    @staticmethod
    def _bill_text(bill: Bill) -> str:
        """Text used for encoding — ementa is concise and representative."""
        return bill.txt_ementa or bill.name
