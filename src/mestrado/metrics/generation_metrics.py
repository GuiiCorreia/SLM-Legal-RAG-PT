"""
Generation quality metrics for RAG evaluation — ROUGE-1, ROUGE-2, ROUGE-L, BERTScore.

Used in SLM comparison experiments to measure quality of generated answers
relative to a reference (oracle LLM output for H5, or gold answers when available).

References:
  - Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
    ACL Workshop on Text Summarization Branches Out.
  - Zhang*, T., Kishore*, V., Wu*, F., Weinberger, K. Q., & Artzi, Y. (2020).
    BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
  - Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
    NeurIPS 2023. [LLM-as-judge methodology]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Aggregated generation metrics for a set of predictions."""
    rouge_1_scores: list[float] = field(default_factory=list)
    rouge_2_scores: list[float] = field(default_factory=list)
    rouge_l_scores: list[float] = field(default_factory=list)
    bertscore_f1_scores: list[float] = field(default_factory=list)

    @property
    def rouge_1_mean(self) -> float:
        return sum(self.rouge_1_scores) / len(self.rouge_1_scores) if self.rouge_1_scores else 0.0

    @property
    def rouge_2_mean(self) -> float:
        return sum(self.rouge_2_scores) / len(self.rouge_2_scores) if self.rouge_2_scores else 0.0

    @property
    def rouge_l_mean(self) -> float:
        return sum(self.rouge_l_scores) / len(self.rouge_l_scores) if self.rouge_l_scores else 0.0

    @property
    def bertscore_f1_mean(self) -> float:
        return sum(self.bertscore_f1_scores) / len(self.bertscore_f1_scores) if self.bertscore_f1_scores else 0.0

    def to_dict(self) -> dict:
        return {
            "rouge_1_mean": round(self.rouge_1_mean, 4),
            "rouge_2_mean": round(self.rouge_2_mean, 4),
            "rouge_l_mean": round(self.rouge_l_mean, 4),
            "bertscore_f1_mean": round(self.bertscore_f1_mean, 4),
            "n_samples": len(self.rouge_l_scores),
        }


def compute_rouge_all(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predictions: List of generated texts (SLM outputs).
        references: List of reference texts (oracle LLM outputs or gold answers).

    Returns:
        Dictionary with mean scores for 'rouge1', 'rouge2', and 'rougeL'.

    Reference: Lin (2004), ACL Workshop on Text Summarization Branches Out.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Install rouge-score: uv add rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            results["rouge1"].append(0.0)
            results["rouge2"].append(0.0)
            results["rougeL"].append(0.0)
            continue
        scores = scorer.score(ref, pred)
        results["rouge1"].append(scores["rouge1"].fmeasure)
        results["rouge2"].append(scores["rouge2"].fmeasure)
        results["rougeL"].append(scores["rougeL"].fmeasure)

    return {
        "rouge_1": sum(results["rouge1"]) / len(results["rouge1"]) if results["rouge1"] else 0.0,
        "rouge_2": sum(results["rouge2"]) / len(results["rouge2"]) if results["rouge2"] else 0.0,
        "rouge_l": sum(results["rougeL"]) / len(results["rougeL"]) if results["rougeL"] else 0.0,
    }


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    """
    Compute ROUGE-L F1 between each prediction and its reference.

    Args:
        predictions: List of generated texts (SLM outputs).
        references: List of reference texts (oracle LLM outputs or gold answers).

    Returns:
        List of ROUGE-L F1 scores (0.0–1.0) per pair.

    Reference: Lin (2004), ACL Workshop on Text Summarization Branches Out.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Install rouge-score: uv add rouge-score")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return scores


def compute_bertscore_f1(
    predictions: list[str],
    references: list[str],
    lang: str = "pt",
    batch_size: int = 32,
) -> list[float]:
    """
    Compute BERTScore F1 between predictions and references.

    Uses bert-base-multilingual-cased explicitly to avoid
    ``BertTokenizer has no attribute build_inputs_with_special_tokens``
    incompatibility in newer transformers versions (>=4.40) when bert-score
    auto-selects a model via lang= parameter.

    Args:
        predictions: List of generated texts (SLM outputs).
        references: List of reference texts.
        lang: Kept for API compatibility; model_type overrides auto-selection.
        batch_size: Batch size for encoding (reduce if OOM).

    Returns:
        List of BERTScore F1 values (0.0–1.0) per pair.

    Reference: Zhang et al. (2020), BERTScore: Evaluating Text Generation with BERT. ICLR 2020.
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        raise ImportError("Install bert-score: uv add bert-score")

    if not predictions or not references:
        return []

    # Replace empty/whitespace-only strings to avoid tokenizer edge cases
    empty_mask = [not (p and p.strip()) for p in predictions]
    safe_preds = [p if p and p.strip() else "." for p in predictions]
    safe_refs = [r if r and r.strip() else "." for r in references]

    try:
        _, _, F1 = bert_score(
            safe_preds,
            safe_refs,
            # Explicit model avoids lang= auto-selection bug in bert-score 0.3.13
            # with transformers >=4.40 (BertTokenizer API change).
            model_type="bert-base-multilingual-cased",
            num_layers=9,  # optimal layer for bert-base-multilingual-cased (ICLR 2020)
            batch_size=batch_size,
            verbose=False,
        )
        scores = F1.tolist()
        # Zero out scores for originally empty predictions
        for i, was_empty in enumerate(empty_mask):
            if was_empty:
                scores[i] = 0.0
        return scores
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}. Returning zeros.")
        return [0.0] * len(predictions)


def evaluate_generation(
    predictions: list[str],
    references: list[str],
    lang: str = "pt",
) -> GenerationMetrics:
    """
    Compute all generation metrics in one call.

    This is the primary entry point for SLM comparison experiments.
    references = oracle LLM outputs (Llama 3.3 70B) when gold answers unavailable,
    following the LLM-as-reference methodology (cf. Zheng et al. 2023 NeurIPS).

    Args:
        predictions: SLM-generated answers.
        references: Oracle LLM answers (E6 outputs) used as reference for H5.
        lang: Language for BERTScore. Default "pt" for Portuguese.

    Returns:
        GenerationMetrics with per-sample scores and aggregated means.
    """
    logger.info(f"Computing generation metrics for {len(predictions)} samples...")

    # Compute all ROUGE variants (per-sample)
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("Install rouge-score: uv add rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            rouge_1_scores.append(0.0)
            rouge_2_scores.append(0.0)
            rouge_l_scores.append(0.0)
            continue
        scores = scorer.score(ref, pred)
        rouge_1_scores.append(scores["rouge1"].fmeasure)
        rouge_2_scores.append(scores["rouge2"].fmeasure)
        rouge_l_scores.append(scores["rougeL"].fmeasure)

    logger.info(f"ROUGE-1 mean: {sum(rouge_1_scores)/len(rouge_1_scores):.4f}")
    logger.info(f"ROUGE-2 mean: {sum(rouge_2_scores)/len(rouge_2_scores):.4f}")
    logger.info(f"ROUGE-L mean: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")

    # Compute BERTScore
    bert_scores = compute_bertscore_f1(predictions, references, lang=lang)
    logger.info(f"BERTScore F1 mean: {sum(bert_scores)/len(bert_scores):.4f}")

    return GenerationMetrics(
        rouge_1_scores=rouge_1_scores,
        rouge_2_scores=rouge_2_scores,
        rouge_l_scores=rouge_l_scores,
        bertscore_f1_scores=bert_scores,
    )
