#!/usr/bin/env python3
"""
analyze_significance.py — Statistical analysis of SLM-RAG generation experiments.

Computes (using per-query faithfulness from checkpoint files):
  1. Bootstrap 95% CI for mean Faithfulness per model (n_iterations=1000)
  2. Wilcoxon signed-rank test for all model pairs (paired, same queries)
  3. Cohen's d effect size for each significant pair
  4. Judge fallback rate: fraction of per-query faith==0.5 exactly

Results saved to results/significance/<exp_id>_significance.json and printed
as a summary table ready to paste into the paper.

Usage:
  uv run python analyze_significance.py --exp E1a
  uv run python analyze_significance.py --exp E1a E2a E3a E4a
  uv run python analyze_significance.py --exp E1a E2a --alpha 0.01
"""
import sys
import json
import argparse
import random
import math
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent / "src"))

CKPT_DIR = Path("results/checkpoints")
OUT_DIR  = Path("results/significance")

EXPERIMENTS = {
    "E1a": "BM25+Ulysses",
    "E1b": "BGE-M3+Ulysses",
    "E2a": "BM25+JurisTCU",
    "E2b": "BGE-M3+JurisTCU",
    "E3a": "BM25+BR-TaxQA",
    "E3b": "BGE-M3+BR-TaxQA",
    "E4a": "BM25+NormasTCU",
    "E4b": "BGE-M3+NormasTCU",
}

SLM_ORDER = [
    "Llama_1B", "Llama_3B", "Gemma3_4B", "Qwen25_7B",
    "Qwen3_8B", "Qwen3_8B_NT", "Ministral_8B", "Phi4_14B", "Gemma4_31B",
]


def _safe_label(label: str) -> str:
    return label.replace("/", "__").replace("-", "_").replace(".", "_")


def load_faith_scores(exp_id: str) -> dict[str, list[float]]:
    """Load per-query faithfulness scores for all models in an experiment."""
    scores: dict[str, list[float]] = {}
    for label_safe in SLM_ORDER:
        path = CKPT_DIR / f"{exp_id}_{label_safe}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        qids  = data["query_ids"]
        faith = data["faithfulness"]
        # Store as ordered dict by query_id for paired alignment
        scores[label_safe] = dict(zip(qids, faith))
    return scores


def align_pairs(a: dict, b: dict) -> tuple[list[float], list[float]]:
    """Return two lists aligned by shared query_ids (paired test requirement)."""
    common = sorted(set(a) & set(b))
    return [a[q] for q in common], [b[q] for q in common]


def bootstrap_ci(values: list[float], n_iter: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap percentile CI. Returns (mean, ci_low, ci_high)."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_iter)]
    hi = means[int((1 - alpha / 2) * n_iter)]
    return sum(values) / n, lo, hi


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test (two-sided) via scipy.stats.wilcoxon.
    Returns (W_statistic, p_value).
    """
    from scipy import stats
    diffs = [xi - yi for xi, yi in zip(x, y)]
    non_zero = [d for d in diffs if d != 0.0]
    n = len(non_zero)
    if n < 10:
        return float("nan"), float("nan")
    try:
        result = stats.wilcoxon(x, y, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def cohens_d(x: list[float], y: list[float]) -> float:
    """Cohen's d for paired observations: d = mean(diff) / std(diff)."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d) if var_d > 0 else 0.0
    return mean_d / std_d if std_d > 0 else float("nan")


def fallback_rate(scores: dict[str, float]) -> float:
    vals = list(scores.values())
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v == 0.5) / len(vals)


def analyze_experiment(exp_id: str, alpha: float) -> dict:
    label = EXPERIMENTS.get(exp_id, exp_id)
    print(f"\n{'='*72}")
    print(f"  {exp_id}: {label}  (α={alpha})")
    print(f"{'='*72}")

    scores_dict = load_faith_scores(exp_id)
    if not scores_dict:
        print(f"  ❌ No checkpoints found in {CKPT_DIR}/")
        return {}

    model_names = list(scores_dict.keys())

    # 1. Bootstrap CI + fallback rate per model
    print(f"\n{'─'*72}")
    print(f"  {'Model':<16} {'Mean':>7} {'95% CI':>18} {'Fallback':>9} {'n':>5}")
    print(f"  {'─'*60}")

    ci_results: dict[str, dict] = {}
    for name in SLM_ORDER:
        if name not in scores_dict:
            continue
        vals = list(scores_dict[name].values())
        mean, lo, hi = bootstrap_ci(vals)
        fb = fallback_rate(scores_dict[name])
        n  = len(vals)
        print(f"  {name:<16} {mean:>7.4f} [{lo:.4f}, {hi:.4f}] {fb:>8.1%} {n:>5}")
        ci_results[name] = {
            "mean": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "fallback_rate": round(fb, 4),
            "n": n,
        }

    # 2. Wilcoxon pairwise + Cohen's d
    print(f"\n  Wilcoxon signed-rank pairwise (significant at α={alpha}):")
    print(f"  {'Pair':<33} {'W':>8} {'p':>10} {'d':>7} {'sig':>4}")
    print(f"  {'─'*60}")

    pairwise: list[dict] = []
    for a_name, b_name in combinations(model_names, 2):
        a_scores, b_scores = align_pairs(scores_dict[a_name], scores_dict[b_name])
        if len(a_scores) < 10:
            continue
        W, p = wilcoxon_signed_rank(a_scores, b_scores)
        d = cohens_d(a_scores, b_scores)
        if math.isnan(p):
            continue
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < alpha else ""))
        pair_label = f"{a_name} vs {b_name}"
        print(f"  {pair_label:<33} {W:>8.1f} {p:>10.4f} {d:>7.3f} {sig:>4}")
        pairwise.append({
            "model_a": a_name,
            "model_b": b_name,
            "W": round(W, 2) if not math.isnan(W) else None,
            "p_value": round(p, 6) if not math.isnan(p) else None,
            "cohens_d": round(d, 4) if not math.isnan(d) else None,
            "significant": p < alpha,
            "stars": sig,
            "n_pairs": len(a_scores),
        })

    result = {
        "experiment": exp_id,
        "label": label,
        "alpha": alpha,
        "bootstrap_n_iterations": 1000,
        "models": ci_results,
        "pairwise_wilcoxon": pairwise,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{exp_id}_significance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Wilcoxon + bootstrap CI for SLM-RAG faithfulness scores"
    )
    parser.add_argument(
        "--exp", nargs="+", choices=list(EXPERIMENTS), required=True,
        help="Experiments to analyse: E1a E2a E3a E4a ..."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold (default: 0.05)"
    )
    args = parser.parse_args()

    all_results = {}
    for exp_id in args.exp:
        all_results[exp_id] = analyze_experiment(exp_id, args.alpha)

    # Cross-experiment summary: per model, mean ± CI across experiments
    if len(args.exp) > 1:
        print(f"\n{'='*72}")
        print(f"  CROSS-EXPERIMENT SUMMARY — mean Faithfulness")
        print(f"{'='*72}")
        print(f"  {'Model':<16}", end="")
        for exp_id in args.exp:
            print(f"  {exp_id:>12}", end="")
        print()
        print(f"  {'─'*60}")
        for name in SLM_ORDER:
            row = f"  {name:<16}"
            for exp_id in args.exp:
                m = all_results.get(exp_id, {}).get("models", {}).get(name)
                if m:
                    row += f"  {m['mean']:>6.4f}±{m['ci_high']-m['ci_low']:>.4f}"
                else:
                    row += f"  {'—':>12}"
            print(row)

    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    main()
