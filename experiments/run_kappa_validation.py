"""
Multi-judge Fleiss' Kappa validation for faithfulness evaluation.

Validates inter-rater agreement among 3 LLM judges on a sample of queries.
Uses Landis & Koch (1977) scale: κ ≥ 0.61 = substantial agreement.

References:
  - Judge's Verdict (arXiv:2510.09738, 2025): Llama-3.3-70B κ=0.786,
    Qwen2.5-72B κ=0.786, Gemma-3-27B κ=0.812 vs. human raters
  - Fleiss (1971), Landis & Koch (1977)

Usage:
    uv run python run_kappa_validation.py --dataset ulysses --retriever hybrid --n-sample 150
    uv run python run_kappa_validation.py --dataset juristcu --retriever bge-m3 --n-sample 150
    uv run python run_kappa_validation.py  # all combinations, 100 samples each
"""
import sys
import json
import argparse
import random
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.generation.rag import DeepInfraRAG, OpenRouterRAG
from mestrado.retrieval.dense import DenseRetriever
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.hybrid import HybridRetriever
from mestrado.reranking.llm_reranker import RerankerResult
from mestrado.metrics.kappa import compute_kappa_for_judges, kappa_validation_summary

CHECKPOINT_DIR = Path("results/checkpoints")
KAPPA_DIR = Path("results/kappa")
BGE_M3_MODEL = "BAAI/bge-m3"
CACHE_DIR = Path("results/cache")

# Reference SLM for Kappa validation (best Faithfulness SLM in prior experiments)
REFERENCE_MODEL = "google/gemma-3-4b-it"

# Three judges validated by Judge's Verdict (arXiv:2510.09738, 2025)
JUDGE_MODELS = [
    ("meta-llama/Llama-3.3-70B-Instruct", "deepinfra"),  # J1 κ=0.786
    ("qwen/qwen2.5-72b-instruct",          "openrouter"), # J2 κ=0.786
    ("google/gemma-3-27b-it",              "deepinfra"),  # J3 κ=0.812
]

RETRY_DELAYS = [5, 15, 30]
MAX_WORKERS = 20


def _safe_name(model: str) -> str:
    return model.replace("/", "__").replace("-", "_")


def get_rag_instance(model: str, provider: str):
    if provider == "openrouter":
        return OpenRouterRAG(model=model)
    return DeepInfraRAG(model=model)


def evaluate_faithfulness_raw(context: str, response: str, judge_rag) -> float:
    """Evaluate faithfulness given pre-built context string. Returns float 0-1."""
    prompt = f"""Avalie a fidelidade da resposta aos documentos fornecidos.

Contexto (documentos recuperados):
{context[:1500]}

Resposta gerada:
{response[:800]}

Retorne APENAS um número de 0.0 a 1.0 onde:
- 0.0 = completamente infiel (inventou informações)
- 0.5 = parcialmente fiel (algumas informações corretas, outras não)
- 1.0 = completamente fiel (todas as informações vêm do contexto)

Score:"""
    try:
        resp = judge_rag._client.chat.completions.create(
            model=judge_rag.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r'\b(0\.\d+|1\.0|0\.0)\b', content)
        if m:
            return float(m.group(1))
        m = re.search(r'(\d+)%', content)
        if m:
            return float(m.group(1)) / 100.0
        cl = content.lower()
        if "completamente fiel" in cl or "totalmente fiel" in cl:
            return 1.0
        if "parcialmente fiel" in cl:
            return 0.5
        if "infiel" in cl:
            return 0.0
    except Exception as e:
        print(f"    ⚠️  Judge error: {e}")
    return 0.5


def build_context_string(bills_retrieved: list) -> str:
    """Build the same context string used during generation."""
    parts = []
    for bill in bills_retrieved:
        name = getattr(bill, "name", getattr(bill, "bill_id", ""))
        text = getattr(bill, "txt_ementa", getattr(bill, "excerto", ""))
        parts.append(f"### {name}\n{text}")
    return "\n\n---\n\n".join(parts)


def run_kappa_for_config(dataset: str, retriever_name: str, n_sample: int, seed: int):
    print(f"\n{'='*70}")
    print(f"KAPPA VALIDATION | dataset={dataset.upper()} | retriever={retriever_name.upper()} | n={n_sample}")
    print(f"{'='*70}")

    # ── Load checkpoint for reference model ─────────────────────────────────
    ckpt_path = CHECKPOINT_DIR / f"{dataset}_{retriever_name}__{_safe_name(REFERENCE_MODEL)}.json"
    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"     Run main experiment first, then retry.")
        return None

    with open(ckpt_path, encoding="utf-8") as f:
        ckpt = json.load(f)

    query_ids = ckpt["query_ids"]
    responses  = ckpt["responses"]
    id_to_response = dict(zip(query_ids, responses))
    print(f"  ✅ Loaded {len(query_ids)} responses from checkpoint")

    # ── Sample queries ───────────────────────────────────────────────────────
    rng = random.Random(seed)
    sample_ids = rng.sample(query_ids, min(n_sample, len(query_ids)))
    print(f"  🎲 Sampled {len(sample_ids)} queries (seed={seed})")

    # ── Load corpus + build retriever ────────────────────────────────────────
    loader = DataLoader()
    if dataset == "ulysses":
        bills, queries_all = loader.load_ulysses()
        queries = [q for q in queries_all if q.query_id in set(sample_ids)]
    else:
        bills, queries_all = loader.load_juristcu()
        queries = [q for q in queries_all if q.query_id in set(sample_ids)]

    print(f"  📚 Corpus: {len(bills)} documents | {len(queries)} sampled queries")

    # Build retriever
    bm25_ret = BM25Retriever(bills)
    bm25_ret.build_index()

    if retriever_name in ("hybrid", "bge-m3"):
        cache = CACHE_DIR / dataset
        cache.mkdir(parents=True, exist_ok=True)
        dense_ret = DenseRetriever(bills, model_name=BGE_M3_MODEL)
        dense_ret.build_index(cache_path=cache / "bge-m3.pkl")
    else:
        dense_ret = None

    if retriever_name == "hybrid":
        retriever = HybridRetriever(bm25_ret, dense_ret, bill_key="name")
    elif retriever_name == "bge-m3":
        retriever = dense_ret
    else:
        retriever = bm25_ret

    # ── Retrieve for sample queries ──────────────────────────────────────────
    print(f"\n  🔍 Retrieving for {len(queries)} sampled queries...")
    contexts = {}
    for q in tqdm(queries, desc="retrieval"):
        retrieved = retriever.retrieve(q.text, top_k=5)
        contexts[q.query_id] = build_context_string([r.bill for r in retrieved])

    # ── Instantiate judges ───────────────────────────────────────────────────
    judge_rags = []
    for model, provider in JUDGE_MODELS:
        judge_rags.append((model, get_rag_instance(model, provider)))
        print(f"  🧑‍⚖️  Judge ready: {model.split('/')[-1]}")

    # ── Evaluate with all 3 judges ───────────────────────────────────────────
    print(f"\n  📊 Evaluating {len(queries)} × {len(judge_rags)} judges...")

    judge_scores: list[list[float]] = [[] for _ in judge_rags]

    def _eval_one(q, judge_idx, judge_rag):
        ctx  = contexts[q.query_id]
        resp = id_to_response.get(q.query_id, "")
        return judge_idx, evaluate_faithfulness_raw(ctx, resp, judge_rag)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []
        for q in queries:
            for j_idx, (_, j_rag) in enumerate(judge_rags):
                futures.append(pool.submit(_eval_one, q, j_idx, j_rag))

        pbar = tqdm(total=len(futures), desc="judging")
        # results indexed by query position
        query_pos = {q.query_id: i for i, q in enumerate(queries)}
        per_query_per_judge = [[None] * len(queries) for _ in judge_rags]

        for future in as_completed(futures):
            j_idx, score = future.result()
            # We can't easily map back to query without refactoring — collect in order
            judge_scores[j_idx].append(score)
            pbar.update(1)
        pbar.close()

    # Reorder: collect scores query-by-query per judge using a simpler sequential pass
    judge_scores = [[] for _ in judge_rags]
    print("  🔄 Re-running sequentially for ordered Kappa matrix...")
    for q in tqdm(queries, desc="sequential eval"):
        ctx  = contexts[q.query_id]
        resp = id_to_response.get(q.query_id, "")
        for j_idx, (_, j_rag) in enumerate(judge_rags):
            score = evaluate_faithfulness_raw(ctx, resp, j_rag)
            judge_scores[j_idx].append(score)
            time.sleep(0.05)

    # ── Compute Fleiss' Kappa ────────────────────────────────────────────────
    kappa = compute_kappa_for_judges(
        judge_scores,
        bins=[0.0, 0.33, 0.67, 1.01],  # Low / Medium / High
    )
    summary = kappa_validation_summary(kappa, n_items=len(queries), n_raters=len(judge_rags))

    print(f"\n  {'─'*50}")
    print(f"  Fleiss' Kappa = {kappa:.4f}  ({summary['interpretation']})")
    print(f"  Status: {summary['status']}  (target: κ ≥ 0.61)")

    for j_idx, (model, _) in enumerate(judge_rags):
        scores = judge_scores[j_idx]
        label  = model.split("/")[-1]
        print(f"    J{j_idx+1} {label:<30} mean={sum(scores)/len(scores):.3f}")

    # ── Save results ─────────────────────────────────────────────────────────
    KAPPA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = KAPPA_DIR / f"kappa_{dataset}_{retriever_name}.json"
    result = {
        "dataset":          dataset,
        "retriever":        retriever_name,
        "reference_model":  REFERENCE_MODEL,
        "n_sample":         len(queries),
        "seed":             seed,
        "kappa":            round(kappa, 4),
        "interpretation":   summary["interpretation"],
        "status":           summary["status"],
        "judges": [
            {
                "model":    model,
                "provider": provider,
                "mean_score": round(sum(judge_scores[j]) / len(judge_scores[j]), 4),
                "scores":   [round(s, 3) for s in judge_scores[j]],
            }
            for j, (model, provider) in enumerate(JUDGE_MODELS)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved → {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["ulysses", "juristcu"], default=None)
    parser.add_argument("--retriever", choices=["hybrid", "bge-m3", "bm25"], default=None)
    parser.add_argument("--n-sample",  type=int, default=100)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    datasets   = [args.dataset]   if args.dataset   else ["ulysses", "juristcu"]
    retrievers = [args.retriever] if args.retriever else ["hybrid", "bge-m3", "bm25"]

    all_results = []
    for dataset in datasets:
        for retriever in retrievers:
            r = run_kappa_for_config(dataset, retriever, args.n_sample, args.seed)
            if r:
                all_results.append(r)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in all_results:
        print(f"  {r['dataset']:10s} {r['retriever']:8s}  κ={r['kappa']:.3f}  {r['interpretation']}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
