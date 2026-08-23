"""
BEIR Pipeline — Retrieval + Generation for JUÁ HuggingFace datasets.

Runs the full SLM-RAG pipeline on any BEIR-format legal dataset:
  - ufca-llms/juris-tcu   (TCU acórdãos, EXCERTO-only, 0-3 scale)
  - ufca-llms/normas-tcu  (TCU regulatory norms, 0-2 scale)
  - ufca-llms/jua         (JUÁ aggregate, binary relevance)
  - ufca-llms/BR-TaxQA    (Brazilian tax law FAQ, 1-2 scale)

The script produces the same checkpoint format as run_experimento_juristcu.py,
so run_analise_estatistica.py can aggregate results across all datasets.

Usage:
    uv run python run_beir_pipeline.py --dataset juris-tcu
    uv run python run_beir_pipeline.py --dataset normas-tcu --retrievers hybrid bm25
    uv run python run_beir_pipeline.py --dataset br-taxqa --n-queries 100 --seed 42
    uv run python run_beir_pipeline.py --dataset jua --skip-generation

Checkpoints saved to:
    results/checkpoints/{dataset}_{retriever}__{model}.json
"""
import sys
import json
import time
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.beir_loader import BEIRLoader
from mestrado.generation.rag import DeepInfraRAG, OpenRouterRAG
from mestrado.retrieval.dense import DenseRetriever
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.hybrid import HybridRetriever
from mestrado.metrics.generation_metrics import evaluate_generation
from mestrado.metrics.relevance_metrics import evaluate_relevance
from mestrado.metrics.ir_metrics import ndcg_at_k, mrr, recall_at_k, average_precision
from mestrado.reranking.llm_reranker import RerankerResult

# Same SLM stack as Ulysses and JurisTCU for cross-dataset consistency
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
    "qwen/qwen3-8b",
    "microsoft/phi-4",
    "google/gemma-4-26B-A4B-it",
    "meta-llama/Llama-3.3-70B-Instruct",   # Oracle
]

ORACLE_MODEL = "qwen/qwen3.5-397b-a17b"
JUDGE_MODEL  = "meta-llama/Llama-3.3-70B-Instruct"

CHECKPOINT_DIR = Path("results/checkpoints")
CACHE_DIR      = Path("results/cache")
BGE_M3_MODEL   = "BAAI/bge-m3"

RETRY_DELAYS       = [5, 15, 30]
MAX_WORKERS_DI     = 40
MAX_WORKERS_OR     = 20
MAX_PARALLEL_MODELS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rag(model: str):
    if "qwen/" in model.lower() or "deepseek/" in model.lower():
        return OpenRouterRAG(model=model)
    return DeepInfraRAG(model=model)


def _safe_name(model: str) -> str:
    return model.replace("/", "__").replace("-", "_")


def _ckpt_key(dataset_key: str, retriever: str, model: str) -> Path:
    return CHECKPOINT_DIR / f"{dataset_key}_{retriever}__{_safe_name(model)}.json"


def load_ckpt(dataset_key: str, retriever: str, model: str) -> dict | None:
    p = _ckpt_key(dataset_key, retriever, model)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_ckpt(dataset_key: str, retriever: str, model: str, data: dict):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _ckpt_key(dataset_key, retriever, model)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_with_retry(rag, query_text: str, reranked, top_bills: int = 5):
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            result = rag.generate(query_text, reranked, top_bills=top_bills)
            if result.tokens_generated == 0 or str(result.response).startswith("[Erro"):
                raise ValueError(f"Empty response: {str(result.response)[:60]}")
            return result
        except Exception as e:
            print(f"    ⚠️  attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == len(RETRY_DELAYS):
                return None
    return None


def evaluate_faithfulness(result, judge_rag) -> float:
    context = "\n\n---\n\n".join(
        f"### {b.name}\n{b.txt_ementa}" for b in result.bills_used
    )
    prompt = f"""Avalie a fidelidade da resposta aos documentos fornecidos.

Contexto (documentos recuperados):
{context[:1500]}

Resposta gerada:
{result.response[:800]}

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
        print(f"    ⚠️  faithfulness error: {e}")
    return 0.5


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_ir(queries, retrieved_by_query: dict) -> dict:
    """Compute nDCG@5, nDCG@10, MRR@10, Recall@10, MAP over all queries."""
    n5, n10, mrr_l, r10_l, ap_l = [], [], [], [], []
    for q in queries:
        grade_map = q.graded_relevance()
        results   = retrieved_by_query.get(q.query_id, [])
        ids = [r.bill.name for r in results]
        n5.append(ndcg_at_k(ids, grade_map, k=5))
        n10.append(ndcg_at_k(ids, grade_map, k=10))
        mrr_l.append(mrr(ids, grade_map))
        r10_l.append(recall_at_k(ids, grade_map, k=10))
        ap_l.append(average_precision(ids, grade_map))
    avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0
    return {
        "nDCG@5":    avg(n5),
        "nDCG@10":   avg(n10),
        "MRR":       avg(mrr_l),
        "Recall@10": avg(r10_l),
        "MAP":       avg(ap_l),
        "_ndcg10":   n10,
    }


# ---------------------------------------------------------------------------
# Main pipeline per retriever
# ---------------------------------------------------------------------------

def run_for_retriever(
    dataset_key: str,
    retriever_name: str,
    bills: dict,
    queries: list,
    dense_ret,
    bm25_ret,
    skip_generation: bool,
):
    print(f"\n{'='*70}")
    print(f"RETRIEVER: {retriever_name.upper()} | dataset={dataset_key}")
    print(f"{'='*70}")

    # Build retriever
    if retriever_name == "hybrid":
        retriever = HybridRetriever(bm25_ret, dense_ret, bill_key="name")
    elif retriever_name == "bge-m3":
        retriever = dense_ret
    else:
        retriever = bm25_ret

    # Retrieve
    print(f"\n🔍 Retrieving {len(queries)} queries...")
    retrieved_by_query: dict = {}
    for q in tqdm(queries, desc=f"{retriever_name} retrieval"):
        retrieved_by_query[q.query_id] = retriever.retrieve(q.text, top_k=12)

    ir_metrics = evaluate_ir(queries, retrieved_by_query)
    ndcg10 = ir_metrics["nDCG@10"]
    print(f"  nDCG@5={ir_metrics['nDCG@5']:.4f}  nDCG@10={ndcg10:.4f}  "
          f"MRR={ir_metrics['MRR']:.4f}  R@10={ir_metrics['Recall@10']:.4f}  "
          f"MAP={ir_metrics['MAP']:.4f}")

    if skip_generation:
        return {"retriever": retriever_name, **ir_metrics}

    # Generation
    query_ids = [q.query_id for q in queries]
    judge_rag = get_rag(JUDGE_MODEL)

    # Oracle
    print(f"\n🔴 Oracle: {ORACLE_MODEL}")
    oracle_ckpt = load_ckpt(dataset_key, retriever_name, ORACLE_MODEL)

    if oracle_ckpt and oracle_ckpt.get("query_ids") == query_ids:
        oracle_responses = dict(zip(oracle_ckpt["query_ids"], oracle_ckpt["responses"]))
        print(f"  ♻️  Oracle from checkpoint ({len(oracle_responses)} responses)")
    else:
        oracle_rag = get_rag(ORACLE_MODEL)
        oracle_responses = {}

        def _oracle_query(q):
            reranked = [RerankerResult(bill=r.bill, rank=i + 1)
                        for i, r in enumerate(retrieved_by_query[q.query_id][:5])]
            result = generate_with_retry(oracle_rag, q.text, reranked)
            return q.query_id, result.response if result else ""

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_OR) as pool:
            futures = {pool.submit(_oracle_query, q): q for q in queries}
            pbar = tqdm(total=len(queries), desc="Oracle")
            for future in as_completed(futures):
                qid, resp = future.result()
                oracle_responses[qid] = resp
                pbar.update(1)
            pbar.close()

        save_ckpt(dataset_key, retriever_name, ORACLE_MODEL, {
            "query_ids": query_ids,
            "responses": [oracle_responses[qid] for qid in query_ids],
        })
        print(f"  ✅ Oracle done")

    # SLMs
    slm_models = [m for m in MODELS if m != ORACLE_MODEL]
    results: dict = {}

    def _run_model(model: str):
        label    = model.split("/")[-1]
        provider = "OpenRouter" if "qwen/" in model or "deepseek/" in model else "DeepInfra"

        ckpt = load_ckpt(dataset_key, retriever_name, model)
        if ckpt and ckpt.get("query_ids") == query_ids and len(ckpt.get("responses", [])) == len(queries):
            print(f"  ♻️  [{label}] checkpoint — skipping")
            return model, {
                "responses":           ckpt["responses"],
                "tokens_in_avg":       ckpt["tokens_in_avg"],
                "tokens_out_avg":      ckpt["tokens_out_avg"],
                "latency_avg":         ckpt["latency_avg"],
                "faithfulness_avg":    ckpt["faithfulness_avg"],
                "faithfulness_scores": ckpt.get("faithfulness_scores", []),
                "provider":            provider,
            }

        rag = get_rag(model)
        n_workers = MAX_WORKERS_OR if ("qwen/" in model or "deepseek/" in model) else MAX_WORKERS_DI

        def _run_query(q):
            reranked = [RerankerResult(bill=r.bill, rank=i + 1)
                        for i, r in enumerate(retrieved_by_query[q.query_id][:5])]
            result = generate_with_retry(rag, q.text, reranked)
            if result is None:
                return q.query_id, "", 0, 0, 0.0, 0.5
            faith = evaluate_faithfulness(result, judge_rag)
            return (q.query_id, result.response, result.tokens_context,
                    result.tokens_generated, result.latency_s, faith)

        qr: dict = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures_map = {pool.submit(_run_query, q): q for q in queries}
            pbar = tqdm(total=len(queries), desc=f"  {label[:30]}", leave=False)
            for future in as_completed(futures_map):
                qid, resp, tok_in, tok_out, lat, faith = future.result()
                qr[qid] = (resp, tok_in, tok_out, lat, faith)
                pbar.update(1)
            pbar.close()

        responses, toks_in, toks_out, lats, faiths = [], [], [], [], []
        for q in queries:
            resp, ti, to, lat, faith = qr[q.query_id]
            responses.append(resp); toks_in.append(ti); toks_out.append(to)
            lats.append(lat); faiths.append(faith)

        n = len(responses)
        model_result = {
            "responses":           responses,
            "tokens_in_avg":       sum(toks_in) / n,
            "tokens_out_avg":      sum(toks_out) / n,
            "latency_avg":         sum(lats) / n,
            "faithfulness_avg":    sum(faiths) / n,
            "faithfulness_scores": faiths,
            "provider":            provider,
        }
        save_ckpt(dataset_key, retriever_name, model, {
            "query_ids":           query_ids,
            "responses":           responses,
            "tokens_in_avg":       model_result["tokens_in_avg"],
            "tokens_out_avg":      model_result["tokens_out_avg"],
            "latency_avg":         model_result["latency_avg"],
            "faithfulness_avg":    model_result["faithfulness_avg"],
            "faithfulness_scores": faiths,
        })
        print(f"  ✅ [{label}] Faith={model_result['faithfulness_avg']:.3f}")
        return model, model_result

    print(f"\n🔵 SLMs ({MAX_PARALLEL_MODELS} in parallel)...")
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_MODELS) as pool:
        futures_map = {pool.submit(_run_model, m): m for m in slm_models}
        for future in as_completed(futures_map):
            model, model_result = future.result()
            results[model] = model_result

    # BERTScore vs Oracle
    print("\n📊 Computing BERTScore vs Oracle...")
    oracle_refs = [oracle_responses.get(q.query_id, "") for q in queries]
    for model in slm_models:
        metrics = evaluate_generation(
            predictions=results[model]["responses"],
            references=oracle_refs,
            lang="pt",
        )
        results[model]["rouge_1"]   = metrics.rouge_1_mean
        results[model]["rouge_l"]   = metrics.rouge_l_mean
        results[model]["bertscore"] = metrics.bertscore_f1_mean

    oracle_bert = evaluate_generation(oracle_refs, oracle_refs, lang="pt").bertscore_f1_mean
    for model in slm_models:
        results[model]["bertscore_pct"] = (
            results[model]["bertscore"] / oracle_bert * 100.0 if oracle_bert > 0 else 0.0
        )

    # Summary table
    print(f"\n{'='*70}")
    print(f"RESULTS | {dataset_key.upper()} | {retriever_name.upper()} | nDCG@10={ndcg10:.3f}")
    print(f"{'='*70}")
    print(f"{'Model':<38} {'BERT':>6} {'%O':>5} {'Faith':>6}")
    print("-" * 58)
    sorted_models = sorted(slm_models, key=lambda m: results[m].get("bertscore", 0), reverse=True)
    for rank, model in enumerate(sorted_models, 1):
        name  = model.split("/")[-1][:36]
        medal = f"{rank}." if rank <= 3 else "  "
        r = results[model]
        print(f"{medal} {name:<36} {r.get('bertscore', 0):>6.3f} "
              f"{r.get('bertscore_pct', 0):>4.0f}% {r.get('faithfulness_avg', 0):>6.3f}")

    return {"retriever": retriever_name, **ir_metrics, "results": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BEIR pipeline for JUÁ legal datasets")
    parser.add_argument(
        "--dataset", required=True,
        choices=["juris-tcu", "normas-tcu", "jua", "br-taxqa"],
        help="BEIR dataset to evaluate",
    )
    parser.add_argument(
        "--retrievers", nargs="+",
        choices=["hybrid", "bge-m3", "bm25"],
        default=["hybrid", "bge-m3", "bm25"],
        help="Retrieval paradigms to evaluate (default: all three)",
    )
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Sample N queries (default: use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-doc-chars", type=int, default=50_000,
                        help="Max document characters (default 50k; lower for normas-tcu)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Run retrieval evaluation only (no RAG generation)")
    args = parser.parse_args()

    # Override for normas-tcu which has very large documents
    if args.dataset == "normas-tcu" and args.max_doc_chars == 50_000:
        args.max_doc_chars = 30_000
        print(f"  ℹ️  normas-tcu: auto-setting --max-doc-chars=30000 (large docs)")

    print(f"\n{'='*70}")
    print(f"BEIR PIPELINE | dataset={args.dataset} | retrievers={args.retrievers}")
    print(f"{'='*70}")

    # Load dataset
    loader = BEIRLoader(args.dataset, max_doc_chars=args.max_doc_chars)
    bills, queries = loader.load()

    if args.n_queries and args.n_queries < len(queries):
        import random
        rng = random.Random(args.seed)
        queries = rng.sample(queries, args.n_queries)
        print(f"  🎲 Sampled {len(queries)} queries (seed={args.seed})")

    dataset_key = args.dataset  # used as checkpoint prefix

    # Build indices (once, shared across retrievers)
    print("\n🔧 Building retrieval indices...")
    bm25_ret = BM25Retriever(bills)
    bm25_ret.build_index()
    print("  ✅ BM25 index built")

    dense_ret = None
    if any(r in args.retrievers for r in ("hybrid", "bge-m3")):
        cache = CACHE_DIR / dataset_key
        cache.mkdir(parents=True, exist_ok=True)
        dense_ret = DenseRetriever(bills, model_name=BGE_M3_MODEL)
        dense_ret.build_index(cache_path=cache / "bge-m3.pkl")
        print(f"  ✅ BGE-M3 index built ({len(bills):,} vectors)")

    # Run pipeline for each retriever
    all_results = []
    for retriever_name in args.retrievers:
        if retriever_name in ("hybrid", "bge-m3") and dense_ret is None:
            print(f"  ⚠️  Skipping {retriever_name}: dense index not built")
            continue
        result = run_for_retriever(
            dataset_key=dataset_key,
            retriever_name=retriever_name,
            bills=bills,
            queries=queries,
            dense_ret=dense_ret,
            bm25_ret=bm25_ret,
            skip_generation=args.skip_generation,
        )
        all_results.append(result)

    # Final summary
    print(f"\n{'='*70}")
    print(f"RETRIEVAL SUMMARY | {args.dataset.upper()}")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['retriever']:10s}  nDCG@10 = {r['ndcg10']:.4f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
