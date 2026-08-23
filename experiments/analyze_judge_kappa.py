#!/usr/bin/env python3
"""
analyze_judge_kappa.py -- Cohen's Kappa inter-rater agreement.

Compares Llama 3.3 70B (primary judge, scores in checkpoints) against
Gemini 2.5-flash (second judge, called here) on the same (context, response) pairs.

Steps:
  1. Load checkpoint for a model/experiment (Llama scores per query)
  2. Reconstruct contexts via retrieval (same pipeline as generation)
  3. Sample N queries randomly (seed=42)
  4. Call Gemini 2.5-flash on each (context, response) pair
  5. Binarize both score sets at threshold 0.5
  6. Compute Cohen's Kappa, agreement rate, per-category analysis
  7. Save results to results/kappa/<exp>_<model>_kappa.json

Requirements:
  - GEMINI_API_KEY in .env
  - openai Python package (already installed -- Gemini supports OpenAI-compatible API)

Usage:
  uv run python analyze_judge_kappa.py --exp E1a --model Ministral-8B --n 150
  uv run python analyze_judge_kappa.py --exp E1a --all-models --n 100
  uv run python analyze_judge_kappa.py --exp E1a E2a --all-models --n 150
"""
import sys
import os
import json
import re
import math
import time
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.data.beir_loader import BEIRLoader
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.dense import DenseRetriever

# ── Constants (mirrored from recalculate_faith.py) ─────────────────────────────
CKPT_DIR  = Path("results/checkpoints")
OUT_DIR   = Path("results/kappa")
CACHE_DIR = Path("results/cache")
CTX_CACHE_DIR = Path("results/kappa/ctx_cache")  # contextos pré-computados por exp
BGE_MODEL = "BAAI/bge-m3"
TOP_K_CONTEXT = 5
CHARS_PER_DOC = 2500

N_WORKERS = 8   # Gemini allows ~60 RPM on free tier; 8 workers = safe

EXPERIMENTS = {
    "E1a": dict(dataset="ulysses",   retriever="bm25",  label="BM25+Ulysses"),
    "E1b": dict(dataset="ulysses",   retriever="bgem3", label="BGE-M3+Ulysses"),
    "E2a": dict(dataset="juristcu",  retriever="bm25",  label="BM25+JurisTCU"),
    "E2b": dict(dataset="juristcu",  retriever="bgem3", label="BGE-M3+JurisTCU"),
    "E3a": dict(dataset="brtaxqa",   retriever="bm25",  label="BM25+BR-TaxQA"),
    "E3b": dict(dataset="brtaxqa",   retriever="bgem3", label="BGE-M3+BR-TaxQA"),
    "E4a": dict(dataset="normastcu", retriever="bm25",  label="BM25+NormasTCU"),
    "E4b": dict(dataset="normastcu", retriever="bgem3", label="BGE-M3+NormasTCU"),
}

SLM_LABELS = [
    "Llama-1B", "Llama-3B", "Gemma3-4B", "Qwen25-7B",
    "Qwen3-8B", "Qwen3-8B-NT", "Ministral-8B", "Phi4-14B", "Gemma4-31B",
]

FAITHFULNESS_PROMPT = """\
Avalie a fidelidade da resposta aos documentos fornecidos.

Contexto (documentos recuperados):
{context}

Resposta gerada:
{response}

Retorne APENAS um número de 0.0 a 1.0:
- 0.0 = completamente infiel (inventou informações não presentes nos documentos)
- 0.5 = parcialmente fiel (algumas informações corretas, outras inventadas)
- 1.0 = completamente fiel (todas as informações vêm dos documentos fornecidos)

Score:"""

GEMINI_BASE_URL = "https://api.deepinfra.com/v1/openai"
GEMINI_MODEL    = "google/gemini-2.5-flash"


# ── Utilities ──────────────────────────────────────────────────────────────────
def _safe(s: str) -> str:
    return s.replace("/", "__").replace("-", "_").replace(".", "_")


_FALLBACK_SAMPLES: list[str] = []   # filled on first parse failures, for diagnostics

def _parse_faith(content: str) -> float:
    # 1. Decimal score: 0.0-1.0 (dot or comma, any number of decimal digits)
    m = re.search(r'\b(1\.0+|0\.\d+|1,0+|0,\d+)\b', content)
    if m:
        return min(1.0, float(m.group(1).replace(",", ".")))
    # 2. Integer score 0 or 1
    m = re.search(r'\b([01])\b', content)
    if m:
        return float(m.group(1))
    # 3. Percentage
    m = re.search(r'(\d+)\s*%', content)
    if m:
        return min(1.0, float(m.group(1)) / 100.0)
    # 4. Keyword fallbacks
    cl = content.lower()
    if any(w in cl for w in ("completamente fiel", "totalmente fiel", "fully faithful", "completely faithful")):
        return 1.0
    if any(w in cl for w in ("parcialmente", "partially", "some")):
        return 0.5
    if any(w in cl for w in ("infiel", "não fiel", "unfaithful", "not faithful", "hallucin")):
        return 0.0
    # 5. True fallback -- log raw output for the first 3 occurrences so we can diagnose
    if len(_FALLBACK_SAMPLES) < 3:
        _FALLBACK_SAMPLES.append(repr(content[:300]))
        print(f"\n  [FALLBACK DEBUG] raw='{content[:300]}'")
    return 0.5


def build_context(bills: list, dataset: str) -> str:
    parts = []
    for bill in bills[:TOP_K_CONTEXT]:
        name = getattr(bill, "name", "")
        if dataset == "ulysses":
            text = getattr(bill, "txt_ementa", "") or getattr(bill, "text", "")
        else:
            text = getattr(bill, "text", "") or getattr(bill, "txt_ementa", "")
        text = text[:CHARS_PER_DOC]
        parts.append(f"### {name}\n{text}")
    return "\n\n---\n\n".join(parts)


def load_dataset(dataset_name: str):
    if dataset_name == "ulysses":
        loader = DataLoader()
        bills, queries = loader.load_all()
    elif dataset_name == "brtaxqa":
        bills, queries = BEIRLoader("br-taxqa").load()
    elif dataset_name == "normastcu":
        bills, queries = BEIRLoader("normas-tcu").load()
    else:
        bills, queries = BEIRLoader("juris-tcu").load()
    return bills, queries


def build_retriever(retriever_type: str, bills: dict, dataset_name: str):
    if retriever_type == "bm25":
        ret = BM25Retriever(bills)
        ret.build_index()
        return ret
    cache = CACHE_DIR / dataset_name
    cache.mkdir(parents=True, exist_ok=True)
    ret = DenseRetriever(bills, model_name=BGE_MODEL)
    ret.build_index(cache_path=cache / "bge-m3.pkl")
    return ret


def load_or_build_contexts(exp_id: str, dataset: str, ret_type: str,
                           needed_qids: set[str]) -> dict[str, str]:
    """
    Returns {query_id: context_str} for all needed_qids.

    Cache stored at results/kappa/ctx_cache/{exp_id}_contexts.json.
    On cache hit the expensive dataset load + retrieval is skipped entirely.
    The cache is additive: if new query_ids are needed, only those are computed
    and merged into the existing cache file.
    """
    CTX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CTX_CACHE_DIR / f"{exp_id}_contexts.json"

    # Load existing cache (may be partial or complete)
    cached: dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)

    missing = needed_qids - set(cached.keys())

    if not missing:
        print(f"  ✅ Contextos carregados do cache ({len(cached)} entradas) -- sem retrieval necessário")
        return {qid: cached[qid] for qid in needed_qids}

    print(f"  📦 Cache: {len(cached)} hits | {len(missing)} miss -> reconstruindo via retrieval...")

    bills, queries = load_dataset(dataset)
    retriever = build_retriever(ret_type, bills, dataset)

    for q in tqdm(queries, desc="contextos"):
        if str(q.query_id) not in missing and q.query_id not in missing:
            continue
        retrieved = retriever.retrieve(q.text, top_k=TOP_K_CONTEXT)
        top_bills = [r.bill for r in retrieved]
        cached[q.query_id] = build_context(top_bills, dataset)

    # Persist updated cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False)
    print(f"  💾 Cache salvo -> {cache_path} ({len(cached)} entradas)")

    return {qid: cached[qid] for qid in needed_qids if qid in cached}


# ── Second judge via DeepInfra (google/gemini-2.5-flash) ──────────────────────
def make_gemini_client():
    from openai import OpenAI
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPINFRA_API_KEY não encontrada no .env")
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def gemini_judge(client, context: str, response: str) -> float:
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:5000], response=response[:1500]
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return _parse_faith(resp.choices[0].message.content.strip())
        except Exception as e:
            wait = [15, 45, 90][attempt]
            print(f"  ⚠️  Gemini error (attempt {attempt+1}): {e}. Aguardando {wait}s...")
            time.sleep(wait)
    return 0.5


# ── Cohen's Kappa (pure Python) ────────────────────────────────────────────────
def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> dict:
    """
    Cohen's Kappa for binary labels (0/1).
    Returns dict with kappa, p_observed, p_expected, agreement_rate.
    """
    n = len(labels_a)
    assert n == len(labels_b) and n > 0

    # Confusion matrix
    tp = sum(1 for a, b in zip(labels_a, labels_b) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(labels_a, labels_b) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(labels_a, labels_b) if a == 1 and b == 0)

    p_o = (tp + tn) / n  # observed agreement

    # Marginal probabilities
    p_a1 = (tp + fn) / n   # P(A says faithful)
    p_b1 = (tp + fp) / n   # P(B says faithful)
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)  # expected agreement

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    # Bootstrap 95% CI for kappa (n=1000 resamples)
    rng = random.Random(42)
    kappas = []
    pairs = list(zip(labels_a, labels_b))
    for _ in range(1000):
        sample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        sa, sb = zip(*sample)
        sa, sb = list(sa), list(sb)
        n_ = len(sa)
        tp_ = sum(1 for a, b in zip(sa, sb) if a == 1 and b == 1)
        tn_ = sum(1 for a, b in zip(sa, sb) if a == 0 and b == 0)
        fp_ = sum(1 for a, b in zip(sa, sb) if a == 0 and b == 1)
        fn_ = sum(1 for a, b in zip(sa, sb) if a == 1 and b == 0)
        po_ = (tp_ + tn_) / n_
        pa1_ = (tp_ + fn_) / n_
        pb1_ = (tp_ + fp_) / n_
        pe_ = pa1_ * pb1_ + (1 - pa1_) * (1 - pb1_)
        k_ = (po_ - pe_) / (1 - pe_) if (1 - pe_) > 0 else 0.0
        kappas.append(k_)
    kappas.sort()
    ci_lo = kappas[25]
    ci_hi = kappas[974]

    return {
        "kappa": round(kappa, 4),
        "kappa_ci_lo": round(ci_lo, 4),
        "kappa_ci_hi": round(ci_hi, 4),
        "p_observed": round(p_o, 4),
        "p_expected": round(p_e, 4),
        "agreement_rate": round(p_o, 4),
        "n": n,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "interpretation": _kappa_label(kappa),
    }


def _kappa_label(k: float) -> str:
    if k >= 0.81: return "almost perfect"
    if k >= 0.61: return "substantial"
    if k >= 0.41: return "moderate"
    if k >= 0.21: return "fair"
    if k >= 0.00: return "slight"
    return "poor"


# ── Main analysis ──────────────────────────────────────────────────────────────
def analyze(exp_id: str, model_label: str, n_sample: int, seed: int) -> dict:
    cfg = EXPERIMENTS[exp_id]
    dataset  = cfg["dataset"]
    ret_type = cfg["retriever"]
    label    = cfg["label"]

    safe_model = _safe(model_label)
    ckpt_path  = CKPT_DIR / f"{exp_id}_{safe_model}.json"
    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint não encontrado: {ckpt_path}")
        return {}

    print(f"\n{'='*72}")
    print(f"  {exp_id} / {model_label}  |  {label}")
    print(f"  Judge A: Llama 3.3 70B (checkpoint)  |  Judge B: Gemini 2.5-flash")
    print(f"{'='*72}")

    with open(ckpt_path, encoding="utf-8") as f:
        ckpt = json.load(f)

    qids      = ckpt["query_ids"]
    responses = ckpt["responses"]
    llama_scores = ckpt["faithfulness"]

    qid2resp  = dict(zip(qids, responses))
    qid2faith = dict(zip(qids, llama_scores))

    # 1. Contexts (cached -- skip dataset load + retrieval on subsequent runs)
    print("\n📄 Contextos...")
    qid2ctx = load_or_build_contexts(exp_id, dataset, ret_type, set(qids))

    # 2. Sample
    available = [qid for qid in qids if qid in qid2ctx]
    rng = random.Random(seed)
    rng.shuffle(available)
    sample_ids = available[:n_sample]
    print(f"\n  Amostra: {len(sample_ids)}/{len(available)} queries (seed={seed})")

    # 4. Gemini judge (parallel)
    gemini_client = make_gemini_client()
    gemini_scores: dict[str, float] = {}
    _lock = threading.Lock()

    def _task(qid):
        ctx   = qid2ctx[qid]
        resp  = qid2resp.get(qid, "")
        score = gemini_judge(gemini_client, ctx, resp)
        return qid, score

    print(f"\n🤖 Chamando Gemini 2.5-flash ({len(sample_ids)} queries, {N_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_task, qid): qid for qid in sample_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Gemini judge"):
            qid, score = fut.result()
            with _lock:
                gemini_scores[qid] = score

    # 5. Align scores
    llama_list  = [qid2faith[qid] for qid in sample_ids]
    gemini_list = [gemini_scores[qid] for qid in sample_ids]

    # 6. Binarize (threshold > 0.5 -> faithful=1, else 0)
    THRESHOLD = 0.5
    llama_bin  = [1 if s > THRESHOLD else 0 for s in llama_list]
    gemini_bin = [1 if s > THRESHOLD else 0 for s in gemini_list]

    kappa_result = cohens_kappa(llama_bin, gemini_bin)

    # 7. Score distribution comparison
    llama_mean  = sum(llama_list)  / len(llama_list)
    gemini_mean = sum(gemini_list) / len(gemini_list)
    llama_fallback  = sum(1 for s in llama_list  if s == 0.5) / len(llama_list)
    gemini_fallback = sum(1 for s in gemini_list if s == 0.5) / len(gemini_list)

    # 8. Print results
    print(f"\n{'─'*72}")
    print(f"  RESULTADOS -- {exp_id} / {model_label}")
    print(f"{'─'*72}")
    print(f"\n  Distribuição de scores:")
    print(f"  {'':20}  {'Llama 70B':>10}  {'Gemini 2.5':>10}")
    print(f"  {'Média':20}  {llama_mean:>10.4f}  {gemini_mean:>10.4f}")
    print(f"  {'Fallback (=0.5)':20}  {llama_fallback:>10.1%}  {gemini_fallback:>10.1%}")
    print(f"  {'Faithful (>0.5)':20}  {sum(llama_bin)/len(llama_bin):>10.1%}  {sum(gemini_bin)/len(gemini_bin):>10.1%}")

    print(f"\n  Cohen's Kappa: {kappa_result['kappa']:.4f}  "
          f"[{kappa_result['kappa_ci_lo']:.4f}, {kappa_result['kappa_ci_hi']:.4f}]  "
          f"-- {kappa_result['interpretation']}")
    print(f"  Observed agreement: {kappa_result['agreement_rate']:.1%}")
    print(f"  Expected agreement: {kappa_result['p_expected']:.1%}")
    print(f"  Confusion matrix: TP={kappa_result['confusion']['tp']}  "
          f"TN={kappa_result['confusion']['tn']}  "
          f"FP={kappa_result['confusion']['fp']}  "
          f"FN={kappa_result['confusion']['fn']}")

    # 9. Save
    result = {
        "experiment": exp_id,
        "model": model_label,
        "exp_label": label,
        "judge_a": "meta-llama/Llama-3.3-70B-Instruct",
        "judge_b": "google/gemini-2.5-flash (DeepInfra)",
        "n_sample": len(sample_ids),
        "seed": seed,
        "threshold": THRESHOLD,
        "judge_a_mean": round(llama_mean, 4),
        "judge_b_mean": round(gemini_mean, 4),
        "judge_a_fallback_rate": round(llama_fallback, 4),
        "judge_b_fallback_rate": round(gemini_fallback, 4),
        "judge_a_faithful_rate": round(sum(llama_bin) / len(llama_bin), 4),
        "judge_b_faithful_rate": round(sum(gemini_bin) / len(gemini_bin), 4),
        "cohens_kappa": kappa_result,
        "per_query": [
            {"query_id": qid,
             "llama_score": qid2faith[qid],
             "gemini_score": gemini_scores[qid],
             "llama_label": llama_bin[i],
             "gemini_label": gemini_bin[i],
             "agree": llama_bin[i] == gemini_bin[i]}
            for i, qid in enumerate(sample_ids)
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{exp_id}_{safe_model}_kappa.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Salvo -> {out_path}")
    return result


def print_cross_model_summary(all_results: list[dict]):
    print(f"\n{'='*72}")
    print(f"  RESUMO CROSS-MODEL -- Cohen's Kappa (Llama 70B vs Gemini 2.5-flash)")
    print(f"{'='*72}")
    print(f"  {'Exp':<6} {'Modelo':<16} {'n':>4}  {'κ':>7}  {'95% CI':>18}  {'Agree':>7}  {'Interp'}")
    print(f"  {'─'*70}")
    for r in all_results:
        if not r:
            continue
        k = r["cohens_kappa"]
        print(f"  {r['experiment']:<6} {r['model']:<16} {r['n_sample']:>4}  "
              f"{k['kappa']:>7.4f}  [{k['kappa_ci_lo']:.4f}, {k['kappa_ci_hi']:.4f}]  "
              f"{k['agreement_rate']:>7.1%}  {k['interpretation']}")


def main():
    parser = argparse.ArgumentParser(
        description="Cohen's Kappa inter-rater agreement: Llama 70B vs Gemini 2.5-flash"
    )
    parser.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS), required=True)
    parser.add_argument("--model", nargs="+",
                        help="Modelo(s) a avaliar (ex: Ministral-8B Qwen3-8B). "
                             "Ignorado se --all-models for passado.")
    parser.add_argument("--all-models", action="store_true",
                        help="Rodar para todos os 8 modelos SLM")
    parser.add_argument("--n", type=int, default=150,
                        help="Número de queries a amostrar (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = SLM_LABELS if args.all_models else (args.model or ["Ministral-8B"])

    all_results = []
    for exp_id in args.exp:
        for model_label in models:
            r = analyze(exp_id, model_label, args.n, args.seed)
            all_results.append(r)

    if len(all_results) > 1:
        print_cross_model_summary(all_results)

    print("\n✅ Kappa analysis complete.")


if __name__ == "__main__":
    main()
