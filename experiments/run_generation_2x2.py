#!/usr/bin/env python3
"""
run_generation_2x2.py — 2×2 Factorial Generation (SLM-RAG-PT-BR Paper)

Design (retrieval results April 2026, EXCERTO-only JurisTCU):
  E1a: BM25   + Ulysses   nDCG=0.545  primary Ulysses experiment
  E1b: BGE-M3 + Ulysses   nDCG=0.311  retriever ablation (−43% vs BM25)
  E2a: BM25   + JurisTCU  nDCG=0.375  primary JurisTCU experiment
  E2b: BGE-M3 + JurisTCU  nDCG=0.173  retriever ablation (−54% vs BM25)

Run order for paper (priority):
  (1) E1a  → main Ulysses results  (~6-10h depending on API rate limits)
  (2) E2a  → main JurisTCU results (~2-4h)
  (3) E1b  → retriever ablation Ulysses
  (4) E2b  → retriever ablation JurisTCU

Usage:
  uv run python run_generation_2x2.py --exp E1a
  uv run python run_generation_2x2.py --exp E2a
  uv run python run_generation_2x2.py --exp E1a --smoke        # 3 queries, 1 model
  uv run python run_generation_2x2.py --exp E1a --oracle-only  # build Oracle refs only
  # Checkpoints are always loaded automatically — re-runs skip completed queries
"""
import sys
import json
import re
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.data.beir_loader import BEIRLoader
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.dense import DenseRetriever
from mestrado.generation.rag import DeepInfraRAG, OpenRouterRAG

# ── Directories ────────────────────────────────────────────────────────────────
CKPT_DIR  = Path("results/checkpoints")
GEN_DIR   = Path("results/generation")
CACHE_DIR = Path("results/cache")
BGE_MODEL = "BAAI/bge-m3"

# ── Parallelism ────────────────────────────────────────────────────────────────
# SLM pass fires 2 DeepInfra requests per query (SLM + Judge), so budget halved.
N_WORKERS_ORACLE     = 20   # Oracle runs alone → can be aggressive
N_WORKERS_DEEPINFRA  = 10   # SLM+Judge together; 10×8 models = 80 concurrent
N_WORKERS_OPENROUTER = 15   # OpenRouter

# ── Experiment Registry ────────────────────────────────────────────────────────
EXPERIMENTS = {
    "E1a": dict(dataset="ulysses",  retriever="bm25",  ndcg=0.545,
                label="BM25+Ulysses",    description="Primary Ulysses experiment"),
    "E1b": dict(dataset="ulysses",  retriever="bgem3", ndcg=0.311,
                label="BGE-M3+Ulysses",  description="Retriever ablation Ulysses (−43%)"),
    "E2a": dict(dataset="juristcu", retriever="bm25",  ndcg=0.375,
                label="BM25+JurisTCU",   description="Primary JurisTCU experiment"),
    "E2b": dict(dataset="juristcu", retriever="bgem3", ndcg=0.173,
                label="BGE-M3+JurisTCU", description="Retriever ablation JurisTCU (−54%)"),
    # E3/E4: run retrieval first to fill nDCG values:
    #   uv run python run_experimento_retrieval.py --dataset br-taxqa
    #   uv run python run_experimento_retrieval.py --dataset normas-tcu
    "E3a": dict(dataset="brtaxqa",   retriever="bm25",  ndcg=0.5947,
                label="BM25+BR-TaxQA",    description="Primary BR-TaxQA (715 queries, tax law)"),
    "E3b": dict(dataset="brtaxqa",   retriever="bgem3", ndcg=0.7095,
                label="BGE-M3+BR-TaxQA",  description="Retriever ablation BR-TaxQA (Dense wins: +19.3%)"),
    "E4a": dict(dataset="normastcu", retriever="bm25",  ndcg=0.3537,
                label="BM25+NormasTCU",   description="NormasTCU exploratory (46 queries, TCU norms)"),
    "E4b": dict(dataset="normastcu", retriever="bgem3", ndcg=0.0,
                label="BGE-M3+NormasTCU", description="Dense fails (nDCG=0): long docs exceed 8192-token limit"),
}

# ── Models ─────────────────────────────────────────────────────────────────────
# (paper_label, model_id, provider)
# Verify IDs at https://deepinfra.com/models  and  https://openrouter.ai/models
SLM_MODELS = [
    ("Llama-1B",    "meta-llama/Llama-3.2-1B-Instruct",  "deepinfra"),  # Dubey2024Llama3
    ("Llama-3B",    "meta-llama/Llama-3.2-3B-Instruct",  "deepinfra"),  # Dubey2024Llama3
    ("Gemma3-4B",   "google/gemma-3-4b-it",              "deepinfra"),  # Google2025Gemma3
    ("Qwen25-7B",   "Qwen/Qwen2.5-7B-Instruct",          "deepinfra"),  # Qwen2025; Qwen2.5 confirmed no DeepInfra bug
    ("Phi4-14B",    "microsoft/phi-4",                   "deepinfra"),  # Abdin2025Phi4Mini; mid-range
    ("Gemma4-31B",  "google/gemma-4-31B-it",             "deepinfra"),  # Google2025; upper bound
    ("Qwen3-8B",    "qwen/qwen3-8b",                     "openrouter"),           # Qwen3 thinking mode ON
    ("Qwen3-8B-NT", "qwen/qwen3-8b",                     "openrouter:no-thinking"), # Qwen3 thinking ablation
    ("Ministral-8B","mistralai/ministral-8b-2512",        "openrouter"), # Mistral family
]

# Oracle: Qwen2.5 72B (Alibaba family, DeepInfra)
#   — família Alibaba: separada de Llama (Judge) → elimina circular eval bias
#   — multilingual frontier model; handles PT-BR
# Judge: Llama 3.3 70B (Meta family, DeepInfra)
#   — validated κ=0.786 vs human raters (JudgeVerdict2025)
#   — família Meta: separada de Alibaba (Oracle) → sem circular bias
ORACLE_MODEL = ("Oracle-72B", "Qwen/Qwen2.5-72B-Instruct", "deepinfra")
JUDGE_MODEL  = ("Judge-70B",  "meta-llama/Llama-3.3-70B-Instruct", "deepinfra")

# ── Generation Parameters ──────────────────────────────────────────────────────
TOP_K_CONTEXT  = 5      # documents injected into prompt (paper: max 5 bills)
MAX_TOKENS     = 1024
TEMPERATURE    = 0.1
CHARS_PER_DOC  = 2500   # per-document character budget in context

# ── System Prompts (from paper Appendix A) ─────────────────────────────────────
PROMPT_ULYSSES = (
    "Você é um especialista em legislação da Câmara dos Deputados.\n"
    "Responda à consulta com base EXCLUSIVAMENTE nas informações dos projetos de lei "
    "(PLs) listados como contexto.\n"
    "REGRAS:\n"
    "(1) Utilize exclusivamente as informações dos PLs listados.\n"
    "(2) Utilize os projetos disponíveis mesmo que parcialmente relacionados e indique "
    "na observação final se a cobertura for limitada.\n"
    "(3) Não invente informações além dos PLs.\n"
    "(4) Seja objetivo e cite os números dos PLs relevantes.\n"
    "(5) Formate: (a) PL principal, (b) PLs relacionados, (c) observação."
)

PROMPT_JURISTCU = (
    "Você é um especialista em jurisprudência do Tribunal de Contas da União (TCU).\n"
    "Responda à consulta com base EXCLUSIVAMENTE nos acórdãos TCU fornecidos.\n"
    "REGRAS:\n"
    "(1) Cite explicitamente os números dos acórdãos (ex: Acórdão 358/2020).\n"
    "(2) Utilize os acórdãos disponíveis mesmo que parcialmente relacionados — "
    "indique cobertura limitada na observação.\n"
    "(3) Não invente informações além dos acórdãos.\n"
    "(4) Seja objetivo — cite a jurisprudência TCU.\n"
    "(5) Formate: (a) acórdão principal, (b) relacionados, (c) observação."
)

PROMPT_BRTAXQA = (
    "Você é um especialista em direito tributário brasileiro.\n"
    "Responda à consulta com base EXCLUSIVAMENTE nos documentos tributários fornecidos.\n"
    "REGRAS:\n"
    "(1) Cite explicitamente os dispositivos legais relevantes (lei, artigo, instrução normativa).\n"
    "(2) Utilize os documentos disponíveis mesmo que parcialmente relacionados — "
    "indique cobertura limitada na observação.\n"
    "(3) Não invente informações além dos documentos.\n"
    "(4) Seja objetivo — a consulta é de um profissional da área fiscal.\n"
    "(5) Formate: (a) fundamento principal, (b) documentos relacionados, (c) observação."
)

PROMPT_NORMASTCU = (
    "Você é um especialista em normas regulatórias do Tribunal de Contas da União (TCU).\n"
    "Responda à consulta com base EXCLUSIVAMENTE nas normas TCU fornecidas.\n"
    "REGRAS:\n"
    "(1) Cite explicitamente os dispositivos normativos (número da norma, artigo).\n"
    "(2) Utilize as normas disponíveis mesmo que parcialmente relacionadas — "
    "indique cobertura limitada na observação.\n"
    "(3) Não invente informações além das normas.\n"
    "(4) Seja objetivo — a consulta é de um auditor ou gestor público.\n"
    "(5) Formate: (a) norma principal, (b) normas relacionadas, (c) observação."
)

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


# ── Utilities ──────────────────────────────────────────────────────────────────
def _safe(s: str) -> str:
    return s.replace("/", "__").replace("-", "_").replace(".", "_")


def _parse_faith(content: str) -> float:
    # aceita ponto ou vírgula como separador decimal (Judge pode usar PT-BR)
    m = re.search(r'\b(0[.,]\d+|1[.,]0|0[.,]0|1\.0|0\.0)\b', content)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r'(\d+)%', content)
    if m:
        return float(m.group(1)) / 100.0
    cl = content.lower()
    if "completamente fiel" in cl or "totalmente fiel" in cl:
        return 1.0
    if "parcialmente" in cl:
        return 0.5
    if "infiel" in cl or "não fiel" in cl:
        return 0.0
    return 0.5


def get_rag(label: str, model_id: str, provider: str):
    no_thinking = provider == "openrouter:no-thinking"
    if provider.startswith("openrouter"):
        return OpenRouterRAG(model=model_id, no_thinking=no_thinking)
    return DeepInfraRAG(model=model_id)


def build_context(bills: list, dataset: str) -> str:
    """Build the context string injected into the prompt."""
    parts = []
    for bill in bills[:TOP_K_CONTEXT]:
        name = getattr(bill, "name", "")
        if dataset == "ulysses":
            # Ulysses: txt_ementa is the legislative summary (indexed field)
            text = getattr(bill, "txt_ementa", "") or getattr(bill, "text", "")
        else:
            # juristcu/brtaxqa/normastcu: bill.text holds full document content
            text = getattr(bill, "text", "") or getattr(bill, "txt_ementa", "")
        text = text[:CHARS_PER_DOC]
        parts.append(f"### {name}\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_response(rag, system_prompt: str, query: str, context: str):
    """Call the API and return (text, input_tokens, output_tokens, latency_s)."""
    user_msg = (
        f"Consulta: {query}\n\n"
        f"Documentos recuperados:\n{context}\n\n"
        f"Responda com base nos documentos acima."
    )
    t0 = time.time()
    resp = rag._client.chat.completions.create(
        model=rag.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    latency = time.time() - t0
    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    tin  = getattr(usage, "prompt_tokens",     0) if usage else 0
    tout = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, tin, tout, latency


def judge_faithfulness(judge_rag, context: str, response: str) -> float:
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:5000], response=response[:1500]
    )
    resp = judge_rag._client.chat.completions.create(
        model=judge_rag.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    return _parse_faith(resp.choices[0].message.content.strip())


def compute_bertscore(predictions: list, references: list) -> float:
    """BERTScore F1, bert-base-multilingual-cased, layer 9 (paper §4.4).
    Usa evaluate>=0.4 (compatível com transformers 5.x)."""
    if not predictions:
        return 0.0
    try:
        import evaluate as ev
        bertscore = ev.load("bertscore")
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type="bert-base-multilingual-cased",
            num_layers=9,
            lang="pt",
            batch_size=4,
            device="cpu",
        )
        return float(sum(result["f1"]) / len(result["f1"]))
    except Exception as e:
        print(f"  ⚠️  BERTScore error: {e}")
        return 0.0


def compute_rouge_l(predictions: list, references: list) -> float:
    if not predictions:
        return 0.0
    try:
        from rouge_score import rouge_scorer as rs_mod
        scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=False)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores)
    except Exception as e:
        print(f"  ⚠️  ROUGE-L error: {e}")
        return 0.0


# ── Parallel task helpers ──────────────────────────────────────────────────────
def _oracle_task(q, oracle_rag, sys_prompt, context):
    for attempt in range(3):
        try:
            text, tin, tout, lat = generate_response(oracle_rag, sys_prompt, q.text, context)
            return (q.query_id, text, tin, tout, lat)
        except Exception as e:
            wait = [5, 15, 30][attempt]
            print(f"  ⚠️  Oracle error (attempt {attempt+1}): {e}. Waiting {wait}s...")
            time.sleep(wait)
    return None


def _slm_task(q, slm_rag, judge_rag, sys_prompt, context):
    for attempt in range(3):
        try:
            text, tin, tout, lat = generate_response(slm_rag, sys_prompt, q.text, context)
            faith = judge_faithfulness(judge_rag, context, text)
            return (q.query_id, text, tin, tout, lat, faith)
        except Exception as e:
            wait = [10, 30, 60][attempt]
            print(f"  ⚠️  SLM error (attempt {attempt+1}): {e}. Waiting {wait}s...")
            time.sleep(wait)
    return None


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def ckpt_path(exp_id: str, label: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return CKPT_DIR / f"{exp_id}_{_safe(label)}.json"


def load_ckpt(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"query_ids": [], "responses": [], "tok_in": [],
            "tok_out": [], "latencies": [], "faithfulness": []}


def save_ckpt(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Data loading ───────────────────────────────────────────────────────────────
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
    print(f"  ✅ {len(bills):,} docs | {len(queries)} queries")
    return bills, queries


def build_retriever(retriever_type: str, bills: dict, dataset_name: str):
    if retriever_type == "bm25":
        ret = BM25Retriever(bills)
        ret.build_index()
        return ret
    else:  # bgem3 — cache re-used from run_experimento_retrieval.py
        cache = CACHE_DIR / dataset_name
        cache.mkdir(parents=True, exist_ok=True)
        ret = DenseRetriever(bills, model_name=BGE_MODEL)
        ret.build_index(cache_path=cache / "bge-m3.pkl")
        return ret


# ── Main experiment ────────────────────────────────────────────────────────────
def run_experiment(exp_id: str, smoke: bool, oracle_only: bool):
    cfg        = EXPERIMENTS[exp_id]
    dataset    = cfg["dataset"]
    ret_type   = cfg["retriever"]
    _PROMPT_MAP = {
        "ulysses":   PROMPT_ULYSSES,
        "juristcu":  PROMPT_JURISTCU,
        "brtaxqa":   PROMPT_BRTAXQA,
        "normastcu": PROMPT_NORMASTCU,
    }
    sys_prompt = _PROMPT_MAP.get(dataset, PROMPT_JURISTCU)

    print(f"\n{'='*72}")
    print(f"  {exp_id}: {cfg['label']}  (nDCG@10={cfg['ndcg']:.3f})")
    print(f"  {cfg['description']}")
    print(f"{'='*72}")

    # 1. Load data
    print("\n📚 Loading dataset...")
    bills, queries = load_dataset(dataset)

    if smoke:
        queries = queries[:3]
        print(f"  🔥 Smoke mode: {len(queries)} queries only")

    # 2 + 3. Retrieve contexts (cached — skips retriever build on reruns)
    # Shared cache dir with analyze_judge_kappa.py
    ctx_cache_path = Path("results/kappa/ctx_cache") / f"{exp_id}_contexts.json"
    if ctx_cache_path.exists():
        print(f"\n📄 Carregando contextos do cache ({ctx_cache_path.name})...")
        with open(ctx_cache_path, encoding="utf-8") as f:
            contexts: dict[str, str] = json.load(f)
        print(f"  ✅ {len(contexts)} contextos carregados — retriever não necessário")
    else:
        print(f"\n🔍 Building {ret_type.upper()} retriever...")
        retriever = build_retriever(ret_type, bills, dataset)
        print(f"\n📄 Retrieving top-{TOP_K_CONTEXT} docs for {len(queries)} queries...")
        contexts: dict[str, str] = {}
        for q in tqdm(queries, desc="retrieval"):
            retrieved = retriever.retrieve(q.text, top_k=TOP_K_CONTEXT)
            top_bills = [r.bill for r in retrieved]
            contexts[q.query_id] = build_context(top_bills, dataset)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(ctx_cache_path, "w", encoding="utf-8") as f:
            json.dump(contexts, f, ensure_ascii=False)
        print(f"  💾 Contextos salvos → {ctx_cache_path}")

    # 4. Oracle pass (generates BERTScore references)
    print(f"\n🔮 ORACLE: {ORACLE_MODEL[1]}")
    oracle_ckpt = ckpt_path(exp_id, "Oracle")
    oracle_data = load_ckpt(oracle_ckpt)

    done_ids   = set(oracle_data["query_ids"])
    remaining  = [q for q in queries if q.query_id not in done_ids]
    oracle_rag = get_rag(*ORACLE_MODEL)

    _ckpt_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=N_WORKERS_ORACLE) as pool:
        futures = {
            pool.submit(_oracle_task, q, oracle_rag, sys_prompt, contexts[q.query_id]): q
            for q in remaining
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Oracle"):
            result = future.result()
            if result:
                qid, text, tin, tout, lat = result
                with _ckpt_lock:
                    oracle_data["query_ids"].append(qid)
                    oracle_data["responses"].append(text)
                    oracle_data["tok_in"].append(tin)
                    oracle_data["tok_out"].append(tout)
                    oracle_data["latencies"].append(lat)
                    oracle_data["faithfulness"].append(1.0)
                    save_ckpt(oracle_ckpt, oracle_data)

    id2oracle = dict(zip(oracle_data["query_ids"], oracle_data["responses"]))
    print(f"  ✅ Oracle responses: {len(id2oracle)}/{len(queries)}")

    if oracle_only:
        print("\n  [--oracle-only] Done. Run without flag to generate SLM responses.")
        return None

    # 5. SLM passes — Phase 1: all models generate in parallel
    judge_rag  = get_rag(*JUDGE_MODEL)
    model_list = SLM_MODELS[:1] if smoke else SLM_MODELS
    all_results: dict[str, dict] = {}

    def _gen_model(args: tuple) -> tuple[str, dict]:
        """Generate responses for one model. Runs concurrently with other models."""
        label, model_id, provider = args
        print(f"\n{'─'*60}")
        print(f"  ▶  {label}  ({provider}: {model_id})")
        print(f"{'─'*60}")

        slm_ckpt  = ckpt_path(exp_id, label)
        slm_data  = load_ckpt(slm_ckpt)
        done_slm  = set(slm_data["query_ids"])
        slm_rag   = get_rag(label, model_id, provider)
        pending   = [q for q in queries if q.query_id not in done_slm]

        n_workers = N_WORKERS_OPENROUTER if provider == "openrouter" else N_WORKERS_DEEPINFRA
        _slm_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_slm_task, q, slm_rag, judge_rag, sys_prompt, contexts[q.query_id]): q
                for q in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=label):
                result = future.result()
                if result:
                    qid, text, tin, tout, lat, faith = result
                    with _slm_lock:
                        slm_data["query_ids"].append(qid)
                        slm_data["responses"].append(text)
                        slm_data["tok_in"].append(tin)
                        slm_data["tok_out"].append(tout)
                        slm_data["latencies"].append(lat)
                        slm_data["faithfulness"].append(faith)
                        save_ckpt(slm_ckpt, slm_data)

        print(f"  ✅ {label}: {len(slm_data['query_ids'])} respostas geradas")
        return label, slm_data

    print(f"\n⚡  Gerando com {len(model_list)} modelos em paralelo...")
    with ThreadPoolExecutor(max_workers=len(model_list)) as outer_pool:
        model_outputs: list[tuple[str, dict]] = list(
            outer_pool.map(_gen_model, [(l, m, p) for l, m, p in model_list])
        )

    # 5b. BERTScore + ROUGE-L — CPU-bound, sequential to avoid OOM
    label_to_meta = {l: (m, p) for l, m, p in model_list}
    for label, slm_data in model_outputs:
        model_id, provider = label_to_meta[label]
        ordered = [q.query_id for q in queries
                   if q.query_id in set(slm_data["query_ids"])]
        id2slm   = dict(zip(slm_data["query_ids"], slm_data["responses"]))
        id2faith = dict(zip(slm_data["query_ids"], slm_data["faithfulness"]))

        preds = [id2slm.get(qid, "")    for qid in ordered]
        refs  = [id2oracle.get(qid, "") for qid in ordered]

        print(f"  📊 BERTScore+ROUGE-L for {label} ({len(preds)} pairs)...")
        bert_f1 = compute_bertscore(preds, refs)
        rouge_l = compute_rouge_l(preds, refs)
        mean_f   = sum(id2faith[qid] for qid in ordered) / max(len(ordered), 1)
        mean_lat = sum(slm_data["latencies"]) / max(len(slm_data["latencies"]), 1)
        mean_tin = sum(slm_data["tok_in"])    / max(len(slm_data["tok_in"]),    1)
        mean_tout = sum(slm_data["tok_out"])  / max(len(slm_data["tok_out"]),   1)

        print(f"  ✅ BF1={bert_f1:.4f}  ROUGE-L={rouge_l:.4f}  Faith={mean_f:.4f}"
              f"  lat={mean_lat:.2f}s  tok_out={mean_tout:.0f}")

        all_results[label] = {
            "label":       label,
            "model":       model_id,
            "provider":    provider,
            "n_queries":   len(ordered),
            "bertscore_f1": round(bert_f1, 4),
            "rouge_l":     round(rouge_l, 4),
            "faithfulness": round(mean_f, 4),
            "latency_s":   round(mean_lat, 3),
            "tok_in_mean": round(mean_tin, 1),
            "tok_out_mean": round(mean_tout, 1),
        }

    # 6. Save consolidated results
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEN_DIR / f"{exp_id}_results.json"
    final = {
        "experiment":   exp_id,
        "label":        cfg["label"],
        "dataset":      dataset,
        "retriever":    ret_type,
        "ndcg":         cfg["ndcg"],
        "n_queries":    len(queries),
        "oracle_model": ORACLE_MODEL[1],
        "judge_model":  JUDGE_MODEL[1],
        "models":       all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # Print summary table
    print(f"\n{'='*72}")
    print(f"  SUMMARY — {exp_id}: {cfg['label']}")
    print(f"{'='*72}")
    print(f"  {'Model':<14} {'BERTScore':>10} {'ROUGE-L':>8} {'Faith':>7} {'lat(s)':>7}")
    print(f"  {'─'*54}")
    for lbl, r in all_results.items():
        print(f"  {lbl:<14} {r['bertscore_f1']:>10.4f} {r['rouge_l']:>8.4f}"
              f" {r['faithfulness']:>7.4f} {r['latency_s']:>7.2f}")
    print(f"\n  💾 Saved → {out_path}")
    return final


def main():
    parser = argparse.ArgumentParser(
        description="2×2 Factorial Generation Experiment for SLM-RAG-PT-BR Paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", nargs="+", choices=list(EXPERIMENTS), required=True,
        help="Experiment(s) to run: E1a E1b E2a E2b"
    )
    parser.add_argument("--smoke",       action="store_true",
                        help="Smoke test: 3 queries, 1 model only")
    parser.add_argument("--oracle-only", action="store_true",
                        help="Generate Oracle reference responses only (no SLMs)")
    args = parser.parse_args()

    results = []
    for exp_id in args.exp:
        r = run_experiment(
            exp_id,
            smoke=args.smoke,
            oracle_only=args.oracle_only,
        )
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n{'='*72}")
        print("  CROSS-EXPERIMENT SUMMARY")
        print(f"{'='*72}")
        for r in results:
            print(f"\n  {r['experiment']}: {r['label']}  (nDCG={r['ndcg']})")
            for lbl, m in r["models"].items():
                print(f"    {lbl:<14} BF1={m['bertscore_f1']:.4f}  "
                      f"Faith={m['faithfulness']:.4f}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
