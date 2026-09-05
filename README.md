# Small Language Models for Legal RAG in Portuguese

Code, experiment outputs, and reproduction instructions for:

> **Small Language Models for Legal RAG in Portuguese: A 4x2 Cross-Domain
> Evaluation of Faithfulness, Retrieval Strategies, and Model Scale**
> Guilherme Dutra, André Caraíba, Nádia Felix, Daniel Ribeiro, Állan Silva,
> Paulo Victor dos Santos, Pedro Albernaz, Sávio Teles
> **ENIAC 2026** (accepted, to appear)

A 4×2 factorial evaluation of Retrieval-Augmented Generation over Brazilian
Portuguese legal text: four sub-domains of the JUÁ benchmark × two retrievers
(BM25 Okapi, BGE-M3), across eight Small Language Models (1B–31B), with
faithfulness scored by a Llama-3.3-70B judge against a Qwen2.5-72B Oracle and
validated with Cohen's κ against a second (Gemini-2.5-flash) judge.

**Headline findings.** BM25 dominates legislative, judicial, and regulatory text
while dense retrieval wins only tax law; 7–8B SLMs match models up to 31B with no
detectable faithfulness difference; whole-document dense retrieval collapses on
ultra-long norms (a truncation boundary); and Oracle-referenced BERTScore can
invert faithfulness rankings (the *BERTScore Paradox*).

---

## Repository layout

```
experiments/            scripts that produce every table in the paper
  run_beir_pipeline.py      retrieval comparison            -> Table 2
  run_generation_2x2.py     the 4x2 factorial (E1a-E4b)     -> Tables 3, 4, 5
  run_kappa_validation.py   inter-judge agreement            -> Table 6
  analyze_significance.py   Wilcoxon signed-rank tests
  analyze_judge_kappa.py    kappa aggregation / reporting
  analyze_token_usage.py    cost and latency accounting
src/mestrado/           core package (data loaders, retrieval, generation, metrics)
results/
  retrieval/            four-system retrieval comparison    (Table 2)
  generation/           aggregated per-experiment results  (Tables 3-5)
  kappa/                inter-judge agreement + per-query   (Table 6)
  significance/         Wilcoxon test outputs
  checkpoints/          raw model responses, 8 experiments x 10 models (49 MB)
```

Every number in the paper can be recomputed from `results/` **without re-running
any paid API call** — the raw generations are included.

## Experimental design

Four JUÁ sub-domains × two retrievers = eight experiments:

| Exp. | Corpus | Retriever | Queries | nDCG@10 |
|---|---|---|---|---|
| E1a / E1b | Ulysses-RFCorpus (legislative) | BM25 / BGE-M3 | 668 | 0.545 / 0.311 |
| E2a / E2b | JurisTCU (judicial) | BM25 / BGE-M3 | 150 | 0.375 / 0.173 |
| E3a / E3b | BR-TaxQA-R (tax) | BM25 / BGE-M3 | 715 | 0.595 / 0.710 |
| E4a / E4b | NormasTCU (regulatory) | BM25 / BGE-M3 | 46 | 0.354 / 0.000 |

**Models.** Llama-3.2-1B/3B, Gemma-3-4B, Qwen2.5-7B, Qwen3-8B, Ministral-8B,
Phi-4 (14B), Gemma-4-31B. **Oracle:** Qwen2.5-72B-Instruct (reference responses
for BERTScore/ROUGE). **Judge:** Llama-3.3-70B-Instruct (faithfulness);
**second judge:** Gemini-2.5-flash (Cohen's κ). All served via the DeepInfra API
at temperature 0.1, 1024 output tokens.

---

## Results

### Table 2 — Retrieval nDCG@10
Source: `results/retrieval/retrieval_comparison_20260424_2355.json` (Ulysses,
JurisTCU) and `retrieval_comparison_20260425_2054.json` (NormasTCU) — the full
four-system comparison including MiniLM and Hybrid, with nDCG@5/10, MRR,
Recall@10 and MAP per system. The BR-TaxQA column comes from the E3a/E3b runs
(`results/generation/E3*_results.json`, `ndcg` field; no separate comparison file
was kept for that corpus). † significantly different from BM25 (p<0.001) ·
⋆ nDCG=0.000 from architectural truncation

| System | Ulysses | JurisTCU | BR-TaxQA | NormasTCU |
|---|---|---|---|---|
| **BM25 Okapi** | **0.545** | **0.375** | 0.595 | **0.354** |
| Dense MiniLM | 0.190† | 0.068† | — | 0.000†⋆ |
| Dense BGE-M3 | 0.311† | 0.173† | **0.710** | 0.000†⋆ |
| Hybrid BM25+BGE | 0.503† | 0.253† | — | 0.216† |

### Table 3 — Generation on Ulysses (n=668)
`results/generation/E1a_results.json`, `E1b_results.json` · Faith = faithfulness,
BF1 = BERTScore F1, ΔF = BGE-M3 − BM25

| Model | Params | Faith (BM25) | BF1 (BM25) | Faith (BGE-M3) | BF1 (BGE-M3) | ΔF |
|---|---|---|---|---|---|---|
| Llama-1B | 1B | 0.590 | 0.764 | 0.547 | 0.767 | −0.044† |
| Llama-3B | 3B | 0.585 | 0.764 | 0.538 | 0.767 | −0.047 |
| Gemma-3-4B | 4B | 0.708 | 0.769 | 0.668 | 0.776 | −0.040† |
| Qwen2.5-7B | 7B | 0.802 | 0.706 | 0.763 | 0.714 | −0.040 |
| Qwen3-8B | 8B | 0.791 | 0.800 | 0.766 | 0.810 | −0.025 |
| Ministral-8B | 8B | 0.806 | 0.768 | 0.764 | 0.772 | −0.041† |
| **Phi-4** | 14B | **0.817** | **0.812** | **0.790** | **0.821** | −0.028† |
| Gemma-4-31B | 31B | 0.748 | 0.784 | 0.734 | 0.787 | −0.015 |

### Table 4 — Faithfulness on JurisTCU (n=150) and BR-TaxQA (n=715)
`results/generation/E2*_results.json`, `E3*_results.json`

| Model | Params | JurisTCU BM25 | JurisTCU BGE-M3 | BR-TaxQA BM25 | BR-TaxQA BGE-M3 |
|---|---|---|---|---|---|
| Llama-1B | 1B | 0.542 | 0.558 | 0.626 | 0.690 |
| Llama-3B | 3B | 0.521 | 0.576 | 0.641 | 0.692 |
| Gemma-3-4B | 4B | 0.673 | 0.695 | 0.725 | 0.764 |
| Qwen2.5-7B | 7B | 0.657 | 0.635 | 0.670 | 0.705 |
| Qwen3-8B | 8B | 0.601 | 0.673† | 0.740 | 0.772 |
| Ministral-8B | 8B | 0.705 | 0.726 | 0.740 | 0.762 |
| Phi-4 | 14B | 0.700 | 0.724 | 0.742 | 0.780 |
| **Gemma-4-31B** | 31B | **0.773** | **0.755** | **0.786** | **0.815** |

### Table 5 — Generation on NormasTCU (n=46) — cascading failure
`results/generation/E4a_results.json`, `E4b_results.json` · E4b retrieval
nDCG=0.000 (8192-token truncation)

| Model | Params | E4a (BM25) | E4b (BGE-M3) | ΔF |
|---|---|---|---|---|
| Llama-1B | 1B | 0.422 | 0.165 | ↓0.257 |
| Llama-3B | 3B | 0.448 | 0.144 | ↓0.304 |
| Gemma-3-4B | 4B | 0.615 | 0.352 | ↓0.263 |
| Qwen2.5-7B | 7B | 0.561 | 0.420 | ↓0.141 |
| Qwen3-8B | 8B | 0.528 | **0.570** | ↑0.042 |
| Ministral-8B | 8B | 0.554 | 0.209 | ↓0.345 |
| Phi-4 | 14B | 0.489 | 0.413 | ↓0.076 |
| **Gemma-4-31B** | 31B | **0.652** | 0.483 | ↓0.169 |

### Table 6 — Cohen's κ (Llama-3.3-70B vs. Gemini-2.5-flash)
`results/kappa/*_kappa.json` (`cohens_kappa.kappa`) · Landis & Koch: ≤0.20
slight (s), 0.21–0.40 fair (f), 0.41–0.60 moderate (m), 0.61–0.80 substantial (S)

| Model | E1a Ulysses | E2a JurisTCU | E3a BR-TaxQA | E4a NormasTCU |
|---|---|---|---|---|
| Gemma-3-4B | 0.19 (s) | **0.46 (m)** | 0.23 (f) | 0.22 (f) |
| Qwen3-8B | 0.04 (s) | 0.33 (f) | 0.34 (f) | 0.49 (m) |
| Ministral-8B | 0.22 (f) | **0.46 (m)** | 0.32 (f) | **0.65 (S)** |

The low κ on Ulysses is a base-rate artifact, not judge disagreement: raw
observed agreement there is 0.72–0.87 while both judges label most responses
faithful (see `judge_a_faithful_rate` / `judge_b_faithful_rate` in the κ files).

---

## Reproduction

### Setup
```bash
uv sync                    # Python 3.12
cp .env.example .env       # set DEEPINFRA_API_KEY (and OPENROUTER_API_KEY if used)
```

### Datasets
The four corpora are public and are **not** redistributed here:

| Corpus | Source |
|---|---|
| Ulysses-RFCorpus | Vitório et al. (2025), *Language Resources and Evaluation* 59:1257–1277 |
| JurisTCU | Fernandes et al. (2025), *Language Resources and Evaluation* |
| BR-TaxQA-R | Domingos Júnior et al. (2025), BRACIS 2025 |
| NormasTCU | https://huggingface.co/datasets/ufca-llms/normas-tcu |

### Recompute the tables from the included results (free, no API)
The aggregated JSONs in `results/generation/`, `results/kappa/`, and
`results/significance/` already contain every value printed in the paper; the
raw generations behind them are in `results/checkpoints/`.

### Re-run the experiments (requires API credits)
```bash
# Table 2 — retrieval comparison
uv run python experiments/run_beir_pipeline.py

# Tables 3-5 — the 4x2 generation factorial (E1a-E4b)
uv run python experiments/run_generation_2x2.py --exp E1a

# Table 6 — inter-judge agreement
uv run python experiments/run_kappa_validation.py
```

Runs are checkpointed: re-invoking a completed experiment reuses
`results/checkpoints/` instead of re-issuing API calls.

---

## Citation

```bibtex
@inproceedings{dutra2026slm,
  title     = {Small Language Models for Legal {RAG} in {P}ortuguese: A 4x2
               Cross-Domain Evaluation of Faithfulness, Retrieval Strategies,
               and Model Scale},
  author    = {Dutra, Guilherme and Cara{\'i}ba, Andr{\'e} and Felix, N{\'a}dia
               and Ribeiro, Daniel and Silva, {\'A}llan and Santos, Paulo Victor
               dos and Albernaz, Pedro and Teles, S{\'a}vio},
  booktitle = {Proceedings of the Encontro Nacional de Intelig{\^e}ncia
               Artificial e Computacional (ENIAC 2026)},
  publisher = {SBC},
  year      = {2026},
  note      = {To appear}
}
```

## License

Code released under the MIT License (see `LICENSE`). The underlying corpora
remain under their original licenses and are not redistributed here.
