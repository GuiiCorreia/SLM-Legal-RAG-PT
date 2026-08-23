"""
LLM-as-Reranker using Ollama — Experiment 4.

DIRECT ADAPTATION of findRelevantMemories.ts from the Claude Code repository:
- formatMemoryManifest()     → format_bill_manifest()
- sideQuery() with JSON schema → ollama.generate(format="json")
- validFilenames.has()       → valid_names set (anti-hallucination filter)
- SELECT_MEMORIES_SYSTEM_PROMPT → LEGISLATIVE_RERANKER_SYSTEM_PROMPT

Also supports Anthropic API as drop-in replacement for comparison experiments.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from mestrado.data.schema import Bill
from mestrado.retrieval.bm25 import RetrievalResult
from mestrado.utils.token_estimation import estimate_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt — adapted from SELECT_MEMORIES_SYSTEM_PROMPT in findRelevantMemories.ts
# ---------------------------------------------------------------------------

LEGISLATIVE_RERANKER_SYSTEM_PROMPT = """\
Você é um especialista em legislação brasileira e em recuperação de informação legislativa.
Sua tarefa é selecionar os projetos de lei mais relevantes para uma consulta de um assessor legislativo.

REGRAS:
1. Retorne APENAS os nomes exatos dos projetos como aparecem no manifesto (ex: "PL 3650/2021")
2. Ordene do mais relevante ao menos relevante
3. Inclua no máximo {top_n} projetos
4. Não invente nomes — use EXATAMENTE os nomes do manifesto
5. Considere relevância semântica, não apenas correspondência literal de palavras
6. Inclua projetos parcialmente relevantes se não houver mais relevantes

Retorne um objeto JSON com a chave "selected_bills" contendo a lista ordenada.
"""

# ---------------------------------------------------------------------------
# Query Expansion prompt — adapted from compact/prompt.ts structured extraction
# ---------------------------------------------------------------------------

QUERY_EXPANSION_PROMPT = """\
Você é um especialista em linguagem legislativa brasileira.

Reformule a seguinte consulta informal de um assessor legislativo para usar:
1. Terminologia técnica legislativa precisa
2. Termos alternativos e sinônimos legislativos
3. Áreas temáticas relacionadas

Consulta original: {query}

Retorne JSON com:
{{
  "query_expanded": "reformulação técnica em 1-2 frases",
  "keywords": ["termo1", "termo2", ...],
  "area_tematica": "área principal"
}}
"""


@dataclass
class RerankerResult:
    bill: Bill
    rank: int
    reasoning: str = ""
    tokens_used: int = 0


@dataclass
class RerankerStats:
    queries_processed: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_s: float = 0.0
    hallucinations_filtered: int = 0

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / max(1, self.queries_processed)

    @property
    def estimated_cost_usd(self) -> float:
        # Ollama local = $0; Anthropic Haiku ≈ $0.25/1M input + $1.25/1M output
        return 0.0  # Override if using Anthropic


class OllamaReranker:
    """
    Zero-shot LLM reranker using local Ollama model.

    Two-stage pipeline:
      Stage 1: BM25/Dense/Hybrid retrieves top_k_retrieval candidates
      Stage 2: LLM reranks and selects top_n final results

    This is the direct Python port of the findRelevantMemories.ts pattern.

    Example:
        reranker = OllamaReranker()
        results = reranker.rerank(query="cavalos de raça", candidates=bm25_results, top_n=12)
    """

    def __init__(
        self,
        model: str | None = None,
        top_n: int | None = None,
    ) -> None:
        from mestrado.config import ollama as cfg, experiment as exp_cfg
        import ollama as ollama_lib
        self._ollama = ollama_lib
        self.model = model or cfg.reranker_model
        self.top_n = top_n or exp_cfg.rerank_top_n
        self.stats = RerankerStats()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        """
        Rerank candidate bills for a query using Ollama.

        Pattern from findRelevantMemories.ts:
        1. Format manifest (formatMemoryManifest equivalent)
        2. Query LLM with JSON schema (sideQuery equivalent)
        3. Filter hallucinations (validFilenames.has() equivalent)
        4. Return ordered results
        """
        top_n = top_n or self.top_n
        if not candidates:
            return []

        bills = [r.bill for r in candidates]
        candidate_map: dict[str, Bill] = {b.name: b for b in bills}
        valid_names: set[str] = set(candidate_map.keys())  # Anti-hallucination set

        # Step 1: Format manifest (adapted from formatMemoryManifest)
        manifest = self._format_bill_manifest(bills)

        # Step 2: Build prompt
        system = LEGISLATIVE_RERANKER_SYSTEM_PROMPT.format(top_n=top_n)
        user_prompt = (
            f'Consulta do assessor: "{query}"\n\n'
            f"Projetos de lei disponíveis:\n\n{manifest}\n\n"
            f"Selecione os {top_n} mais relevantes em ordem decrescente de relevância."
        )

        # Step 3: Call LLM (sideQuery equivalent — no conversation history)
        t0 = time.time()
        raw_response = self._call_ollama(system, user_prompt)
        latency = time.time() - t0

        # Step 4: Parse JSON
        selected_names, tokens_in, tokens_out = self._parse_response(raw_response, valid_names)

        # Step 5: Anti-hallucination filter (validFilenames.has() equivalent)
        filtered = [name for name in selected_names if name in valid_names]
        hallucinated = len(selected_names) - len(filtered)
        if hallucinated:
            logger.debug(f"Filtered {hallucinated} hallucinated bill names for query: {query[:50]}")

        # Step 6: Build results in LLM-determined order, append unselected by BM25 score
        result_bills = [candidate_map[n] for n in filtered if n in candidate_map]
        selected_set = set(filtered)
        for cand in candidates:
            if cand.bill.name not in selected_set and len(result_bills) < top_n:
                result_bills.append(cand.bill)

        # Update stats
        self.stats.queries_processed += 1
        self.stats.total_tokens_in += tokens_in
        self.stats.total_tokens_out += tokens_out
        self.stats.total_latency_s += latency
        self.stats.hallucinations_filtered += hallucinated

        return [
            RerankerResult(bill=b, rank=i + 1, tokens_used=tokens_in + tokens_out)
            for i, b in enumerate(result_bills[:top_n])
        ]

    def expand_query(self, query: str) -> dict[str, Any]:
        """
        Query expansion for Experiment 5.
        Translates informal PT-BR queries to formal legislative language.
        Adapted from QUERY_EXPANSION_PROMPT in ideias/10_prompt_engineering.md.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        raw = self._call_ollama("", prompt)
        try:
            data = json.loads(raw.get("response", "{}"))
            return {
                "query_expanded": data.get("query_expanded", query),
                "keywords": data.get("keywords", []),
                "area_tematica": data.get("area_tematica", ""),
                "original": query,
            }
        except (json.JSONDecodeError, AttributeError):
            return {"query_expanded": query, "keywords": [], "area_tematica": "", "original": query}

    # ------------------------------------------------------------------
    # Internal helpers — adapted from findRelevantMemories.ts
    # ------------------------------------------------------------------

    def _format_bill_manifest(self, bills: list[Bill], max_ementa_chars: int = 250) -> str:
        """
        Adapted from formatMemoryManifest() in findRelevantMemories.ts.
        Creates a markdown manifest of bill names and summaries for LLM input.
        """
        sections = []
        for bill in bills:
            ementa = bill.txt_ementa[:max_ementa_chars]
            if len(bill.txt_ementa) > max_ementa_chars:
                ementa += "..."
            section = f"## {bill.name} ({bill.sig_tipo})\n{ementa}"
            sections.append(section)
        return "\n\n---\n\n".join(sections)

    def _call_ollama(self, system: str, prompt: str) -> dict[str, Any]:
        """
        Calls Ollama with JSON format output.
        Equivalent to sideQuery() in findRelevantMemories.ts:
        - No conversation history (fresh context)
        - JSON schema output
        - Low temperature (deterministic)
        """
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            response = self._ollama.generate(
                model=self.model,
                prompt=full_prompt,
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 512,
                },
            )
            return response
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}. Returning empty response.")
            return {"response": '{"selected_bills": []}', "eval_count": 0, "prompt_eval_count": 0}

    def _parse_response(
        self,
        raw: dict[str, Any],
        valid_names: set[str],
    ) -> tuple[list[str], int, int]:
        """
        Parse LLM JSON response and extract selected bill names.
        Robust to malformed JSON via regex fallback.
        Returns (selected_names, tokens_in, tokens_out).
        """
        response_text = raw.get("response", "")
        tokens_in = raw.get("prompt_eval_count", estimate_tokens(response_text))
        tokens_out = raw.get("eval_count", estimate_tokens(response_text))

        # Try structured JSON parse first
        try:
            data = json.loads(response_text)
            names = data.get("selected_bills", [])
            if isinstance(names, list):
                return [str(n) for n in names], tokens_in, tokens_out
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: extract anything that looks like a bill name (PL XXXX/YYYY)
        pattern = r'\b(?:PL|PEC|PLP|PDC|INC)\s+[\d]+/\d{4}\b'
        found = re.findall(pattern, response_text)
        logger.debug(f"JSON parse failed; regex fallback found {len(found)} names.")
        return found, tokens_in, tokens_out


class DeepInfraReranker:
    """
    SLM/LLM reranker via DeepInfra OpenAI-compatible API.

    Enables testing SLMs (Phi-4 Mini, Qwen2.5-7B) as zero-shot rerankers,
    extending Sun et al. (2023 EMNLP) from LLMs to SLMs.

    Same interface as OllamaReranker — drop-in for E4 experiments.
    Requires DEEPINFRA_API_KEY in environment.

    Reference: Sun et al. (2023). Is ChatGPT Good at Search? Investigating
      Large Language Models as Re-Ranking Agents. EMNLP 2023.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        top_n: int | None = None,
    ) -> None:
        import os
        import openai
        from mestrado.config import experiment as exp_cfg
        key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        if not key:
            raise ValueError("DEEPINFRA_API_KEY not set.")
        self._client = openai.OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=key,
        )
        self.model = model or os.environ.get(
            "SLM_RERANKER_MODEL", "microsoft/phi-4-mini-instruct"
        )
        self.top_n = top_n or exp_cfg.rerank_top_n
        self.stats = RerankerStats()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        """Zero-shot reranking via DeepInfra SLM. Same pattern as OllamaReranker."""
        top_n = top_n or self.top_n
        if not candidates:
            return []

        bills = [r.bill for r in candidates]
        candidate_map: dict[str, Bill] = {b.name: b for b in bills}
        valid_names: set[str] = set(candidate_map.keys())

        manifest = OllamaReranker._format_bill_manifest(None, bills)  # type: ignore[arg-type]
        system = LEGISLATIVE_RERANKER_SYSTEM_PROMPT.format(top_n=top_n)
        user_content = (
            f'Consulta do assessor: "{query}"\n\n'
            f"Projetos de lei disponíveis:\n\n{manifest}\n\n"
            f"Selecione os {top_n} mais relevantes em ordem decrescente de relevância."
        )

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            latency = time.time() - t0
            response_text = response.choices[0].message.content or "{}"
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            logger.warning(f"DeepInfra rerank call failed ({self.model}): {e}")
            return [
                RerankerResult(bill=r.bill, rank=i + 1)
                for i, r in enumerate(candidates[:top_n])
            ]

        try:
            data = json.loads(response_text)
            selected_names = data.get("selected_bills", [])
            if not isinstance(selected_names, list):
                selected_names = []
        except json.JSONDecodeError:
            pattern = r'\b(?:PL|PEC|PLP|PDC|INC)\s+[\d]+/\d{4}\b'
            selected_names = re.findall(pattern, response_text)

        filtered = [n for n in selected_names if n in valid_names]
        hallucinated = len(selected_names) - len(filtered)
        if hallucinated:
            logger.debug(f"DeepInfra: filtered {hallucinated} hallucinated names.")

        result_bills = [candidate_map[n] for n in filtered if n in candidate_map]
        selected_set = set(filtered)
        for cand in candidates:
            if cand.bill.name not in selected_set and len(result_bills) < top_n:
                result_bills.append(cand.bill)

        cost = (tokens_in * 0.06 + tokens_out * 0.06) / 1_000_000  # approx
        self.stats.queries_processed += 1
        self.stats.total_tokens_in += tokens_in
        self.stats.total_tokens_out += tokens_out
        self.stats.total_latency_s += latency
        self.stats.hallucinations_filtered += hallucinated

        logger.debug(
            f"DeepInfra rerank ({self.model}): {tokens_in}+{tokens_out} tok, "
            f"${cost:.5f}, {latency:.1f}s"
        )
        return [
            RerankerResult(bill=b, rank=i + 1, tokens_used=tokens_in + tokens_out)
            for i, b in enumerate(result_bills[:top_n])
        ]


class AnthropicReranker:
    """
    Optional drop-in replacement for OllamaReranker using Anthropic API.
    Same interface — swap in for LLM comparison experiments.
    Requires ANTHROPIC_API_KEY in .env.
    """

    def __init__(self, model: str | None = None, top_n: int | None = None) -> None:
        from mestrado.config import anthropic as cfg, experiment as exp_cfg
        import anthropic as ant
        self._client = ant.Anthropic(api_key=cfg.api_key)
        self.model = model or cfg.reranker_model
        self.top_n = top_n or exp_cfg.rerank_top_n
        self.stats = RerankerStats()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        top_n = top_n or self.top_n
        if not candidates:
            return []

        bills = [r.bill for r in candidates]
        candidate_map = {b.name: b for b in bills}
        valid_names = set(candidate_map.keys())

        manifest = OllamaReranker._format_bill_manifest(None, bills)  # type: ignore[arg-type]
        system = LEGISLATIVE_RERANKER_SYSTEM_PROMPT.format(top_n=top_n)
        user_content = (
            f'Consulta do assessor: "{query}"\n\n'
            f"Projetos de lei disponíveis:\n\n{manifest}\n\n"
            f"Selecione os {top_n} mais relevantes em ordem decrescente de relevância."
        )

        t0 = time.time()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        latency = time.time() - t0

        response_text = message.content[0].text if message.content else "{}"
        try:
            data = json.loads(response_text)
            selected_names = data.get("selected_bills", [])
        except json.JSONDecodeError:
            pattern = r'\b(?:PL|PEC|PLP|PDC|INC)\s+[\d]+/\d{4}\b'
            selected_names = re.findall(pattern, response_text)

        filtered = [n for n in selected_names if n in valid_names]
        result_bills = [candidate_map[n] for n in filtered]
        selected_set = set(filtered)
        for cand in candidates:
            if cand.bill.name not in selected_set and len(result_bills) < top_n:
                result_bills.append(cand.bill)

        tokens_in = message.usage.input_tokens
        tokens_out = message.usage.output_tokens
        cost = (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000

        self.stats.queries_processed += 1
        self.stats.total_tokens_in += tokens_in
        self.stats.total_tokens_out += tokens_out
        self.stats.total_latency_s += latency

        logger.debug(f"Anthropic rerank: {tokens_in}+{tokens_out} tokens, ${cost:.5f}")

        return [
            RerankerResult(bill=b, rank=i + 1, tokens_used=tokens_in + tokens_out)
            for i, b in enumerate(result_bills[:top_n])
        ]


class OpenRouterReranker:
    """
    SLM/LLM reranker via OpenRouter API.

    Supports additional reranker models not available on DeepInfra:
    - Mistral Ministral 3B (2025, 128K context) - excellent for reranking
    - Google Gemma 3n 2B (FREE!) - ultra-fast, cost-effective
    - Qwen2.5 Coder 7B - code-specialized (experimental for legislative domain)

    Same interface as OllamaReranker and DeepInfraReranker — drop-in replacement.
    Requires OPENROUTER_API_KEY in environment.

    Reference for new models:
      - Mistral AI (2025). Ministral Technical Report.
      - Google (2025). Gemma 3 Technical Report. arXiv:2503.19786.

    Example:
        reranker = OpenRouterReranker(model="mistralai/ministral-3b-2512")
        results = reranker.rerank(query="cavalos de raça", candidates=bm25_results, top_n=12)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        top_n: int | None = None,
    ) -> None:
        import openai
        from mestrado.config import openrouter as cfg, experiment as exp_cfg
        key = api_key or cfg.api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self.model = model or cfg.reranker_model
        self.top_n = top_n or exp_cfg.rerank_top_n
        self.stats = RerankerStats()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RerankerResult]:
        """Zero-shot reranking via OpenRouter SLM. Same pattern as OllamaReranker."""
        top_n = top_n or self.top_n
        if not candidates:
            return []

        bills = [r.bill for r in candidates]
        candidate_map: dict[str, Bill] = {b.name: b for b in bills}
        valid_names: set[str] = set(candidate_map.keys())

        manifest = OllamaReranker._format_bill_manifest(None, bills)  # type: ignore[arg-type]
        system = LEGISLATIVE_RERANKER_SYSTEM_PROMPT.format(top_n=top_n)
        user_content = (
            f'Consulta do assessor: "{query}"\n\n'
            f"Projetos de lei disponíveis:\n\n{manifest}\n\n"
            f"Selecione os {top_n} mais relevantes em ordem decrescente de relevância."
        )

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            latency = time.time() - t0
            response_text = response.choices[0].message.content or "{}"
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
        except Exception as e:
            logger.warning(f"OpenRouter rerank call failed ({self.model}): {e}")
            return [
                RerankerResult(bill=r.bill, rank=i + 1)
                for i, r in enumerate(candidates[:top_n])
            ]

        try:
            data = json.loads(response_text)
            selected_names = data.get("selected_bills", [])
            if not isinstance(selected_names, list):
                selected_names = []
        except json.JSONDecodeError:
            pattern = r'\b(?:PL|PEC|PLP|PDC|INC)\s+[\d]+/\d{4}\b'
            selected_names = re.findall(pattern, response_text)

        filtered = [n for n in selected_names if n in valid_names]
        hallucinated = len(selected_names) - len(filtered)
        if hallucinated:
            logger.debug(f"OpenRouter: filtered {hallucinated} hallucinated names.")

        result_bills = [candidate_map[n] for n in filtered if n in candidate_map]
        selected_set = set(filtered)
        for cand in candidates:
            if cand.bill.name not in selected_set and len(result_bills) < top_n:
                result_bills.append(cand.bill)

        # Cost estimation (OpenRouter pricing varies by model)
        # This is a rough estimate; actual cost depends on model-specific pricing
        cost = (tokens_in * 0.0001 + tokens_out * 0.0001) / 1_000_000

        self.stats.queries_processed += 1
        self.stats.total_tokens_in += tokens_in
        self.stats.total_tokens_out += tokens_out
        self.stats.total_latency_s += latency
        self.stats.hallucinations_filtered += hallucinated

        logger.debug(
            f"OpenRouter rerank ({self.model}): {tokens_in}+{tokens_out} tok, "
            f"${cost:.5f}, {latency:.1f}s"
        )
        return [
            RerankerResult(bill=b, rank=i + 1, tokens_used=tokens_in + tokens_out)
            for i, b in enumerate(result_bills[:top_n])
        ]
