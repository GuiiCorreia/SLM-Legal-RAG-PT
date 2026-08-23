"""
Configuration management — loads .env and exposes typed settings.
Pattern: centralized config avoids scattered os.getenv() calls.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (codigos_mestrado/)
# config.py is in src/mestrado/, so we need 3 levels up
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    reranker_model: str = os.getenv("OLLAMA_RERANKER_MODEL", "llama3.2:3b")
    generator_model: str = os.getenv("OLLAMA_GENERATOR_MODEL", "phi3:mini")
    timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))


class AnthropicConfig:
    api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    reranker_model: str = os.getenv("ANTHROPIC_RERANKER_MODEL", "claude-haiku-4-5-20251001")
    generator_model: str = os.getenv("ANTHROPIC_GENERATOR_MODEL", "claude-sonnet-4-6")


class DeepInfraConfig:
    """DeepInfra — OpenAI-compatible API para modelos open-source. Base: api.deepinfra.com/v1/openai"""
    api_key: str | None = os.getenv("DEEPINFRA_API_KEY")
    base_url: str = "https://api.deepinfra.com/v1/openai"
    timeout: int = int(os.getenv("DEEPINFRA_TIMEOUT", "120"))
    # Modelos recomendados para cada uso (IDs exatos da plataforma)
    reranker_model: str = os.getenv("DEEPINFRA_RERANKER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    generator_model: str = os.getenv("DEEPINFRA_GENERATOR_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    judge_model: str = os.getenv("DEEPINFRA_JUDGE_MODEL", "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo")
    qe_model: str = os.getenv("DEEPINFRA_QE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")


class OpenRouterConfig:
    """OpenRouter API — acesso a modelos modernos (Llama-3.3-70B, Qwen-2.5-72B, etc.)"""
    api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    base_url: str = "https://openrouter.ai/api/v1"
    judge_model: str = os.getenv(
        "OPENROUTER_JUDGE_MODEL",
        "meta-llama/llama-3.3-70b-instruct",  # forte em PT-BR, ~$0.40/M tokens
    )
    reranker_model: str = os.getenv("OPENROUTER_RERANKER_MODEL", "meta-llama/llama-3.2-3b-instruct")
    generator_model: str = os.getenv("OPENROUTER_GENERATOR_MODEL", "meta-llama/llama-3.1-8b-instruct")


class MaritacaConfig:
    """Maritaca AI — Gaia/Sabiá, especializado em português brasileiro."""
    api_key: str | None = os.getenv("MARITACA_API_KEY")
    base_url: str = "https://chat.maritaca.ai/api"
    judge_model: str = os.getenv("MARITACA_JUDGE_MODEL", "sabia-3")


class LLMProviderConfig:
    """
    Seleciona qual provedor LLM usar para inferência remota.
    Valores: "deepinfra" | "openrouter" | "ollama"
    Pode ser sobrescrito por argumento explícito em get_client().
    """
    provider: str = os.getenv("LLM_PROVIDER", "deepinfra")


class SLMComparisonConfig:
    """
    Configuration for SLM comparison experiments (cross-domain paper).
    Models tested: 7 SLMs (Google: 1, Meta: 3, Qwen2.5: 3) + Oracle LLM 70B.
    Oracle LLM: Llama 3.3 70B (reference for H5 evaluation).

    Full analysis of all SLM models on DeepInfra (≤10B parameters):
      See mestrado/slm_comparison/TODOS_MODELOS_DEEPINFRA.md

    CRITICAL BUG DISCOVERED (2026-04-07 diagnostic):
      ALL Qwen3.5 models (0.8B, 2B, 4B, 9B) have systematic bug on DeepInfra API.
      Diagnostic results: tokens generated (50-1024) but message.content = ''.
      Bug confirmed across ALL scenarios (simple prompt, RAG, all max_tokens values).
      Root cause: DeepInfra API bug for Qwen3.5 family, not implementation issue.
      Solution: Use Qwen2.5 models (1.5B, 3B, 7B) which work perfectly.

    Working SLM models (7 total, validated 2026-04-07):
      - Google: Gemma 3 4B
      - Meta: Llama 3.1 8B, Llama 3.2 3B, Llama 3.2 1B
      - Qwen: Qwen2.5 1.5B, 3B, 7B

    References:
      - Belcak & Wattenhofer (2025) arXiv:2506.02153 — SLMs as future of agentic AI
      - Google (2025) arXiv:2503.19786 — Gemma 3 technical report
      - Dubey et al. (2024) arXiv:2407.21783 — Llama 3.1 technical report
      - Meta AI (2024) — Llama 3.2 model card
      - Qwen Team (2024) — Qwen2.5 technical report
    """
    # Comma-separated list of SLM model IDs on DeepInfra
    # 7 functional models validated via diagnostic (2026-04-07)
    # All Qwen3.5 models excluded due to DeepInfra API bug (empty responses)
    # Full analysis: mestrado/slm_comparison/TODOS_MODELOS_DEEPINFRA.md
    slm_models_str: str = os.getenv(
        "SLM_MODELS",
        "google/gemma-3-4b-it,"
        "meta-llama/Meta-Llama-3.1-8B-Instruct,"
        "meta-llama/Llama-3.2-3B-Instruct,"
        "meta-llama/Llama-3.2-1B-Instruct,"
        "Qwen/Qwen2.5-1.5B-Instruct,"
        "Qwen/Qwen2.5-3B-Instruct,"
        "Qwen/Qwen2.5-7B-Instruct",
    )
    # Oracle LLM for H5 reference quality (E6)
    oracle_model: str = os.getenv(
        "LLM_ORACLE_MODEL",
        "meta-llama/Llama-3.3-70B-Instruct",
    )
    # SLM used as reranker (E4 variant, extends Sun et al. 2023 to SLMs)
    slm_reranker_model: str = os.getenv(
        "SLM_RERANKER_MODEL",
        "microsoft/phi-4-mini-instruct",
    )
    # Number of queries for experiments (100 = validation, 692 = full Ulysses)
    n_queries: int = int(os.getenv("SLM_N_QUERIES", "100"))
    # Random seed for reproducible query sampling
    seed: int = int(os.getenv("SLM_SEED", "42"))

    @property
    def slm_models(self) -> list[str]:
        """Parse comma-separated SLM_MODELS into a list."""
        return [m.strip() for m in self.slm_models_str.split(",") if m.strip()]


class DataConfig:
    project_root: Path = _PROJECT_ROOT
    bills_path: Path = _PROJECT_ROOT / os.getenv("BILLS_DATASET_PATH", "bills_dataset.csv")
    feedback_path: Path = _PROJECT_ROOT / os.getenv("FEEDBACK_DATASET_PATH", "relevance_feedback_dataset.csv")
    # BGE-M3 is the default for this branch (Chen et al. 2024, arXiv:2402.03216)
    # Alternatives: paraphrase-multilingual-MiniLM-L12-v2 | intfloat/multilingual-e5-large
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


class ExperimentConfig:
    sample_size: int = int(os.getenv("EXPERIMENT_SAMPLE_SIZE", "200"))
    top_k_retrieval: int = int(os.getenv("EXPERIMENT_TOP_K_RETRIEVAL", "50"))
    rerank_top_n: int = int(os.getenv("EXPERIMENT_RERANK_TOP_N", "12"))
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("BM25_B", "0.75"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))


class OutputConfig:
    results_dir: Path = _PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")
    logs_dir: Path = _PROJECT_ROOT / os.getenv("LOGS_DIR", "logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self) -> None:
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Singleton instances — import these throughout the project
ollama = OllamaConfig()
anthropic = AnthropicConfig()
deepinfra = DeepInfraConfig()
openrouter = OpenRouterConfig()
maritaca = MaritacaConfig()
llm_provider = LLMProviderConfig()
slm_comparison = SLMComparisonConfig()
data = DataConfig()
experiment = ExperimentConfig()
output = OutputConfig()

# Ensure output directories exist
output.results_dir.mkdir(exist_ok=True)
output.logs_dir.mkdir(exist_ok=True)
