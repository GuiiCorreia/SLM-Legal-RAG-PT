"""
Token estimation utilities — ported from src/services/tokenEstimation.ts.

The TypeScript implementation uses:
- API-based counting (exact)
- Haiku fallback (approximate)
- Rough heuristic: 4 chars/token for English text

We adapt for PT-BR legislative text where:
- Regular text: ~3.5 chars/token (Portuguese has more diacritics than English)
- JSON/structured: ~2.0 chars/token (numbers, keys are compact)
- Preprocessed token lists: ~2.5 chars/token (already tokenized)
- Legislative text (formal, long words): ~4.0 chars/token

References:
- tokenEstimation.ts: roughTokenCountEstimation(), bytesPerTokenForFileType()
- Adapted for PT-BR: adjusted ratio from empirical observation
"""
from __future__ import annotations

import math
from typing import Union


# Chars-per-token ratios — from tokenEstimation.ts, adjusted for PT-BR
CHARS_PER_TOKEN: dict[str, float] = {
    "text": 3.5,          # Regular PT-BR text
    "legislative": 4.0,   # Formal legislative text (long technical words)
    "json": 2.0,          # JSON (numbers, brackets, keys are compact)
    "list": 2.5,          # Python list strings (preprocessed_tokens field)
    "query": 3.0,         # Short informal queries
    "ementa": 3.8,        # Bill summaries (semi-formal)
    "default": 3.5,
}


def estimate_tokens(
    text: str | list,
    content_type: str = "default",
) -> int:
    """
    Rough token count estimation — ported from roughTokenCountEstimation().

    Analogous to tokenEstimation.ts:
        Math.ceil(content.length / ROUGH_CHARS_PER_TOKEN)

    Args:
        text: String or list (lists are converted to their repr)
        content_type: key into CHARS_PER_TOKEN dict

    Returns:
        Estimated token count (integer, rounded up)
    """
    if isinstance(text, list):
        # List of tokens — count directly (already tokenized)
        return len(text)

    ratio = CHARS_PER_TOKEN.get(content_type, CHARS_PER_TOKEN["default"])
    return math.ceil(len(text) / ratio)


def estimate_tokens_for_bill(
    ementa: str,
    full_text: str = "",
    mode: str = "ementa",
) -> int:
    """
    Estimate tokens for a bill in different representation modes.
    Used to determine if a bill fits in SLM context window.

    Analogous to bytesPerTokenForFileType() in tokenEstimation.ts.
    """
    if mode == "ementa":
        return estimate_tokens(ementa, "ementa")
    elif mode == "full":
        return estimate_tokens(ementa, "ementa") + estimate_tokens(full_text, "legislative")
    elif mode == "summary":
        return 512  # Estimated tokens for a structured LLM-generated summary
    return estimate_tokens(ementa, "ementa")


def fits_in_context(
    text: Union[str, list],
    max_tokens: int = 4096,
    content_type: str = "default",
) -> bool:
    """
    Check if text fits within a context window.
    Used for SLM context budget management.

    Analogous to the token threshold checks in compact.ts:
        if (tokenCount > COMPACT_THRESHOLD) { compact() }
    """
    return estimate_tokens(text, content_type) <= max_tokens


def context_utilization(
    texts: list[str],
    max_tokens: int = 4096,
    content_types: list[str] | None = None,
) -> dict:
    """
    Compute context window utilization statistics.
    Analogous to analyzeContext() / TokenStats in contextAnalysis.ts.

    Returns breakdown by component with total and utilization percentage.
    """
    if content_types is None:
        content_types = ["default"] * len(texts)

    totals = [estimate_tokens(t, ct) for t, ct in zip(texts, content_types)]
    total = sum(totals)
    utilization = min(1.0, total / max_tokens)

    return {
        "total_tokens": total,
        "max_tokens": max_tokens,
        "utilization_pct": round(100 * utilization, 1),
        "fits": total <= max_tokens,
        "per_component": totals,
        "overflow_tokens": max(0, total - max_tokens),
    }


def truncate_to_budget(
    text: str,
    token_budget: int,
    content_type: str = "default",
) -> tuple[str, bool]:
    """
    Truncate text to fit within a token budget.
    Returns (truncated_text, was_truncated).

    Analogous to the truncation logic in compact.ts for file restoration.
    """
    current = estimate_tokens(text, content_type)
    if current <= token_budget:
        return text, False

    ratio = CHARS_PER_TOKEN.get(content_type, CHARS_PER_TOKEN["default"])
    max_chars = int(token_budget * ratio)
    return text[:max_chars] + "\n[...]", True
