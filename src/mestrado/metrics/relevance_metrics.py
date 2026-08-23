"""
Relevance and Coherence metrics using LLM-as-judge methodology.

Based on MT-Bench evaluation protocol (Zheng et al., NeurIPS 2023).
These metrics evaluate the quality of RAG responses beyond lexical similarity.

References:
  - Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
    NeurIPS 2023.
  - MT-Bench Evaluation Protocol. https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# LLM-as-judge prompts (based on MT-Bench protocol)

RELEVANCE_PROMPT_TEMPLATE = """\
Avalie a relevância da resposta gerada para a consulta do usuário.

Consulta:
{query}

Resposta gerada:
{response}

Instruções:
1. Avalie se a resposta responde diretamente à consulta.
2. Considere se a resposta aborda todos os aspectos da consulta.
3. Desconsidere a qualidade da escrita - foque apenas na relevância.

Score (retorne APENAS um número de 0.0 a 1.0):
- 1.0 = Responde completamente à consulta, aborda todos os aspectos
- 0.7 = Responde bem à consulta, mas pode faltar algum detalhe menor
- 0.5 = Responde parcialmente à consulta, apenas alguns aspectos
- 0.3 = Responde muito superficialmente ou tangencialmente
- 0.0 = Não responde à consulta ou é completamente irrelevante

Score:
"""

COHERENCE_PROMPT_TEMPLATE = """\
Avalie a coerência e fluidez da resposta gerada.

Resposta gerada:
{response}

Instruções:
1. Avalie se a resposta é lógica e coerente internamente.
2. Verifique se há contradições ou inconsistências.
3. Avalie se a resposta flui naturalmente de início ao fim.
4. Considere a clareza da estrutura e organização.

Score (retorne APENAS um número de 0.0 a 1.0):
- 1.0 = Completamente coerente, lógica, bem estruturada e fluida
- 0.7 = Principalmente coerente, pequenas inconsistências ou desorganização menor
- 0.5 = Moderadamente coerente, algumas inconsistências ou problemas de estrutura
- 0.3 = Pouco coerente, múltiplas inconsistências ou dificuldade de seguir o raciocínio
- 0.0 = Incoerente, contraditória, impossível de seguir

Score:
"""


def parse_llm_score(content: str) -> float:
    """
    Parse LLM output to extract a score between 0.0 and 1.0.

    Multiple parsing strategies for robustness:
    1. Look for decimal number pattern (0.XX or 1.0 or 0.0)
    2. Look for percentage (XX%)
    3. Look for descriptive keywords

    Args:
        content: Raw LLM response

    Returns:
        Score between 0.0 and 1.0 (default 0.5 if parsing fails)
    """
    # Strategy 1: Look for decimal number pattern
    match = re.search(r'\b(0\.\d+|1\.0|0\.0)\b', content)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Strategy 2: Look for percentage
    match = re.search(r'(\d+)%', content)
    if match:
        try:
            return float(match.group(1)) / 100.0
        except ValueError:
            pass

    # Strategy 3: Look for descriptive keywords
    content_lower = content.lower()
    if any(phrase in content_lower for phrase in ['completamente', 'totalmente', 'excelente', 'perfeito']):
        return 1.0
    elif any(phrase in content_lower for phrase in ['principalmente', 'bem', 'boa', 'adequada']):
        return 0.7
    elif any(phrase in content_lower for phrase in ['moderadamente', 'algumas', 'parcial', 'razoável']):
        return 0.5
    elif any(phrase in content_lower for phrase in ['pouco', 'limitada', 'fraca', 'insuficiente']):
        return 0.3
    elif any(phrase in content_lower for phrase in ['incoerente', 'contraditória', 'impossível', 'não']):
        return 0.0

    # Default fallback
    logger.warning(f"Failed to parse score from: {content[:100]}... Using default 0.5")
    return 0.5


def evaluate_relevance(
    query: str,
    response: str,
    judge_model: Any,
) -> float:
    """
    Evaluate relevance of response to query using LLM-as-judge.

    Args:
        query: User query/question
        response: Generated RAG response
        judge_model: LLM instance for evaluation (e.g., Oracle or separate judge)

    Returns:
        Relevance score between 0.0 and 1.0

    Reference: Zheng et al. (2023) - MT-Bench relevance evaluation
    """
    if not query or not response:
        return 0.0

    prompt = RELEVANCE_PROMPT_TEMPLATE.format(query=query, response=response)

    try:
        # Use judge model's client directly (supports both DeepInfra and OpenRouter)
        result = judge_model._client.chat.completions.create(
            model=judge_model.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )

        content = result.choices[0].message.content.strip()
        score = parse_llm_score(content)

        logger.debug(f"Relevance evaluation: score={score:.2f}, query_len={len(query)}, response_len={len(response)}")
        return score

    except Exception as e:
        logger.warning(f"Relevance evaluation failed: {e}. Using default 0.5")
        return 0.5


def evaluate_coherence(
    response: str,
    judge_model: Any,
) -> float:
    """
    Evaluate coherence and fluency of response using LLM-as-judge.

    Args:
        response: Generated RAG response
        judge_model: LLM instance for evaluation (e.g., Oracle or separate judge)

    Returns:
        Coherence score between 0.0 and 1.0

    Reference: Zheng et al. (2023) - MT-Bench coherence evaluation
    """
    if not response:
        return 0.0

    prompt = COHERENCE_PROMPT_TEMPLATE.format(response=response)

    try:
        # Use judge model's client directly
        result = judge_model._client.chat.completions.create(
            model=judge_model.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )

        content = result.choices[0].message.content.strip()
        score = parse_llm_score(content)

        logger.debug(f"Coherence evaluation: score={score:.2f}, response_len={len(response)}")
        return score

    except Exception as e:
        logger.warning(f"Coherence evaluation failed: {e}. Using default 0.5")
        return 0.5


def evaluate_relevance_batch(
    queries: list[str],
    responses: list[str],
    judge_model: Any,
) -> list[float]:
    """
    Evaluate relevance for multiple query-response pairs.

    Args:
        queries: List of user queries
        responses: List of generated responses (same length as queries)
        judge_model: LLM instance for evaluation

    Returns:
        List of relevance scores (0.0-1.0) for each pair
    """
    if len(queries) != len(responses):
        raise ValueError(f"Length mismatch: {len(queries)} queries vs {len(responses)} responses")

    scores = []
    for query, response in zip(queries, responses):
        score = evaluate_relevance(query, response, judge_model)
        scores.append(score)

    return scores


def evaluate_coherence_batch(
    responses: list[str],
    judge_model: Any,
) -> list[float]:
    """
    Evaluate coherence for multiple responses.

    Args:
        responses: List of generated responses
        judge_model: LLM instance for evaluation

    Returns:
        List of coherence scores (0.0-1.0) for each response
    """
    scores = []
    for response in responses:
        score = evaluate_coherence(response, judge_model)
        scores.append(score)

    return scores
