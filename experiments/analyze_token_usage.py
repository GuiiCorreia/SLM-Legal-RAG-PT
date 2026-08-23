"""
Analyze Token Usage and Context Patterns Across SLMs.

This script investigates how different SLMs use their context window and generates responses.

Metrics analyzed:
- Average tokens generated per query
- Average context tokens used
- Response length in words
- Token efficiency (quality per token)
- Correlation between response length and quality

Objective:
- Investigate if Llama 3.2 1B uses resources differently than larger models
- Determine if more concise responses correlate with higher quality
- Identify "sweet spots" for token usage vs quality

Reference:
- Belcak & Wattenhofer (2025). Small Language Models Are the Future of Agentic AI. arXiv:2506.02153.
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.

Usage:
    python analyze_token_usage.py --results results/llama_family_100queries_YYYYMMDD_HHMMSS.json
"""
import sys
from pathlib import Path
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass
class TokenAnalysis:
    """Token usage analysis for a model."""
    model: str
    provider: str
    avg_tokens_in: float
    avg_tokens_out: float
    avg_latency: float
    tokens_per_second: float
    bertscore: float
    rouge_l: float
    faithfulness: float
    quality_per_token: float  # BERTScore / total tokens


def analyze_tokens(results_file: Path):
    """Analyze token usage patterns across models."""
    print("=" * 80)
    print("ANÁLISE DE USO DE TOKENS E CONTEXTO")
    print("=" * 80)

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    oracle_model = data.get("oracle_model", "")
    slm_models = [m for m in results.keys() if m != oracle_model]

    print(f"\n📂 Resultados carregados: {results_file}")
    print(f"📊 Modelos analisados: {len(slm_models)} SLMs")

    # Create token analyses for each model
    token_analyses = []
    for model in slm_models:
        r = results[model]
        tokens_in = r["tokens_in_avg"]
        tokens_out = r["tokens_out_avg"]
        latency = r["latency_avg"]

        # Calculate tokens per second
        total_tokens = tokens_in + tokens_out
        tokens_per_sec = total_tokens / latency if latency > 0 else 0

        analysis = TokenAnalysis(
            model=model.split('/')[-1],
            provider=r.get("provider", "Unknown"),
            avg_tokens_in=tokens_in,
            avg_tokens_out=tokens_out,
            avg_latency=latency,
            tokens_per_second=tokens_per_sec,
            bertscore=r.get("bertscore", 0),
            rouge_l=r.get("rouge_l", 0),
            faithfulness=r.get("faithfulness_avg", 0),
            quality_per_token=r.get("bertscore", 0) / total_tokens if total_tokens > 0 else 0,
        )
        token_analyses.append(analysis)

    # Print token usage table
    print("\n" + "=" * 80)
    print("USO DE TOKENS POR MODELO")
    print("=" * 80)
    print(f"{'Modelo':<35} {'In':>8} {'Out':>8} {'Total':>9} {'Tokens/s':>10} {'Lat(s)':>7}")
    print("-" * 80)

    for ta in token_analyses:
        print(f"{ta.model:<35} {ta.avg_tokens_in:>8.0f} {ta.avg_tokens_out:>8.0f} "
              f"{(ta.avg_tokens_in + ta.avg_tokens_out):>9.0f} {ta.tokens_per_second:>10.1f} {ta.avg_latency:>7.1f}")

    print("=" * 80)

    # Print quality vs token efficiency
    print("\n" + "=" * 80)
    print("QUALIDADE VS EFICIÊNCIA DE TOKENS")
    print("=" * 80)
    print(f"{'Modelo':<35} {'BERTScore':>11} {'Tokens':>9} {'Qual/Token':>12}")
    print("-" * 80)

    for ta in token_analyses:
        print(f"{ta.model:<35} {ta.bertscore:>11.4f} "
              f"{(ta.avg_tokens_in + ta.avg_tokens_out):>9.0f} {ta.quality_per_token:>12.6f}")

    print("=" * 80)

    # Find best by different metrics
    best_bertscore = max(token_analyses, key=lambda x: x.bertscore)
    fastest = max(token_analyses, key=lambda x: x.tokens_per_second)
    most_efficient = max(token_analyses, key=lambda x: x.quality_per_token)
    most_concise = min(token_analyses, key=lambda x: x.avg_tokens_out)
    most_verbose = max(token_analyses, key=lambda x: x.avg_tokens_out)

    print("\n" + "=" * 80)
    print("MELHORES POR CATEGORIA")
    print("=" * 80)

    print(f"\n🏆 Melhor BERTScore (Qualidade):")
    print(f"   {best_bertscore.model}: {best_bertscore.bertscore:.4f} "
          f"(tokens: {best_bertscore.avg_tokens_out:.0f} out)")

    print(f"\n⚡ Mais Rápido (Tokens/s):")
    print(f"   {fastest.model}: {fastest.tokens_per_second:.1f} tokens/s "
          f"(latência: {fastest.avg_latency:.1f}s)")

    print(f"\n💎 Mais Eficiente (Qualidade/Token):")
    print(f"   {most_efficient.model}: {most_efficient.quality_per_token:.6f} "
          f"(BERTScore: {most_efficient.bertscore:.4f})")

    print(f"\n📝 Mais Conciso (Menos tokens out):")
    print(f"   {most_concise.model}: {most_concise.avg_tokens_out:.0f} tokens out")

    print(f"\n📚 Mais Verboso (Mais tokens out):")
    print(f"   {most_verbose.model}: {most_verbose.avg_tokens_out:.0f} tokens out")

    # Analyze correlation
    print("\n" + "=" * 80)
    print("CORRELAÇÕES")
    print("=" * 80)

    # Correlation: tokens_out vs bertscore
    tokens_out = [ta.avg_tokens_out for ta in token_analyses]
    bertscores = [ta.bertscore for ta in token_analyses]

    # Simple Pearson correlation
    import statistics
    mean_tokens = statistics.mean(tokens_out)
    mean_bert = statistics.mean(bertscores)

    covariance = sum((t - mean_tokens) * (b - mean_bert) for t, b in zip(tokens_out, bertscores))
    std_tokens = statistics.stdev(tokens_out) if len(tokens_out) > 1 else 0
    std_bert = statistics.stdev(bertscores) if len(bertscores) > 1 else 0

    correlation = covariance / (std_tokens * std_bert) if std_tokens > 0 and std_bert > 0 else 0

    print(f"\n📊 Correlação: Tokens Gerados vs BERTScore")
    print(f"   r = {correlation:.3f}")

    if correlation > 0.3:
        print(f"   ✅ POSITIVA: Mais tokens = melhor qualidade (prolixidade ajuda)")
    elif correlation < -0.3:
        print(f"   ✅ NEGATIVA: Menos tokens = melhor qualidade (concisão ajuda)")
    else:
        print(f"   ⚠️  NEUTRA: Sem correlação clara")

    # Correlation: latency vs bertscore
    latencies = [ta.avg_latency for ta in token_analyses]

    mean_latency = statistics.mean(latencies)
    covariance_latency = sum((l - mean_latency) * (b - mean_bert) for l, b in zip(latencies, bertscores))
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

    correlation_latency = covariance_latency / (std_latency * std_bert) if std_latency > 0 and std_bert > 0 else 0

    print(f"\n📊 Correlação: Latência vs BERTScore")
    print(f"   r = {correlation_latency:.3f}")

    if correlation_latency > 0.3:
        print(f"   ✅ POSITIVA: Mais lento = melhor qualidade")
    elif correlation_latency < -0.3:
        print(f"   ✅ NEGATIVA: Mais rápido = melhor qualidade")
    else:
        print(f"   ⚠️  NEUTRA: Sem correlação clara")

    # Check Llama 3.2 1B specifically
    llama_1b = next((ta for ta in token_analyses if "1B" in ta.model), None)
    if llama_1b:
        print("\n" + "=" * 80)
        print("ANÁLISE: Llama 3.2 1B (O Menor Modelo)")
        print("=" * 80)

        avg_tokens_out = statistics.mean(tokens_out)
        avg_bertscore = statistics.mean(bertscores)

        print(f"\n📊 Comparação com a média:")
        print(f"   Tokens out:  {llama_1b.avg_tokens_out:.0f} vs média {avg_tokens_out:.0f} "
              f"({'+' if llama_1b.avg_tokens_out > avg_tokens_out else ''}{llama_1b.avg_tokens_out - avg_tokens_out:.0f})")
        print(f"   BERTScore:   {llama_1b.bertscore:.4f} vs média {avg_bertscore:.4f} "
              f"({'+' if llama_1b.bertscore > avg_bertscore else ''}{llama_1b.bertscore - avg_bertscore:.4f})")
        print(f"   Qual/Token:  {llama_1b.quality_per_token:.6f} vs média {most_efficient.quality_per_token:.6f}")

        if llama_1b.avg_tokens_out < avg_tokens_out and llama_1b.bertscore > avg_bertscore:
            print(f"\n   ✅ Llama 3.2 1B é MAIS CONCISO E MELHOR que a média!")
            print(f"   💡 Isso sugere que o modelo usa tokens de forma mais eficiente")
        elif llama_1b.avg_tokens_out < avg_tokens_out:
            print(f"\n   📝 Llama 3.2 1B é mais conciso que a média")
        elif llama_1b.bertscore > avg_bertscore:
            print(f"\n   🎯 Llama 3.2 1B tem melhor qualidade que a média")

    print("\n" + "=" * 80)
    print("CONCLUSÕES")
    print("=" * 80)

    conclusions = []

    if correlation < -0.3:
        conclusions.append("- Modelos mais concisos tendem a ter melhor qualidade")

    if correlation_latency < -0.3:
        conclusions.append("- Modelos mais rápidos tendem a ter melhor qualidade")

    if most_efficient.model == best_bertscore.model:
        conclusions.append(f"- O modelo mais eficiente ({most_efficient.model}) também é o melhor em qualidade")

    if llama_1b and llama_1b == best_bertscore:
        conclusions.append("- Llama 3.2 1B (menor) é o melhor, sugerindo que tamanho ≠ qualidade")

    if not conclusions:
        conclusions.append("- Não há padrões claros; qualidade depende de fatores além de tokens/latência")

    for i, c in enumerate(conclusions, 1):
        print(f"{i}. {c}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token usage patterns across SLMs"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results JSON file from experiment"
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"❌ Erro: Arquivo não encontrado: {args.results}")
        return 1

    analyze_tokens(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
