"""Benchmark runner for PromptFuse evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.evaluation.metrics import BenchmarkMetrics, compute_rouge_l
from promptfuse.pipeline import PromptFusePipeline

logger = logging.getLogger(__name__)


def load_prompts(path: Path) -> list[str]:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [d.get("prompt", d.get("text", "")) for d in data]
            return data
        return data.get("prompts", [])
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_benchmark(
    prompts: list[str],
    compression_ratio: float = 0.40,
    *,
    enable_compression: bool = True,
    enable_unification: bool = True,
    mock_responses: bool = True,
) -> BenchmarkMetrics:
    """
    Run PromptFuse pipeline over a prompt set and collect metrics.

    When mock_responses=True, quality is estimated by comparing compressed vs raw
    prompt text (proxy for output fidelity without running vLLM).
    """
    config = Settings().load()
    pipeline = PromptFusePipeline(
        config,
        lazy_load=False,
        enable_compression=enable_compression,
        enable_unification=enable_unification,
    )

    metrics = BenchmarkMetrics()

    for prompt in prompts:
        result = pipeline.process(prompt, compression_ratio=compression_ratio)

        rouge_l = None
        if mock_responses and result.compression:
            rouge_l = compute_rouge_l(result.raw_prompt, result.final_prompt)

        metrics.record(
            token_reduction=result.token_reduction,
            rouge_l=rouge_l,
            latency_ms=result.total_ms,
            cache_hit=result.cache_hit,
        )

    return metrics


def compare_baselines(prompts: list[str], compression_ratio: float = 0.40) -> dict:
    """Compare no-compression, compression-only, and full PromptFuse."""
    results = {}

    # Baseline: no compression, no unification
    t0 = time.perf_counter()
    no_comp = run_benchmark(
        prompts,
        compression_ratio=0.0,
        enable_compression=False,
        enable_unification=False,
        mock_responses=False,
    )
    results["no_compression"] = {**no_comp.summary(), "wall_time_s": time.perf_counter() - t0}

    # Compression only (LLMLingua-style, no unifier)
    t0 = time.perf_counter()
    comp_only = run_benchmark(
        prompts,
        compression_ratio=compression_ratio,
        enable_compression=True,
        enable_unification=False,
    )
    results["compression_only"] = {**comp_only.summary(), "wall_time_s": time.perf_counter() - t0}

    # Full PromptFuse
    t0 = time.perf_counter()
    full = run_benchmark(
        prompts,
        compression_ratio=compression_ratio,
        enable_compression=True,
        enable_unification=True,
    )
    results["promptfuse_full"] = {**full.summary(), "wall_time_s": time.perf_counter() - t0}

    config = Settings().load()
    baseline_hits = comp_only.unifier_hit_rate  # 0 for compression-only
    full_hits = full.unifier_hit_rate
    results["cache_hit_improvement"] = (
        full_hits / baseline_hits if baseline_hits > 0 else float("inf") if full_hits > 0 else 1.0
    )

    results["goals"] = {
        "token_reduction_target": 0.30,
        "rouge_l_target": config.evaluation.rouge_l_threshold,
        "latency_p99_target_ms": config.evaluation.latency_p99_ms,
        "meets_token_reduction": full.avg_token_reduction >= 0.30,
        "meets_rouge_l": full.avg_rouge_l >= config.evaluation.rouge_l_threshold,
        "meets_latency": full.p99_latency_ms <= config.evaluation.latency_p99_ms,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PromptFuse benchmark")
    parser.add_argument("--prompts", type=Path, default=Path("data/sample_prompts.txt"))
    parser.add_argument("--ratio", type=float, default=0.40)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--compare-baselines", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.prompts.exists():
        logger.error("Prompt file not found: %s", args.prompts)
        raise SystemExit(1)

    prompts = load_prompts(args.prompts)
    logger.info("Loaded %d prompts", len(prompts))

    if args.compare_baselines:
        results = compare_baselines(prompts, args.ratio)
    else:
        metrics = run_benchmark(prompts, args.ratio)
        results = metrics.summary()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
