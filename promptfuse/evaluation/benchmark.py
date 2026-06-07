"""Benchmark runner for PromptFuse evaluation."""

from __future__ import annotations

import argparse
from enum import StrEnum
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.evaluation.metrics import BenchmarkMetrics, compute_rouge_l
from promptfuse.pipeline import PromptFusePipeline

logger = logging.getLogger(__name__)


class BaselineMode(StrEnum):
    NO_COMPRESSION = "no_compression"
    COMPRESSION_ONLY = "compression_only"
    PROMPTFUSE_FULL = "promptfuse_full"
    LLMLINGUA_ONLY = "llmlingua_only"
    EXACT_CACHE_ONLY_STUB = "exact_cache_only_stub"


@dataclass
class BenchmarkModeResult:
    mode: str
    status: str
    metrics: dict
    wall_time_s: float
    compression_ratio: float
    settings: dict
    warnings: list[str] = field(default_factory=list)


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


def _run_mode(
    mode: BaselineMode,
    prompts: list[str],
    compression_ratio: float,
) -> BenchmarkModeResult:
    """
    Run one baseline mode and return normalized output.

    LLMLingua and exact-cache modes are CPU placeholders in this implementation.
    """
    mode_settings: dict[str, bool]
    warnings: list[str] = []
    status = "ok"

    if mode == BaselineMode.NO_COMPRESSION:
        mode_settings = {"enable_compression": False, "enable_unification": False, "mock_responses": False}
    elif mode == BaselineMode.COMPRESSION_ONLY:
        mode_settings = {"enable_compression": True, "enable_unification": False, "mock_responses": True}
    elif mode == BaselineMode.PROMPTFUSE_FULL:
        mode_settings = {"enable_compression": True, "enable_unification": True, "mock_responses": True}
    elif mode == BaselineMode.LLMLINGUA_ONLY:
        mode_settings = {"enable_compression": False, "enable_unification": False, "mock_responses": False}
        warnings.append(
            "LLMLingua adapter is not wired in this CPU workflow. "
            "Result emitted as unavailable placeholder."
        )
        try:
            import llmlingua  # type: ignore # noqa: F401
        except Exception:
            warnings.append("llmlingua package is not installed in current environment.")
        status = "unavailable"
        return BenchmarkModeResult(
            mode=mode.value,
            status=status,
            metrics={},
            wall_time_s=0.0,
            compression_ratio=compression_ratio,
            settings=mode_settings,
            warnings=warnings,
        )
    elif mode == BaselineMode.EXACT_CACHE_ONLY_STUB:
        mode_settings = {"enable_compression": False, "enable_unification": False, "mock_responses": False}
        warnings.append(
            "Exact-match cache-only mode requires live vLLM/prefix caching and is a CPU stub here."
        )
        status = "stub"
        return BenchmarkModeResult(
            mode=mode.value,
            status=status,
            metrics={},
            wall_time_s=0.0,
            compression_ratio=compression_ratio,
            settings=mode_settings,
            warnings=warnings,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    t0 = time.perf_counter()
    metrics = run_benchmark(
        prompts,
        compression_ratio=0.0 if mode == BaselineMode.NO_COMPRESSION else compression_ratio,
        enable_compression=mode_settings["enable_compression"],
        enable_unification=mode_settings["enable_unification"],
        mock_responses=mode_settings["mock_responses"],
    )
    wall_time_s = time.perf_counter() - t0
    return BenchmarkModeResult(
        mode=mode.value,
        status=status,
        metrics=metrics.summary(),
        wall_time_s=wall_time_s,
        compression_ratio=compression_ratio,
        settings=mode_settings,
        warnings=warnings,
    )


def compare_baselines(
    prompts: list[str],
    compression_ratio: float = 0.40,
    modes: list[BaselineMode] | None = None,
) -> dict:
    """Compare baseline modes and return normalized artifact schema."""
    selected_modes = modes or [
        BaselineMode.NO_COMPRESSION,
        BaselineMode.COMPRESSION_ONLY,
        BaselineMode.PROMPTFUSE_FULL,
        BaselineMode.LLMLINGUA_ONLY,
        BaselineMode.EXACT_CACHE_ONLY_STUB,
    ]
    settings = Settings()
    config = settings.load()
    mode_results: dict[str, dict] = {}
    global_warnings: list[str] = []

    for mode in selected_modes:
        result = _run_mode(mode, prompts, compression_ratio)
        mode_results[mode.value] = asdict(result)
        if result.warnings:
            global_warnings.extend(result.warnings)

    compression_metrics = mode_results.get(BaselineMode.COMPRESSION_ONLY.value, {}).get("metrics", {})
    full_metrics = mode_results.get(BaselineMode.PROMPTFUSE_FULL.value, {}).get("metrics", {})
    baseline_hits = float(compression_metrics.get("unifier_hit_rate", 0.0))
    full_hits = float(full_metrics.get("unifier_hit_rate", 0.0))
    cache_hit_improvement = (
        full_hits / baseline_hits if baseline_hits > 0 else float("inf") if full_hits > 0 else 1.0
    )

    goals = {
        "token_reduction_target": 0.30,
        "rouge_l_target": config.evaluation.rouge_l_threshold,
        "latency_p99_target_ms": config.evaluation.latency_p99_ms,
        "meets_token_reduction": float(full_metrics.get("avg_token_reduction", 0.0)) >= 0.30,
        "meets_rouge_l": float(full_metrics.get("avg_rouge_l", 0.0))
        >= config.evaluation.rouge_l_threshold,
        "meets_latency": float(full_metrics.get("p99_latency_ms", float("inf")))
        <= config.evaluation.latency_p99_ms,
    }
    return {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compression_ratio": compression_ratio,
        "prompts_count": len(prompts),
        "modes": mode_results,
        "cache_hit_improvement": cache_hit_improvement,
        "goals": goals,
        "config_snapshot": config.model_dump(),
        "global_warnings": global_warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PromptFuse benchmark")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/long_prompts.txt"),
        help="Prompt workload (use long_prompts.txt for compression; synthetic JSON for unifier).",
    )
    parser.add_argument("--ratio", type=float, default=0.40)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Optional subset of baseline modes (space-separated).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.prompts.exists():
        logger.error("Prompt file not found: %s", args.prompts)
        raise SystemExit(1)

    prompts = load_prompts(args.prompts)
    logger.info("Loaded %d prompts", len(prompts))

    if args.compare_baselines:
        selected_modes = None
        if args.modes:
            selected_modes = [BaselineMode(mode) for mode in args.modes]
        results = compare_baselines(prompts, args.ratio, modes=selected_modes)
    else:
        metrics = run_benchmark(prompts, args.ratio)
        results = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "compression_ratio": args.ratio,
            "prompts_count": len(prompts),
            "mode": BaselineMode.PROMPTFUSE_FULL.value,
            "metrics": metrics.summary(),
            "warnings": [],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
