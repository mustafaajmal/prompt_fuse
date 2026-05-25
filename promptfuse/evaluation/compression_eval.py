"""Evaluate segment compressor token reduction across compression ratios."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from promptfuse.config import PromptFuseConfig, Settings
from promptfuse.evaluation.metrics import compute_rouge_l, percentile
from promptfuse.pipeline import PromptFusePipeline

logger = logging.getLogger(__name__)


def load_prompts(path: Path) -> list[str]:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                if "prompts" in data[0]:
                    out: list[str] = []
                    for cluster in data:
                        out.extend(cluster["prompts"])
                    return out
                return [d.get("text", d.get("prompt", "")) for d in data]
            return [str(x) for x in data]
        return data.get("prompts", [])
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def evaluate_compression(
    prompts: list[str],
    *,
    ratios: list[float] | None = None,
    config: PromptFuseConfig | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    settings = Settings(config_path=config_path) if config_path else Settings()
    cfg = config or settings.load()
    ratios = ratios or cfg.evaluation.compression_ratios

    pipeline = PromptFusePipeline(
        cfg,
        lazy_load=False,
        enable_compression=True,
        enable_unification=False,
    )

    by_ratio: dict[str, Any] = {}
    for ratio in ratios:
        reductions: list[float] = []
        rouge_scores: list[float] = []
        latencies: list[float] = []

        for prompt in prompts:
            result = pipeline.process(prompt, compression_ratio=ratio)
            reductions.append(result.token_reduction)
            latencies.append(result.total_ms)
            if result.compression:
                rouge_scores.append(compute_rouge_l(result.raw_prompt, result.final_prompt))

        by_ratio[str(ratio)] = {
            "avg_token_reduction": round(statistics.mean(reductions), 4) if reductions else 0.0,
            "min_token_reduction": round(min(reductions), 4) if reductions else 0.0,
            "avg_rouge_l_proxy": round(statistics.mean(rouge_scores), 4) if rouge_scores else 0.0,
            "p99_latency_ms": round(percentile(latencies, 0.99), 2),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "meets_30pct_reduction": statistics.mean(reductions) >= 0.30 if reductions else False,
            "meets_rouge_threshold": (
                statistics.mean(rouge_scores) >= cfg.evaluation.rouge_l_threshold
                if rouge_scores
                else False
            ),
        }

    target_ratio = str(cfg.compressor.compression_ratio)
    target_stats = by_ratio.get(target_ratio, {})
    return {
        "num_prompts": len(prompts),
        "proxy_model": cfg.compressor.proxy_model,
        "by_ratio": by_ratio,
        "target_ratio": cfg.compressor.compression_ratio,
        "meets_token_goal": target_stats.get("meets_30pct_reduction", False),
        "meets_rouge_proxy_goal": target_stats.get("meets_rouge_threshold", False),
        "meets_latency_goal": target_stats.get("p99_latency_ms", 999) <= cfg.evaluation.latency_p99_ms,
    }


def print_compression_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("Segment Compressor Evaluation")
    print("=" * 68)
    print(f"{'Ratio':>8} {'Tok↓ avg':>10} {'ROUGE-L':>10} {'p99 ms':>10}")
    print("-" * 68)
    for ratio, stats in report.get("by_ratio", {}).items():
        print(
            f"{float(ratio):>7.0%} "
            f"{stats['avg_token_reduction']:>9.1%} "
            f"{stats['avg_rouge_l_proxy']:>9.3f} "
            f"{stats['p99_latency_ms']:>9.1f}"
        )
    print("-" * 68)
    print(f"  ≥30% reduction @ target: {report.get('meets_token_goal')}")
    print(f"  ROUGE proxy @ target:    {report.get('meets_rouge_proxy_goal')}")
    print("=" * 68 + "\n")


def main_cli() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="PromptFuse compression evaluation")
    parser.add_argument("--prompts", type=Path, default=Path("data/complex_workload.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/compression_eval.json"))
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    if not args.prompts.exists():
        logger.error("Prompt file not found: %s", args.prompts)
        sys.exit(1)

    prompts = load_prompts(args.prompts)[: args.limit]
    report = evaluate_compression(prompts, config_path=args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print_compression_summary(report)
