#!/usr/bin/env python3
"""Aggregate CPU benchmark and tau-sweep outputs into summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _collect_benchmark_rows(benchmarks: list[dict], source_files: list[str]) -> list[dict]:
    rows: list[dict] = []
    for payload, source_file in zip(benchmarks, source_files, strict=False):
        ratio = payload.get("compression_ratio")
        if "modes" in payload:
            for mode_name, mode_data in payload["modes"].items():
                metrics = mode_data.get("metrics", {})
                rows.append(
                    {
                        "source": source_file,
                        "compression_ratio": ratio,
                        "mode": mode_name,
                        "status": mode_data.get("status"),
                        "avg_token_reduction": metrics.get("avg_token_reduction"),
                        "avg_rouge_l": metrics.get("avg_rouge_l"),
                        "p99_latency_ms": metrics.get("p99_latency_ms"),
                        "unifier_hit_rate": metrics.get("unifier_hit_rate"),
                        "total_requests": metrics.get("total_requests"),
                    }
                )
        else:
            metrics = payload.get("metrics", {})
            rows.append(
                {
                    "source": source_file,
                    "compression_ratio": ratio,
                    "mode": payload.get("mode", "single"),
                    "status": "ok",
                    "avg_token_reduction": metrics.get("avg_token_reduction"),
                    "avg_rouge_l": metrics.get("avg_rouge_l"),
                    "p99_latency_ms": metrics.get("p99_latency_ms"),
                    "unifier_hit_rate": metrics.get("unifier_hit_rate"),
                    "total_requests": metrics.get("total_requests"),
                }
            )
    return rows


def _collect_tau_rows(tau_payloads: list[dict], source_files: list[str]) -> list[dict]:
    rows: list[dict] = []
    for payload, source_file in zip(tau_payloads, source_files, strict=False):
        for item in payload.get("results", []):
            rows.append(
                {
                    "source": source_file,
                    "tau": item.get("tau"),
                    "cache_hit_rate": item.get("cache_hit_rate"),
                    "canonical_count": item.get("canonical_count"),
                    "merged_cluster_count": item.get("merged_cluster_count"),
                    "cluster_purity": item.get("cluster_purity"),
                    "total_prompts": item.get("total_prompts"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PromptFuse CPU artifacts.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/cpu_final/metrics"),
        help="Directory containing benchmark and tau_sweep JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/cpu_final/summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/cpu_final/summary.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.40, 0.55])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.75, 0.80, 0.85, 0.90])
    args = parser.parse_args()

    benchmark_paths = sorted(args.input_dir.glob("benchmark*.json"))
    tau_paths = sorted(args.input_dir.glob("tau_sweep*.json"))

    benchmarks = [_load_json(path) for path in benchmark_paths]
    tau_payloads = [_load_json(path) for path in tau_paths]

    benchmark_rows = _collect_benchmark_rows(benchmarks, [str(p) for p in benchmark_paths])
    tau_rows = _collect_tau_rows(tau_payloads, [str(p) for p in tau_paths])

    derived = {
        "benchmark_runs": len(benchmark_rows),
        "tau_points": len(tau_rows),
        "avg_token_reduction": (
            sum(r["avg_token_reduction"] for r in benchmark_rows if r["avg_token_reduction"] is not None)
            / max(1, sum(1 for r in benchmark_rows if r["avg_token_reduction"] is not None))
        ),
        "avg_unifier_hit_rate": (
            sum(r["unifier_hit_rate"] for r in benchmark_rows if r["unifier_hit_rate"] is not None)
            / max(1, sum(1 for r in benchmark_rows if r["unifier_hit_rate"] is not None))
        ),
        "best_tau_by_purity": max(tau_rows, key=lambda row: row["cluster_purity"], default=None),
    }

    summary = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "seed": args.seed,
            "config_path": str(args.config),
            "ratios": args.ratios,
            "taus": args.taus,
            "input_dir": str(args.input_dir),
        },
        "files": {
            "benchmark_files": [str(path) for path in benchmark_paths],
            "tau_files": [str(path) for path in tau_paths],
        },
        "benchmark_rows": benchmark_rows,
        "tau_rows": tau_rows,
        "derived_metrics": derived,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        fieldnames = [
            "row_type",
            "source",
            "compression_ratio",
            "mode",
            "status",
            "avg_token_reduction",
            "avg_rouge_l",
            "p99_latency_ms",
            "unifier_hit_rate",
            "total_requests",
            "tau",
            "cache_hit_rate",
            "canonical_count",
            "merged_cluster_count",
            "cluster_purity",
            "total_prompts",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in benchmark_rows:
            writer.writerow({"row_type": "benchmark", **row})
        for row in tau_rows:
            writer.writerow({"row_type": "tau_sweep", **row})

    print(json.dumps({"summary_json": str(args.output_json), "summary_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
