#!/usr/bin/env python3
"""Run the end-to-end CPU artifact pipeline for PromptFuse."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StepResult:
    name: str
    command: list[str]
    ok: bool
    return_code: int
    log_file: str
    error: str | None = None


def run_step(name: str, command: list[str], log_file: Path, required: bool = True) -> StepResult:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    ok = process.returncode == 0
    error = None if ok else f"Step '{name}' failed with exit code {process.returncode}."
    if not ok and required:
        error = (
            f"{error} Review log: {log_file}. Install missing deps and retry. "
            "This step is required for a complete CPU artifact bundle."
        )
    return StepResult(
        name=name,
        command=command,
        ok=ok,
        return_code=process.returncode,
        log_file=str(log_file),
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PromptFuse CPU-only pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cpu_final"))
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/long_prompts.txt"),
        help="Multi-sentence workload for compression benchmarks (not single-line paraphrases).",
    )
    parser.add_argument("--synthetic-output", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.40, 0.55])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.75, 0.80, 0.85, 0.90])
    args = parser.parse_args()

    metrics_dir = args.output_dir / "metrics"
    logs_dir = args.output_dir / "logs"
    notes_dir = args.output_dir / "notes"
    configs_dir = args.output_dir / "configs"
    for directory in (metrics_dir, logs_dir, notes_dir, configs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if Path("configs/default.yaml").exists():
        shutil.copy2("configs/default.yaml", configs_dir / "final_cpu.yaml")

    steps: list[StepResult] = []
    python = sys.executable

    steps.append(
        run_step(
            "readiness_checks",
            [python, "scripts/run_cpu_readiness_checks.py"],
            logs_dir / "readiness_checks.log",
            required=False,
        )
    )

    steps.append(
        run_step(
            "generate_synthetic",
            [
                python,
                "scripts/generate_synthetic_paraphrases.py",
                "--target-size",
                "500",
                "--clusters",
                "50",
                "--seed",
                str(args.seed),
                "--output",
                str(args.synthetic_output),
            ],
            logs_dir / "generate_synthetic.log",
            required=True,
        )
    )

    for ratio in args.ratios:
        ratio_name = str(ratio).replace(".", "_")
        steps.append(
            run_step(
                f"benchmark_ratio_{ratio_name}",
                [
                    python,
                    "-m",
                    "promptfuse.evaluation.benchmark",
                    "--prompts",
                    str(args.prompts),
                    "--ratio",
                    str(ratio),
                    "--compare-baselines",
                    "--output",
                    str(metrics_dir / f"benchmark_ratio_{ratio_name}.json"),
                ],
                logs_dir / f"benchmark_ratio_{ratio_name}.log",
                required=False,
            )
        )

    steps.append(
        run_step(
            "tau_sweep",
            [
                python,
                "scripts/tau_sweep.py",
                "--prompts",
                str(args.synthetic_output),
                "--taus",
                *[str(tau) for tau in args.taus],
                "--output",
                str(metrics_dir / "tau_sweep.json"),
            ],
            logs_dir / "tau_sweep.log",
            required=False,
        )
    )

    steps.append(
        run_step(
            "aggregate_results",
            [
                python,
                "scripts/aggregate_results.py",
                "--input-dir",
                str(metrics_dir),
                "--output-json",
                str(args.output_dir / "summary.json"),
                "--output-csv",
                str(args.output_dir / "summary.csv"),
                "--seed",
                str(args.seed),
                "--ratios",
                *[str(r) for r in args.ratios],
                "--taus",
                *[str(t) for t in args.taus],
            ],
            logs_dir / "aggregate_results.log",
            required=False,
        )
    )

    report = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "required_failures": [asdict(s) for s in steps if not s.ok and "required" in (s.error or "")],
        "steps": [asdict(step) for step in steps],
    }
    with open(notes_dir / "pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    required_failed = any(step.error and "required" in step.error for step in steps)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if required_failed else 0)


if __name__ == "__main__":
    main()
