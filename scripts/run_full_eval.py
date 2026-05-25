#!/usr/bin/env python3
"""
Run the full PromptFuse evaluation suite for the class demo.

Stages (skip with flags):
  1. Unifier paraphrase eval (CPU, MiniLM only)
  2. Compression sweep (GPU, Llama-3.2-1B proxy)
  3. Pipeline A/B/C on demo workload (--no-vllm if vLLM down)
  4. Optional: ROUGE quality via vLLM (--with-quality)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> int:
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptFuse full evaluation suite")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-unifier", action="store_true")
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--no-vllm", action="store_true", help="Pipeline-only demo experiment")
    parser.add_argument("--with-quality", action="store_true", help="ROUGE via vLLM (needs server)")
    parser.add_argument("--with-complex", action="store_true", help="Also run demo on complex workload")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    root = Path(__file__).resolve().parent.parent
    rc = 0

    if not args.skip_unifier:
        rc |= _run(
            [
                py,
                str(root / "scripts/run_unifier_eval.py"),
                "--config",
                str(args.config),
                "--output",
                str(args.output_dir / "unifier_eval.json"),
            ]
        )

    if not args.skip_compression:
        rc |= _run(
            [
                py,
                str(root / "scripts/run_compression_eval.py"),
                "--config",
                str(args.config),
                "--prompts",
                str(root / "data/complex_workload.json"),
                "--output",
                str(args.output_dir / "compression_eval.json"),
            ]
        )

    if not args.skip_demo:
        demo_cmd = [
            py,
            str(root / "scripts/run_demo_experiment.py"),
            "--config",
            str(args.config),
            "--output-dir",
            str(args.output_dir),
            "--workload",
            str(root / "data/demo_workload.json"),
        ]
        if args.no_vllm:
            demo_cmd.append("--no-vllm")
        rc |= _run(demo_cmd)

        if args.with_complex:
            complex_cmd = demo_cmd.copy()
            complex_cmd[complex_cmd.index(str(root / "data/demo_workload.json"))] = str(
                root / "data/complex_workload.json"
            )
            complex_cmd += ["--output-dir", str(args.output_dir / "complex")]
            rc |= _run(complex_cmd)

    if args.with_quality:
        rc |= _run(
            [
                py,
                str(root / "scripts/run_quality_eval.py"),
                "--config",
                str(args.config),
                "--prompts",
                str(root / "data/demo_workload.json"),
                "--output",
                str(args.output_dir / "quality_eval.json"),
                "--limit",
                "8",
            ]
        )

    summary_path = args.output_dir / "full_eval_summary.json"
    summary = {
        "config": str(args.config),
        "artifacts": {
            "unifier": str(args.output_dir / "unifier_eval.json"),
            "compression": str(args.output_dir / "compression_eval.json"),
            "demo": str(args.output_dir / "demo_metrics.json"),
            "quality": str(args.output_dir / "quality_eval.json"),
            "complex_demo": str(args.output_dir / "complex/demo_metrics.json"),
        },
        "exit_code": rc,
    }
    for key, path in list(summary["artifacts"].items()):
        if not Path(path).exists():
            summary["artifacts"][key] = None

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Full eval summary: %s (exit=%d)", summary_path, rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
