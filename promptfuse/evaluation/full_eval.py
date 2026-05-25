"""CLI entry point for promptfuse-eval (full local evaluation suite)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.evaluation.compression_eval import run_compression_sweep, save_sweep_report
from promptfuse.evaluation.demo_experiment import print_summary_table, run_full_experiment
from promptfuse.evaluation.vllm_client import VLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptFuse full evaluation suite")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--demo-workload", type=Path, default=Path("data/demo_workload.json"))
    parser.add_argument("--complex-workload", type=Path, default=Path("data/complex_workload.json"))
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--with-quality", action="store_true", help="ROUGE eval if vLLM is up")
    parser.add_argument("--quality-limit", type=int, default=5)
    args = parser.parse_args()

    if not args.complex_workload.exists():
        _run_script("generate_complex_workload.py", "--output", str(args.complex_workload))

    if not args.skip_warmup:
        _run_script(
            "warm_demo_inventory.py",
            "--config",
            str(args.config),
            "--workload",
            str(args.demo_workload),
        )

    config = Settings(config_path=args.config).load()
    sweep = run_compression_sweep(args.complex_workload, config=config)
    sweep_path = save_sweep_report(sweep, ROOT / "results" / "compression_sweep.json")
    logger.info("Compression sweep → %s", sweep_path)

    client = VLLMClient(base_url=config.serving.vllm_base_url, model=config.serving.vllm_model)
    use_vllm = client.health_check()
    if not use_vllm:
        logger.warning("vLLM not reachable — demo experiment runs pipeline-only metrics")

    comparison = run_full_experiment(
        args.demo_workload,
        config_path=args.config,
        use_vllm=use_vllm,
        output_dir=ROOT / "results",
    )
    print_summary_table(comparison)

    complex_comparison = run_full_experiment(
        args.complex_workload,
        config_path=args.config,
        use_vllm=use_vllm,
        output_dir=ROOT / "results",
    )
    complex_out = ROOT / "results" / "complex_demo_metrics.json"
    complex_comparison.pop("_output_file", None)
    with open(complex_out, "w") as f:
        json.dump(complex_comparison, f, indent=2)
    logger.info("Complex workload metrics → %s", complex_out)
    print_summary_table(complex_comparison)

    if args.with_quality and use_vllm:
        _run_script(
            "run_quality_eval.py",
            "--config",
            str(args.config),
            "--prompts",
            str(args.demo_workload),
            "--limit",
            str(args.quality_limit),
        )

    print("\n=== Evaluation complete ===")
    print(f"  {sweep_path}")
    print(f"  {comparison.get('_output_file')}")
    print(f"  {complex_out}")


if __name__ == "__main__":
    main()
