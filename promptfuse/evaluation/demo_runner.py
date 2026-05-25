"""CLI entry point for promptfuse-demo."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from promptfuse.evaluation.demo_experiment import print_summary_table, run_full_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptFuse demo A/B/C experiment")
    parser.add_argument("--workload", type=Path, default=Path("data/demo_workload.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--ratio", type=float, default=0.40)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument(
        "--warm-inventory",
        action="store_true",
        help="Use pre-built demo inventory (run warm_demo_inventory.py first)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["raw_vllm", "compress_only", "promptfuse_full", "pipeline_only"],
        default=None,
    )
    args = parser.parse_args()

    if not args.workload.exists():
        logger.error("Workload not found: %s", args.workload)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    comparison = run_full_experiment(
        args.workload,
        config_path=args.config,
        modes=args.modes,
        compression_ratio=args.ratio,
        vllm_base_url=args.vllm_url,
        max_tokens=args.max_tokens,
        use_vllm=not args.no_vllm,
        output_dir=args.output_dir,
        warm_inventory=args.warm_inventory,
    )
    print_summary_table(comparison)
    logger.info("Results saved to %s", comparison.get("_output_file"))


if __name__ == "__main__":
    main()
