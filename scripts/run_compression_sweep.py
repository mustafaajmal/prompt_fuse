#!/usr/bin/env python3
"""Sweep compression ratios (25%, 40%, 55%) on a workload."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.evaluation.compression_eval import run_compression_sweep, save_sweep_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression ratio sweep")
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("data/complex_workload.json"),
        help="Workload JSON (generate with generate_complex_workload.py if missing)",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/compression_sweep.json"))
    parser.add_argument("--lazy", action="store_true", help="Lazy-load proxy LM (slower first prompt)")
    args = parser.parse_args()

    if not args.workload.exists():
        logger.error(
            "Workload missing: %s — run: python scripts/generate_complex_workload.py",
            args.workload,
        )
        sys.exit(1)

    config = Settings(config_path=args.config).load()
    report = run_compression_sweep(
        args.workload,
        config=config,
        lazy_load=args.lazy,
    )
    out = save_sweep_report(report, args.output)

    print("\n=== Compression Sweep ===")
    for ratio, stats in report["ratios"].items():
        print(
            f"  ratio={ratio}: avg reduction={stats['avg_token_reduction']:.1%} "
            f"p99={stats['p99_pipeline_ms']:.0f}ms "
            f"≥30%={stats['meets_30pct_target']}"
        )
    print(f"Saved: {out}\n")


if __name__ == "__main__":
    main()
