#!/usr/bin/env python3
"""Sweep compression ratios and measure token reduction + ROUGE proxy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from promptfuse.evaluation.compression_eval import (
    evaluate_compression,
    load_prompts,
    print_compression_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
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
    if not prompts:
        logger.error("No prompts loaded")
        sys.exit(1)

    report = evaluate_compression(prompts, config_path=args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print_compression_summary(report)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
