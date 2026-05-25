#!/usr/bin/env python3
"""Evaluate semantic unifier hit rate on paraphrase clusters (GPU not required)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from promptfuse.evaluation.unifier_eval import evaluate_unifier, print_unifier_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptFuse unifier evaluation")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/unifier_eval.json"))
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        logger.error("Dataset not found: %s", args.data)
        sys.exit(1)

    report = evaluate_unifier(
        args.data,
        config_path=args.config,
        max_clusters=args.max_clusters,
        reset_inventory=not args.no_reset,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print_unifier_summary(report)
    logger.info("Saved to %s", args.output)

    if not report.get("passes_demo_threshold"):
        logger.warning("Unifier hit rate below 70%% demo threshold — try lowering similarity_threshold")
        sys.exit(2)


if __name__ == "__main__":
    main()
