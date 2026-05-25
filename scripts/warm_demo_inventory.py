#!/usr/bin/env python3
"""Reset and warm the demo canonical inventory from demo_workload.json."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.pipeline import PromptFusePipeline
from promptfuse.unifier import SemanticUnifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=Path("data/demo_workload.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Unifier-only warmup (no Llama proxy LM; use before HF_TOKEN is set)",
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="compress → unify like production (requires HF access for Llama-3.2-1B)",
    )
    args = parser.parse_args()

    settings = Settings(config_path=args.config)
    config = settings.load()
    inv_path = Path(config.unifier.inventory_path)

    if inv_path.exists():
        shutil.rmtree(inv_path)
        logger.info("Cleared inventory at %s", inv_path)

    with open(args.workload) as f:
        clusters = json.load(f)

    use_compress = args.full_pipeline or not args.no_compress

    count = 0
    if use_compress:
        logger.info("Warming with full pipeline (compress → unify)...")
        pipeline = PromptFusePipeline(config, lazy_load=False)
        if not pipeline.unifier:
            logger.error("Unifier disabled in config")
            raise SystemExit(1)
        for cluster in clusters:
            for text in cluster["prompts"]:
                pipeline.process(text)
                count += 1
        pipeline.unifier.save_inventory()
        size = pipeline.unifier.store.size
    else:
        logger.info("Warming unifier only (no compression; no HF token required)...")
        unifier = SemanticUnifier(config.unifier, lazy_load=False)
        for cluster in clusters:
            for text in cluster["prompts"]:
                tok = unifier.count_tokens_approx(text)
                unifier.unify(text, token_count=tok)
                count += 1
        unifier.save_inventory()
        size = unifier.store.size

    logger.info("Warmed inventory with %d prompts → %d canonical entries", count, size)


if __name__ == "__main__":
    main()
