#!/usr/bin/env python3
"""Reset and warm the demo canonical inventory from demo_workload.json."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.unifier import SemanticUnifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=Path("data/demo_workload.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    args = parser.parse_args()

    settings = Settings(config_path=args.config)
    config = settings.load()
    inv_path = Path(config.unifier.inventory_path)

    if inv_path.exists():
        shutil.rmtree(inv_path)
        logger.info("Cleared inventory at %s", inv_path)

    with open(args.workload) as f:
        clusters = json.load(f)

    unifier = SemanticUnifier(config.unifier, lazy_load=False)
    count = 0
    for cluster in clusters:
        for text in cluster["prompts"]:
            tok = unifier.count_tokens_approx(text)
            unifier.unify(text, token_count=tok)
            count += 1

    unifier.save_inventory()
    logger.info("Warmed inventory with %d prompts → %d canonical entries", count, unifier.store.size)


if __name__ == "__main__":
    main()
