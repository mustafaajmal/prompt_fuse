#!/usr/bin/env python3
"""Build canonical prompt inventory from historical prompts (warmup phase)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from promptfuse.config import Settings
from promptfuse.pipeline import PromptFusePipeline

logger = logging.getLogger(__name__)


def load_prompts(path: Path) -> list[str]:
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [d.get("text", d.get("prompt", "")) for d in data]
            return data
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    settings = Settings(config_path=args.config)
    config = settings.load()
    pipeline = PromptFusePipeline(config, lazy_load=False)

    prompts = load_prompts(args.prompts)
    logger.info("Warming up inventory with %d prompts (compress → unify)", len(prompts))

    for p in prompts:
        pipeline.process(p)

    out = args.output or Path(config.unifier.inventory_path)
    if pipeline.unifier:
        pipeline.unifier.store.save(out)
        logger.info("Canonical inventory size: %d", pipeline.unifier.store.size)
    else:
        logger.error("Unifier disabled in config")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
