#!/usr/bin/env python3
"""Download and prepare evaluation datasets (ShareGPT, LMSYS-Chat-1M samples)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sample_sharegpt(output: Path, max_samples: int = 1000) -> None:
    from datasets import load_dataset

    logger.info("Loading ShareGPT sample...")
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
    prompts = []
    for row in ds:
        conv = row.get("conversations", [])
        for turn in conv:
            if turn.get("from") == "human":
                prompts.append(turn.get("value", ""))
                break
        if len(prompts) >= max_samples:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info("Wrote %d ShareGPT prompts to %s", len(prompts), output)


def sample_lmsys(output: Path, max_samples: int = 1000) -> None:
    from datasets import load_dataset

    logger.info("Loading LMSYS-Chat-1M sample...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    prompts = []
    for row in ds:
        messages = row.get("conversation", [])
        for msg in messages:
            if msg.get("role") == "user":
                prompts.append(msg.get("content", ""))
                break
        if len(prompts) >= max_samples:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info("Wrote %d LMSYS prompts to %s", len(prompts), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sharegpt", "lmsys", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.dataset in ("sharegpt", "all"):
        try:
            sample_sharegpt(args.output_dir / "sharegpt_prompts.json", args.max_samples)
        except Exception as exc:
            logger.warning("ShareGPT download failed: %s", exc)

    if args.dataset in ("lmsys", "all"):
        try:
            sample_lmsys(args.output_dir / "lmsys_prompts.json", args.max_samples)
        except Exception as exc:
            logger.warning("LMSYS download failed: %s", exc)


if __name__ == "__main__":
    main()
