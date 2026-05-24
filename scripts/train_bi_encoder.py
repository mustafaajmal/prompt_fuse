#!/usr/bin/env python3
"""Fine-tune bi-encoder on prompt paraphrase pairs for semantic unification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_paraphrase_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path) as f:
        records = json.load(f)

    pairs: list[tuple[str, str]] = []
    clusters: dict[int, list[str]] = {}
    for r in records:
        clusters.setdefault(r["cluster_id"], []).append(r["text"])

    for texts in clusters.values():
        canonical = min(texts, key=len)
        for text in texts:
            if text != canonical:
                pairs.append((text, canonical))
            pairs.append((canonical, text))  # symmetric

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=Path("models/finetuned-minilm"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.data.exists():
        logger.error("Data file not found: %s. Run generate_synthetic_paraphrases.py first.", args.data)
        raise SystemExit(1)

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    pairs = load_paraphrase_pairs(args.data)
    logger.info("Loaded %d training pairs", len(pairs))

    model = SentenceTransformer(args.base_model)
    examples = [InputExample(texts=[a, b]) for a, b in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=100,
        show_progress_bar=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    logger.info("Saved fine-tuned encoder to %s", args.output)


if __name__ == "__main__":
    main()
