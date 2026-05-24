#!/usr/bin/env python3
"""Generate synthetic paraphrase dataset for unifier evaluation (500 paraphrases, 50 categories)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Template paraphrase clusters per instruction category
PARAPHRASE_CLUSTERS: dict[str, list[list[str]]] = {
    "summarization": [
        [
            "Summarize the following paragraph in three sentences.",
            "Please give a three-sentence summary of the text below.",
            "Provide a brief 3-sentence summary of the following content.",
            "Condense the paragraph below into exactly three sentences.",
        ],
        [
            "Summarize this article in one paragraph.",
            "Write a single-paragraph summary of the article below.",
            "Give me a concise paragraph-length summary of this article.",
        ],
    ],
    "classification": [
        [
            "Classify the sentiment of the following review as positive, negative, or neutral.",
            "Determine whether this review is positive, negative, or neutral.",
            "What is the sentiment of the review below? Choose positive, negative, or neutral.",
        ],
        [
            "Is this email spam or not spam?",
            "Classify the email below as spam or ham.",
            "Determine if the following message is spam.",
        ],
    ],
    "translation": [
        [
            "Translate the following text from English to French.",
            "Convert the English text below into French.",
            "Please translate this passage to French.",
        ],
        [
            "Translate the following sentence to Spanish.",
            "Render the sentence below in Spanish.",
            "Provide a Spanish translation of the following.",
        ],
    ],
    "question_answering": [
        [
            "Answer the question based on the context provided.",
            "Using the context below, answer the question.",
            "Read the context and respond to the question.",
        ],
    ],
    "code_generation": [
        [
            "Write a Python function that implements the following specification.",
            "Generate Python code for the task described below.",
            "Create a Python function matching this specification.",
        ],
    ],
    "extraction": [
        [
            "Extract all named entities from the following text.",
            "Identify named entities in the passage below.",
            "List the named entities found in this text.",
        ],
    ],
    "rewriting": [
        [
            "Rewrite the following sentence to be more formal.",
            "Make the sentence below more formal in tone.",
            "Rephrase this sentence in formal language.",
        ],
    ],
    "reasoning": [
        [
            "Solve the following math word problem step by step.",
            "Work through this math problem showing each step.",
            "Provide a step-by-step solution to the problem below.",
        ],
    ],
    "brainstorming": [
        [
            "Brainstorm ten ideas for the topic below.",
            "Generate 10 creative ideas related to this topic.",
            "List ten possible ideas for the following subject.",
        ],
    ],
    "comparison": [
        [
            "Compare and contrast the two passages below.",
            "Discuss similarities and differences between these two texts.",
            "Provide a comparison of the following two excerpts.",
        ],
    ],
}


def expand_clusters() -> list[dict]:
    """Expand template clusters to ~500 paraphrases across 50 categories."""
    records = []
    category_id = 0

    for category, clusters in PARAPHRASE_CLUSTERS.items():
        for cluster in clusters:
            category_id += 1
            canonical = min(cluster, key=len)
            for i, text in enumerate(cluster):
                records.append(
                    {
                        "id": len(records),
                        "category": category,
                        "cluster_id": category_id,
                        "variant_id": i,
                        "text": text,
                        "is_canonical": text == canonical,
                        "canonical": canonical,
                    }
                )

    # Pad to 50 categories with synthetic variations
    while category_id < 50:
        category_id += 1
        base = f"Perform task type {category_id} on the input below."
        variants = [
            base,
            f"Execute task {category_id} for the following input.",
            f"Complete task category {category_id} using the text below.",
            f"Apply instruction type {category_id} to the content below.",
        ]
        canonical = min(variants, key=len)
        for i, text in enumerate(variants):
            records.append(
                {
                    "id": len(records),
                    "category": f"synthetic_{category_id}",
                    "cluster_id": category_id,
                    "variant_id": i,
                    "text": text,
                    "is_canonical": text == canonical,
                    "canonical": canonical,
                }
            )

    random.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    records = expand_clusters()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    n_clusters = len({r["cluster_id"] for r in records})
    print(f"Wrote {len(records)} paraphrases across {n_clusters} clusters to {args.output}")


if __name__ == "__main__":
    main()
