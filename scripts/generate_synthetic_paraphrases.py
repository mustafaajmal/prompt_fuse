#!/usr/bin/env python3
"""Generate synthetic paraphrase dataset for unifier evaluation (500 paraphrases, 50 categories)."""

from __future__ import annotations

import argparse
import json
import random
import re
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

_FALLBACK_TASKS = [
    ("policy_qa", "Answer compliance questions"),
    ("incident_summary", "Summarize an incident report"),
    ("action_items", "Extract action items"),
    ("risk_assessment", "Assess the main risks"),
    ("meeting_minutes", "Turn notes into meeting minutes"),
    ("requirements", "Convert the text into software requirements"),
    ("bug_triage", "Classify bug severity"),
    ("sql_generation", "Generate an SQL query"),
    ("test_case_generation", "Write test cases"),
    ("data_extraction", "Extract structured fields"),
]

_FALLBACK_DOMAINS = [
    ("finance", "financial"),
    ("healthcare", "healthcare"),
    ("legal", "legal"),
    ("education", "education"),
    ("ecommerce", "e-commerce"),
    ("cybersecurity", "cybersecurity"),
    ("operations", "operations"),
    ("research", "research"),
]


_REWRITE_PATTERNS = [
    "Please {text}",
    "{text} Keep the response concise.",
    "{text} Return only the final answer.",
    "Task: {text}",
    "Instruction: {text}",
    "Could you {text_lc}",
    "For the text below, {text_lc}",
    "{text} Use clear language.",
    "Kindly {text_lc}",
    "{text} Limit the response to key points.",
]

_CONTEXT_SNIPPETS = [
    "The team is preparing a weekly update for leadership and needs a concise, accurate response.",
    "The input may contain noisy details, but the final answer should focus on the main objective only.",
    "Assume this output will be consumed by a downstream automation step that is sensitive to formatting.",
    "The requester is under time pressure and prefers direct, actionable language over long explanations.",
    "The content may include domain-specific terminology; preserve meaning even when simplifying.",
]

_CONSTRAINT_SNIPPETS = [
    "Return the final answer only.",
    "Do not include preamble or extra commentary.",
    "Follow the requested format exactly.",
    "Keep the output concise and unambiguous.",
    "Preserve key entities and numeric values where present.",
]

_WRAPPER_PATTERNS = [
    "Context: {context}\nInstruction: {text}\nConstraints:\n- {c1}\n- {c2}",
    "You are an assistant helping with a production workflow.\nTask: {text}\nRequirements:\n1. {c1}\n2. {c2}\nBackground: {context}",
    "System note: prioritize precision over verbosity.\nUser request: {text}\nAdditional notes: {context}\nOutput rules: {c1} {c2}",
]


def _normalize_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _lowercase_first_alpha(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.lower()
            break
    return "".join(chars)


def _expand_cluster_variants(base_variants: list[str], target_variants: int) -> list[str]:
    """Expand each semantic cluster with light paraphrastic templates."""
    unique = []
    seen = set()

    def add(text: str) -> None:
        norm = _normalize_sentence(text)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)

    for variant in base_variants:
        add(variant)

    pattern_idx = 0
    source_idx = 0
    while len(unique) < target_variants:
        source = unique[source_idx % len(unique)]
        rewritten = _REWRITE_PATTERNS[pattern_idx % len(_REWRITE_PATTERNS)].format(
            text=source.rstrip(".!?"),
            text_lc=_lowercase_first_alpha(source.rstrip(".!?")),
        )
        add(rewritten)

        if len(unique) < target_variants:
            wrapper = _WRAPPER_PATTERNS[pattern_idx % len(_WRAPPER_PATTERNS)]
            context = _CONTEXT_SNIPPETS[pattern_idx % len(_CONTEXT_SNIPPETS)]
            c1 = _CONSTRAINT_SNIPPETS[pattern_idx % len(_CONSTRAINT_SNIPPETS)]
            c2 = _CONSTRAINT_SNIPPETS[(pattern_idx + 1) % len(_CONSTRAINT_SNIPPETS)]
            enriched = wrapper.format(
                text=source.rstrip(".!?"),
                context=context,
                c1=c1,
                c2=c2,
            )
            add(enriched)

        pattern_idx += 1
        source_idx += 1

    return unique[:target_variants]


def _build_realistic_fallback(cluster_id: int) -> tuple[str, list[str]]:
    """Create realistic fallback paraphrase clusters instead of generic task IDs."""
    task_slug, task_text = _FALLBACK_TASKS[(cluster_id - 1) % len(_FALLBACK_TASKS)]
    domain_slug, domain_text = _FALLBACK_DOMAINS[((cluster_id - 1) // len(_FALLBACK_TASKS)) % len(_FALLBACK_DOMAINS)]
    category = f"{task_slug}_{domain_slug}"
    context_phrase = f"the {domain_text} text below"

    variants = [
        f"{task_text} using {context_phrase}.",
        f"Please {task_text.lower()} from {context_phrase}.",
        f"For {context_phrase}, {task_text.lower()}.",
        f"{task_text} based on {context_phrase}.",
    ]
    return category, variants


def _build_seed_clusters() -> list[tuple[str, list[str]]]:
    clusters: list[tuple[str, list[str]]] = []
    for category, grouped in PARAPHRASE_CLUSTERS.items():
        for cluster in grouped:
            clusters.append((category, cluster))

    while len(clusters) < 50:
        cluster_id = len(clusters) + 1
        category, variants = _build_realistic_fallback(cluster_id)
        clusters.append((category, variants))
    return clusters[:50]


def expand_clusters(target_size: int = 500, n_clusters: int = 50) -> list[dict]:
    """Expand template clusters to target size (default: 500 across 50 clusters)."""
    n_clusters = max(1, n_clusters)
    target_size = max(target_size, n_clusters)
    per_cluster = target_size // n_clusters
    remainder = target_size % n_clusters

    seed_clusters = _build_seed_clusters()[:n_clusters]
    records: list[dict] = []

    for idx, (category, base_variants) in enumerate(seed_clusters, start=1):
        target_variants = per_cluster + (1 if idx <= remainder else 0)
        expanded = _expand_cluster_variants(base_variants, max(target_variants, len(base_variants)))
        canonical = min(expanded, key=lambda text: len(text.split()))
        for variant_id, text in enumerate(expanded[:target_variants]):
            records.append(
                {
                    "id": len(records),
                    "category": category,
                    "cluster_id": idx,
                    "variant_id": variant_id,
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
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--clusters", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    records = expand_clusters(target_size=args.target_size, n_clusters=args.clusters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    n_clusters = len({r["cluster_id"] for r in records})
    print(f"Wrote {len(records)} paraphrases across {n_clusters} clusters to {args.output}")


if __name__ == "__main__":
    main()
