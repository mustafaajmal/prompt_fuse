"""Evaluate semantic unifier precision on paraphrase clusters (no vLLM required)."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from promptfuse.config import PromptFuseConfig, Settings
from promptfuse.unifier import SemanticUnifier

logger = logging.getLogger(__name__)


@dataclass
class ClusterUnifierResult:
    cluster_id: int
    category: str
    variants: int
    first_miss: bool
    subsequent_hits: int
    subsequent_total: int
    canonical_text: str | None
    unified_forms: list[str]


def load_paraphrase_clusters(path: Path) -> dict[int, list[dict[str, Any]]]:
    with open(path) as f:
        records = json.load(f)
    clusters: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        clusters.setdefault(int(r["cluster_id"]), []).append(r)
    for cid in clusters:
        clusters[cid].sort(key=lambda x: int(x.get("variant_id", 0)))
    return clusters


def evaluate_unifier(
    data_path: Path,
    *,
    config: PromptFuseConfig | None = None,
    config_path: Path | None = None,
    max_clusters: int | None = None,
    reset_inventory: bool = True,
) -> dict[str, Any]:
    """
    Run each paraphrase cluster through a fresh unifier inventory.

    After the first variant establishes a canonical form, later variants in the
    same cluster should unify (cache hit) when the bi-encoder similarity >= τ.
    """
    import shutil

    settings = Settings(config_path=config_path) if config_path else Settings()
    cfg = config or settings.load()

    inv_path = Path(cfg.unifier.inventory_path)
    if reset_inventory and inv_path.exists():
        shutil.rmtree(inv_path)

    unifier = SemanticUnifier(cfg.unifier, lazy_load=False)
    clusters = load_paraphrase_clusters(data_path)
    cluster_ids = sorted(clusters.keys())
    if max_clusters is not None:
        cluster_ids = cluster_ids[:max_clusters]

    results: list[ClusterUnifierResult] = []
    total_hits = 0
    total_after_first = 0

    for cid in cluster_ids:
        variants = clusters[cid]
        texts = [v["text"] for v in variants]
        hits_after_first = 0
        unified: list[str] = []
        first_miss = False
        canonical: str | None = None

        for i, text in enumerate(texts):
            r = unifier.unify(text, token_count=unifier.count_tokens_approx(text))
            unified.append(r.unified)
            if i == 0:
                first_miss = not r.cache_hit
                canonical = r.unified
            else:
                total_after_first += 1
                if r.cache_hit:
                    hits_after_first += 1
                    total_hits += 1

        results.append(
            ClusterUnifierResult(
                cluster_id=cid,
                category=str(variants[0].get("category", "")),
                variants=len(texts),
                first_miss=first_miss,
                subsequent_hits=hits_after_first,
                subsequent_total=max(0, len(texts) - 1),
                canonical_text=canonical,
                unified_forms=unified,
            )
        )

    unifier.save_inventory()

    hit_rate = total_hits / total_after_first if total_after_first else 0.0
    clusters_with_full_hits = sum(
        1 for r in results if r.subsequent_total > 0 and r.subsequent_hits == r.subsequent_total
    )
    clusters_tested = sum(1 for r in results if r.subsequent_total > 0)

    return {
        "dataset": str(data_path),
        "similarity_threshold": cfg.unifier.similarity_threshold,
        "clusters_evaluated": len(results),
        "subsequent_hit_rate": round(hit_rate, 4),
        "clusters_fully_unified": clusters_with_full_hits,
        "clusters_with_variants": clusters_tested,
        "cluster_full_hit_rate": (
            round(clusters_with_full_hits / clusters_tested, 4) if clusters_tested else 0.0
        ),
        "inventory_size": unifier.store.size,
        "per_cluster": [asdict(r) for r in results],
        "passes_demo_threshold": hit_rate >= 0.70,
    }


def print_unifier_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Semantic Unifier Evaluation")
    print("=" * 60)
    print(f"  Clusters evaluated:     {report['clusters_evaluated']}")
    print(f"  Subsequent hit rate:    {report['subsequent_hit_rate']:.1%}")
    print(f"  Full-cluster hit rate:  {report['cluster_full_hit_rate']:.1%}")
    print(f"  Inventory size:         {report['inventory_size']}")
    print(f"  Similarity threshold τ: {report['similarity_threshold']}")
    print(f"  Passes demo (≥70%):     {report['passes_demo_threshold']}")
    print("=" * 60 + "\n")


def main_cli() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="PromptFuse unifier evaluation")
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/unifier_eval.json"))
    parser.add_argument("--max-clusters", type=int, default=None)
    args = parser.parse_args()

    if not args.data.exists():
        logger.error("Dataset not found: %s", args.data)
        sys.exit(1)

    report = evaluate_unifier(
        args.data,
        config_path=args.config,
        max_clusters=args.max_clusters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print_unifier_summary(report)
    if not report.get("passes_demo_threshold"):
        sys.exit(2)
