#!/usr/bin/env python3
"""Sweep unifier similarity thresholds and emit CPU-friendly diagnostics."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promptfuse.config import Settings, UnifierConfig
from promptfuse.unifier import SemanticUnifier


@dataclass
class TauSweepResult:
    tau: float
    total_prompts: int
    cache_hits: int
    cache_hit_rate: float
    canonical_count: int
    merged_cluster_count: int
    cluster_purity: float


def load_prompts(path: Path) -> list[tuple[str, int]]:
    """Load prompts with optional cluster IDs for merge diagnostics."""
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            prompts = []
            for idx, row in enumerate(data):
                text = row.get("text", row.get("prompt", "")).strip()
                if not text:
                    continue
                cluster_id = int(row.get("cluster_id", idx))
                prompts.append((text, cluster_id))
            return prompts
        if isinstance(data, list):
            return [(str(item).strip(), idx) for idx, item in enumerate(data) if str(item).strip()]
        prompt_list = data.get("prompts", [])
        return [(str(item).strip(), idx) for idx, item in enumerate(prompt_list) if str(item).strip()]

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return [(line, idx) for idx, line in enumerate(lines)]


def run_tau(prompts: list[tuple[str, int]], base_cfg: UnifierConfig, tau: float) -> TauSweepResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = base_cfg.model_copy(deep=True)
        cfg.similarity_threshold = tau
        cfg.inventory_path = str(Path(tmpdir) / "inventory")
        unifier = SemanticUnifier(cfg, lazy_load=True)

        cache_hits = 0
        canonical_to_clusters: dict[int, set[int]] = defaultdict(set)
        for text, cluster_id in prompts:
            result = unifier.unify(text, token_count=unifier.count_tokens_approx(text))
            if result.cache_hit:
                cache_hits += 1
            if result.canonical_id is not None:
                canonical_to_clusters[result.canonical_id].add(cluster_id)

        total = len(prompts)
        merged_cluster_count = sum(1 for clusters in canonical_to_clusters.values() if len(clusters) > 1)
        purity_values = [1.0 / len(clusters) for clusters in canonical_to_clusters.values() if clusters]
        cluster_purity = sum(purity_values) / len(purity_values) if purity_values else 1.0
        return TauSweepResult(
            tau=tau,
            total_prompts=total,
            cache_hits=cache_hits,
            cache_hit_rate=(cache_hits / total) if total else 0.0,
            canonical_count=len(canonical_to_clusters),
            merged_cluster_count=merged_cluster_count,
            cluster_purity=cluster_purity,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep similarity threshold tau for unifier behavior.")
    parser.add_argument("--prompts", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.75, 0.80, 0.85, 0.90],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/cpu_final/metrics/tau_sweep.json"))
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    settings = Settings(config_path=args.config)
    base_cfg = settings.load().unifier

    results = [asdict(run_tau(prompts, base_cfg, tau)) for tau in args.taus]
    payload = {
        "schema_version": "1.0",
        "prompts_path": str(args.prompts),
        "prompts_count": len(prompts),
        "taus": args.taus,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
