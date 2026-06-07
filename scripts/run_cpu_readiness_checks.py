#!/usr/bin/env python3
"""Run lightweight CPU-only checks for PromptFuse development workflow."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.generate_synthetic_paraphrases import expand_clusters
except ModuleNotFoundError:  # Support direct execution: python scripts/run_cpu_readiness_checks.py
    from generate_synthetic_paraphrases import expand_clusters


def check_synthetic_generator() -> dict:
    records = expand_clusters()
    clusters = {row["cluster_id"] for row in records}
    return {
        "name": "synthetic_generator",
        "ok": len(records) == 500 and len(clusters) == 50,
        "details": {"records": len(records), "clusters": len(clusters)},
    }


def check_config_defaults() -> dict:
    try:
        from promptfuse.config import Settings
    except ModuleNotFoundError as exc:
        return {
            "name": "default_config",
            "ok": False,
            "details": {"error": f"missing dependency: {exc.name}"},
        }

    config = Settings().load()
    ratios = config.evaluation.compression_ratios
    return {
        "name": "default_config",
        "ok": len(ratios) >= 3 and config.serving.port > 0 and config.serving.vllm_timeout_s > 0,
        "details": {
            "compression_ratios": ratios,
            "serving_port": config.serving.port,
            "timeout_s": config.serving.vllm_timeout_s,
        },
    }


def check_canonical_store_roundtrip() -> dict:
    try:
        from promptfuse.unifier.canonical_store import CanonicalStore
    except ModuleNotFoundError as exc:
        return {
            "name": "canonical_store_roundtrip",
            "ok": False,
            "details": {"error": f"missing dependency: {exc.name}"},
        }

    try:
        import numpy as np
    except ModuleNotFoundError:
        return {
            "name": "canonical_store_roundtrip",
            "ok": False,
            "details": {"error": "numpy is not installed in the current Python environment"},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        inventory = Path(tmpdir) / "inventory"
        store = CanonicalStore(embedding_dim=4, inventory_path=inventory)
        store.add("summarize this text", np.array([1.0, 0.0, 0.0, 0.0]), token_count=3)
        store.save()

        loaded = CanonicalStore(embedding_dim=4, inventory_path=inventory)
        match, similarity = loaded.search(np.array([0.99, 0.01, 0.0, 0.0]), threshold=0.8)

    return {
        "name": "canonical_store_roundtrip",
        "ok": match is not None and similarity >= 0.8,
        "details": {"similarity": round(similarity, 4), "match_text": match.text if match else None},
    }


def main() -> None:
    checks = [
        check_synthetic_generator(),
        check_config_defaults(),
        check_canonical_store_roundtrip(),
    ]
    all_ok = all(item["ok"] for item in checks)
    print(json.dumps({"ok": all_ok, "checks": checks}, indent=2))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
