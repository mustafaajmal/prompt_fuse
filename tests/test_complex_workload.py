"""Tests for complex workload and unifier eval helpers (no GPU)."""

from __future__ import annotations

from pathlib import Path

from promptfuse.evaluation.demo_experiment import load_demo_workload
from promptfuse.evaluation.unifier_eval import load_paraphrase_clusters


def test_complex_workload_loads():
    path = Path("data/complex_workload.json")
    clusters = load_demo_workload(path)
    assert len(clusters) == 8
    assert all(len(c.prompts) >= 3 for c in clusters)
    assert any("agent" in c.cluster for c in clusters)


def test_paraphrase_clusters_load():
    path = Path("data/synthetic_paraphrases.json")
    clusters = load_paraphrase_clusters(path)
    assert len(clusters) >= 50
    assert all(len(v) >= 2 for v in clusters.values())
