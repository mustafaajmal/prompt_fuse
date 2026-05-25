"""Tests for demo experiment (no GPU required)."""

from __future__ import annotations

from pathlib import Path

from promptfuse.evaluation.demo_experiment import load_demo_workload, summarize_mode
from promptfuse.evaluation.demo_experiment import RequestRecord


def test_load_demo_workload():
    path = Path("data/demo_workload.json")
    clusters = load_demo_workload(path)
    assert len(clusters) == 5
    assert len(clusters[0].prompts) >= 2


def test_summarize_mode():
    records = [
        RequestRecord(
            mode="promptfuse_full",
            cluster="summarize",
            paraphrase_index=0,
            run_index=1,
            raw_prompt="a",
            final_prompt="canonical",
            unifier_hit=False,
            unifier_similarity=None,
            token_reduction=0.3,
            original_tokens=10,
            final_tokens=7,
            pipeline_ms=5.0,
            vllm_latency_ms=100.0,
            vllm_prompt_tokens=7,
            vllm_completion_tokens=10,
        ),
        RequestRecord(
            mode="promptfuse_full",
            cluster="summarize",
            paraphrase_index=1,
            run_index=1,
            raw_prompt="b",
            final_prompt="canonical",
            unifier_hit=True,
            unifier_similarity=0.9,
            token_reduction=0.35,
            original_tokens=12,
            final_tokens=7,
            pipeline_ms=3.0,
            vllm_latency_ms=50.0,
            vllm_prompt_tokens=7,
            vllm_completion_tokens=10,
        ),
    ]
    summary = summarize_mode(records)
    assert summary.unifier_hit_rate == 0.5
    assert summary.unique_final_prompts_per_cluster == 1.0
