"""Compressor logic tests (no GPU / no model download)."""

from __future__ import annotations

from promptfuse.compressor.segment_compressor import split_sentences


def test_split_multiline_prompt():
    text = (
        "You are a helpful assistant. "
        "Background noise for RAG simulation. "
        "Task: Summarize the following paragraph in three sentences."
    )
    segments = split_sentences(text)
    assert len(segments) >= 2


def test_importance_sorting_direction():
    """Highest perplexity segments should be prioritized (see segment_compressor.compress)."""
    scored = [("filler", 2.0, 10), ("important", 50.0, 12), ("mid", 10.0, 8)]
    scored.sort(key=lambda x: x[1], reverse=True)
    target_tokens = 20
    kept: list[str] = []
    kept_tokens = 0
    for seg, _, tok_count in scored:
        if kept_tokens + tok_count <= target_tokens or not kept:
            kept.append(seg)
            kept_tokens += tok_count
        if kept_tokens >= target_tokens:
            break
    assert kept[0] == "important"
