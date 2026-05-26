"""Compressor logic tests (no GPU / no model download)."""

from __future__ import annotations

from promptfuse.compressor.segment_compressor import split_sentences
from promptfuse.compressor.segment_compressor import SegmentCompressor


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


def test_compress_keeps_protected_constraint_segment():
    compressor = SegmentCompressor(lazy_load=True)
    compressor.count_tokens = lambda text: max(1, len(text.split()))

    score_map = {
        "You are a careful assistant.": 80.0,
        "You must return JSON with keys answer and confidence.": 1.0,
        "Add brief rationale.": 10.0,
    }
    compressor._segment_perplexity = lambda seg: score_map.get(seg, 5.0)

    prompt = (
        "You are a careful assistant. "
        "You must return JSON with keys answer and confidence. "
        "Add brief rationale."
    )
    result = compressor.compress(prompt, compression_ratio=0.55)
    assert "must return JSON" in result.compressed


def test_token_fallback_is_not_used_when_only_slightly_over_target():
    compressor = SegmentCompressor(lazy_load=True)
    compressor.count_tokens = lambda text: max(1, len(text.split()))
    compressor._segment_perplexity = lambda seg: 10.0

    called = {"token_fallback": False}

    def _fallback(prompt: str, target_tokens: int) -> str:
        called["token_fallback"] = True
        return prompt

    compressor._compress_token_level = _fallback

    prompt = (
        "You are a helpful assistant. "
        "Provide a concise explanation with one example. "
        "Keep the tone professional."
    )
    compressor.compress(prompt, compression_ratio=0.34)
    assert called["token_fallback"] is False
