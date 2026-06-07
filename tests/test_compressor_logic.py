"""Compressor logic tests (no GPU / no model download)."""

from __future__ import annotations

from promptfuse.compressor.segment_compressor import (
    SegmentCompressor,
    _SegmentScore,
    split_sentences,
)


def _mock_score_segments(
    compressor: SegmentCompressor,
    perplexities: list[float],
    *,
    token_fn=None,
) -> None:
    def fake(segments: list[str]) -> tuple[list[_SegmentScore], list[int]]:
        infos = []
        for idx, seg in enumerate(segments):
            tok = token_fn(seg) if token_fn else len(seg.split())
            infos.append(
                _SegmentScore(
                    index=idx,
                    text=seg,
                    token_count=tok,
                    perplexity=perplexities[idx],
                )
            )
        return infos, []

    compressor._score_segments = fake  # type: ignore[method-assign]


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
    segments = [
        "You are a careful assistant.",
        "You must return JSON with keys answer and confidence.",
        "Add brief rationale.",
    ]
    prompt = " ".join(segments)

    _mock_score_segments(
        compressor,
        [80.0, 1.0, 10.0],
        token_fn=lambda s: len(s.split()),
    )
    compressor.count_tokens = lambda text: max(1, len(text.split()))  # type: ignore[method-assign]

    result = compressor.compress(prompt, compression_ratio=0.55)
    assert "must return JSON" in result.compressed


def test_token_fallback_is_not_used_when_only_slightly_over_target():
    compressor = SegmentCompressor(lazy_load=True)
    segments = [
        "You are a helpful assistant.",
        "Provide a concise explanation with one example.",
        "Keep the tone professional.",
    ]
    prompt = " ".join(segments)

    _mock_score_segments(
        compressor,
        [10.0, 10.0, 10.0],
        token_fn=lambda s: len(s.split()),
    )
    compressor.count_tokens = lambda text: max(1, len(text.split()))  # type: ignore[method-assign]

    called = {"token_fallback": False}

    def _fallback(prompt_text: str, target_tokens: int) -> str:
        called["token_fallback"] = True
        return prompt_text

    compressor._compress_token_level = _fallback  # type: ignore[method-assign]

    compressor.compress(prompt, compression_ratio=0.34)
    assert called["token_fallback"] is False
