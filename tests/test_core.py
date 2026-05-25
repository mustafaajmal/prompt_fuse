"""Tests for PromptFuse core components."""

from __future__ import annotations

import numpy as np
import pytest

from promptfuse.compressor.segment_compressor import (
    SegmentCompressor,
    _SegmentScore,
    split_sentences,
)
from promptfuse.config import CompressorConfig
from promptfuse.evaluation.metrics import BenchmarkMetrics, compute_rouge_l
from promptfuse.unifier.canonical_store import CanonicalStore
from scripts.generate_synthetic_paraphrases import expand_clusters


class TestSplitSentences:
    def test_basic_split(self):
        text = "Hello world. This is a test. Another sentence!"
        segments = split_sentences(text)
        assert len(segments) == 3

    def test_single_sentence(self):
        assert split_sentences("One sentence only") == ["One sentence only"]

    def test_single_paragraph_splits_without_blank_lines(self, monkeypatch):
        """Multi-sentence text must split even without paragraph breaks."""
        text = "Hello world. This is a test. Another sentence!"
        monkeypatch.setattr(
            "promptfuse.compressor.segment_compressor._nltk_punkt_tokenize",
            lambda _t: None,
        )
        monkeypatch.setattr(
            "promptfuse.compressor.segment_compressor._nltk_sent_tokenize",
            lambda _t: None,
        )
        segments = split_sentences(text)
        assert len(segments) == 3


class TestCanonicalStore:
    def test_add_and_search(self):
        store = CanonicalStore(embedding_dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        store.add("canonical prompt", emb, token_count=2)

        query = np.array([0.99, 0.01, 0.0, 0.0])
        entry, sim = store.search(query, threshold=0.85)
        assert entry is not None
        assert sim >= 0.85
        assert entry.text == "canonical prompt"

    def test_no_match_below_threshold(self):
        store = CanonicalStore(embedding_dim=4)
        store.add("prompt a", np.array([1.0, 0.0, 0.0, 0.0]), token_count=2)

        entry, _ = store.search(np.array([0.0, 1.0, 0.0, 0.0]), threshold=0.85)
        assert entry is None

    def test_save_and_load(self, tmp_path):
        store = CanonicalStore(embedding_dim=4, inventory_path=tmp_path)
        store.add("test prompt", np.array([1.0, 0.0, 0.0, 0.0]), token_count=2)
        store.save()

        loaded = CanonicalStore(embedding_dim=4, inventory_path=tmp_path)
        assert loaded.size == 1
        entry, sim = loaded.search(np.array([1.0, 0.0, 0.0, 0.0]), threshold=0.99)
        assert entry is not None


class TestMetrics:
    def test_rouge_l_identical(self):
        text = "Summarize the following paragraph in three sentences."
        assert compute_rouge_l(text, text) == pytest.approx(1.0, abs=0.01)

    def test_benchmark_summary(self):
        m = BenchmarkMetrics()
        m.record(token_reduction=0.35, rouge_l=0.90, latency_ms=10.0, cache_hit=True)
        m.record(token_reduction=0.40, rouge_l=0.88, latency_ms=20.0, cache_hit=False)
        summary = m.summary()
        assert summary["total_requests"] == 2
        assert summary["unifier_hit_rate"] == 0.5


class TestSemanticUnifier:
    @pytest.fixture()
    def unifier(self, tmp_path, monkeypatch):
        from promptfuse.unifier.semantic_unifier import SemanticUnifier
        from promptfuse.config import UnifierConfig

        # Use a temp inventory + deterministic embeddings for isolated unit tests.
        config = UnifierConfig(
            inventory_path=str(tmp_path / "inventory"),
            similarity_threshold=0.80,
            embedding_dim=4,
        )
        unifier = SemanticUnifier(config, lazy_load=True)

        embedding_map = {
            "Summarize the following paragraph in three sentences.": np.array([1.0, 0.0, 0.0, 0.0]),
            "Please give a three-sentence summary of the text below.": np.array(
                [0.98, 0.02, 0.0, 0.0]
            ),
        }

        def fake_embed(text: str) -> np.ndarray:
            return embedding_map.get(text, np.array([0.0, 1.0, 0.0, 0.0]))

        monkeypatch.setattr(unifier, "embed", fake_embed)
        return unifier

    def test_unify_paraphrases(self, unifier):
        p1 = "Summarize the following paragraph in three sentences."
        p2 = "Please give a three-sentence summary of the text below."

        r1 = unifier.unify(p1, token_count=10)
        assert r1.cache_hit is False

        r2 = unifier.unify(p2, token_count=12)
        assert r2.cache_hit is True
        assert r2.similarity >= unifier.config.similarity_threshold


class TestSyntheticDataGenerator:
    def test_default_target_size_and_clusters(self):
        records = expand_clusters()
        assert len(records) == 500
        assert len({r["cluster_id"] for r in records}) == 50

    def test_custom_target_size(self):
        records = expand_clusters(target_size=120, n_clusters=12)
        assert len(records) == 120
        assert len({r["cluster_id"] for r in records}) == 12


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


class TestSegmentCompressor:
    def test_keeps_high_perplexity_segments_first(self, monkeypatch):
        compressor = SegmentCompressor(CompressorConfig(), lazy_load=True)
        segments = [
            "Generic filler sentence one.",
            "Generic filler sentence two.",
            "Unique detailed evidence with rare terminology and specifics.",
        ]
        prompt = " ".join(segments)

        monkeypatch.setattr(
            "promptfuse.compressor.segment_compressor.split_sentences",
            lambda _prompt: segments,
        )
        def token_fn(seg: str) -> int:
            if "Unique" in seg:
                return 3
            return 4

        _mock_score_segments(compressor, [1.0, 1.1, 9.0], token_fn=token_fn)
        monkeypatch.setattr(compressor, "count_tokens", token_fn)

        result = compressor.compress(prompt, compression_ratio=0.55)
        assert "Unique detailed evidence" in result.compressed
        assert "Generic filler sentence one." not in result.compressed
        assert result.segments_kept >= 1
        assert result.original_tokens == 11

    def test_preserves_guardrail_segments(self, monkeypatch):
        compressor = SegmentCompressor(CompressorConfig(), lazy_load=True)
        segments = ["You must return JSON only.", "Some filler details.", "More filler details."]

        monkeypatch.setattr(
            "promptfuse.compressor.segment_compressor.split_sentences",
            lambda _prompt: segments,
        )
        _mock_score_segments(compressor, [10.0, 1.0, 1.1], token_fn=lambda s: len(s.split()))
        monkeypatch.setattr(compressor, "count_tokens", lambda text: len(text.split()))

        result = compressor.compress("unused", compression_ratio=0.80)
        assert "You must return JSON only." in result.compressed
        assert result.protected_segments_kept >= 1
