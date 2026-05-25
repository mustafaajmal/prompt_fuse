"""Tests for PromptFuse core components."""

from __future__ import annotations

import numpy as np
import pytest

from promptfuse.compressor.segment_compressor import SegmentCompressor, split_sentences
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
        monkeypatch.setattr(compressor, "_segment_perplexities", lambda _segments: [1.0, 1.1, 9.0])
        monkeypatch.setattr(compressor, "_is_protected_segment", lambda _seg: False)
        monkeypatch.setattr(compressor, "count_tokens", lambda text: len(text.split()))

        result = compressor.compress(prompt, compression_ratio=0.55)
        assert "Unique detailed evidence" in result.compressed
        assert "Generic filler sentence one." not in result.compressed
        assert result.segments_kept >= 1

    def test_preserves_guardrail_segments(self, monkeypatch):
        compressor = SegmentCompressor(CompressorConfig(), lazy_load=True)
        segments = ["You must return JSON only.", "Some filler details.", "More filler details."]

        monkeypatch.setattr(
            "promptfuse.compressor.segment_compressor.split_sentences",
            lambda _prompt: segments,
        )
        monkeypatch.setattr(compressor, "_segment_perplexities", lambda _segments: [10.0, 1.0, 1.1])
        monkeypatch.setattr(compressor, "count_tokens", lambda text: len(text.split()))

        result = compressor.compress("unused", compression_ratio=0.80)
        assert "You must return JSON only." in result.compressed
        assert result.protected_segments_kept >= 1
