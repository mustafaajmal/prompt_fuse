"""Tests for PromptFuse core components."""

from __future__ import annotations

import numpy as np
import pytest

from promptfuse.compressor.segment_compressor import split_sentences
from promptfuse.evaluation.metrics import BenchmarkMetrics, compute_rouge_l
from promptfuse.unifier.canonical_store import CanonicalStore


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
    def unifier(self):
        from promptfuse.unifier.semantic_unifier import SemanticUnifier
        from promptfuse.config import UnifierConfig

        # Base MiniLM reaches ~0.81 on paraphrase pairs; fine-tuned model uses 0.85
        config = UnifierConfig(
            encoder_model="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.80,
        )
        return SemanticUnifier(config, lazy_load=False)

    def test_unify_paraphrases(self, unifier):
        p1 = "Summarize the following paragraph in three sentences."
        p2 = "Please give a three-sentence summary of the text below."

        r1 = unifier.unify(p1, token_count=10)
        assert r1.cache_hit is False

        r2 = unifier.unify(p2, token_count=12)
        assert r2.cache_hit is True
        assert r2.similarity >= unifier.config.similarity_threshold
