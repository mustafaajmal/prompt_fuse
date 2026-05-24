"""Evaluation metrics for PromptFuse benchmarks."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from rouge_score import rouge_scorer


@dataclass
class BenchmarkMetrics:
    token_reductions: list[float] = field(default_factory=list)
    rouge_l_scores: list[float] = field(default_factory=list)
    pipeline_latencies_ms: list[float] = field(default_factory=list)
    unifier_hits: int = 0
    total_requests: int = 0
    vllm_cache_hits: int = 0

    def record(
        self,
        *,
        token_reduction: float,
        rouge_l: float | None = None,
        latency_ms: float,
        cache_hit: bool = False,
        vllm_cache_hit: bool = False,
    ) -> None:
        self.token_reductions.append(token_reduction)
        if rouge_l is not None:
            self.rouge_l_scores.append(rouge_l)
        self.pipeline_latencies_ms.append(latency_ms)
        self.total_requests += 1
        if cache_hit:
            self.unifier_hits += 1
        if vllm_cache_hit:
            self.vllm_cache_hits += 1

    @property
    def avg_token_reduction(self) -> float:
        return statistics.mean(self.token_reductions) if self.token_reductions else 0.0

    @property
    def avg_rouge_l(self) -> float:
        return statistics.mean(self.rouge_l_scores) if self.rouge_l_scores else 0.0

    @property
    def p99_latency_ms(self) -> float:
        if not self.pipeline_latencies_ms:
            return 0.0
        sorted_lat = sorted(self.pipeline_latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def unifier_hit_rate(self) -> float:
        return self.unifier_hits / self.total_requests if self.total_requests else 0.0

    @property
    def vllm_cache_hit_rate(self) -> float:
        return self.vllm_cache_hits / self.total_requests if self.total_requests else 0.0

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "avg_token_reduction": round(self.avg_token_reduction, 4),
            "avg_rouge_l": round(self.avg_rouge_l, 4),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "unifier_hit_rate": round(self.unifier_hit_rate, 4),
            "vllm_cache_hit_rate": round(self.vllm_cache_hit_rate, 4),
        }


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]
