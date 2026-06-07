"""End-to-end PromptFuse pipeline: compress → unify → serve."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from promptfuse.compressor import CompressionResult, SegmentCompressor
from promptfuse.config import PromptFuseConfig, Settings
from promptfuse.unifier import SemanticUnifier, UnificationResult


@dataclass
class PipelineResult:
    raw_prompt: str
    final_prompt: str
    compression: CompressionResult | None = None
    unification: UnificationResult | None = None
    compression_ms: float = 0.0
    unification_ms: float = 0.0
    total_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def token_reduction(self) -> float:
        if self.compression:
            return self.compression.token_reduction
        return 0.0

    @property
    def cache_hit(self) -> bool:
        if self.unification:
            return self.unification.cache_hit
        return False


class PromptFusePipeline:
    """
    Orchestrates the full PromptFuse flow:
    raw_prompt → compressor → unifier → (vLLM)
    """

    def __init__(
        self,
        config: PromptFuseConfig | None = None,
        *,
        lazy_load: bool = False,
        enable_compression: bool | None = None,
        enable_unification: bool | None = None,
    ):
        self.config = config or Settings().load()
        self.enable_compression = (
            enable_compression
            if enable_compression is not None
            else self.config.serving.enable_compression
        )
        self.enable_unification = (
            enable_unification
            if enable_unification is not None
            else self.config.serving.enable_unification
        )

        self.compressor = (
            SegmentCompressor(self.config.compressor, lazy_load=lazy_load)
            if self.enable_compression
            else None
        )
        self.unifier = (
            SemanticUnifier(self.config.unifier, lazy_load=lazy_load)
            if self.enable_unification
            else None
        )

    def process(
        self,
        raw_prompt: str,
        compression_ratio: float | None = None,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        current = raw_prompt
        compression_result = None
        unification_result = None
        compression_ms = 0.0
        unification_ms = 0.0

        if self.compressor:
            t_comp = time.perf_counter()
            compression_result = self.compressor.compress(raw_prompt, compression_ratio)
            compression_ms = (time.perf_counter() - t_comp) * 1000
            current = compression_result.compressed

        if self.unifier:
            t_unify = time.perf_counter()
            tok_count = compression_result.compressed_tokens if compression_result else None
            unification_result = self.unifier.unify(current, token_count=tok_count)
            unification_ms = (time.perf_counter() - t_unify) * 1000
            current = unification_result.unified

        total_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            raw_prompt=raw_prompt,
            final_prompt=current,
            compression=compression_result,
            unification=unification_result,
            compression_ms=compression_ms,
            unification_ms=unification_ms,
            total_ms=total_ms,
        )

    def noop_result(self, raw_prompt: str) -> PipelineResult:
        """Return a pass-through result when no compressible content is found."""
        return PipelineResult(
            raw_prompt=raw_prompt,
            final_prompt=raw_prompt,
        )
