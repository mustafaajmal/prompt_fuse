"""Segment-level prompt compressor using proxy LM perplexity scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from promptfuse.config import CompressorConfig

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class CompressionResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    segments_kept: int
    segments_total: int

    @property
    def token_reduction(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-level segments."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        parts = _SENTENCE_RE.split(text.strip())
        return [p.strip() for p in parts if p.strip()]


class SegmentCompressor:
    """
    Sentence-segment-level compressor inspired by LLMLingua.

    Scores each sentence segment by average token perplexity under a small proxy LM,
    then drops lowest-importance segments until the target compression ratio is met.
    """

    def __init__(self, config: CompressorConfig | None = None, *, lazy_load: bool = False):
        from promptfuse.config import CompressorConfig as _CompressorConfig

        self.config = config or _CompressorConfig()
        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None

        if not lazy_load:
            self._load_model()

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._device = self._resolve_device()
        logger.info("Loading proxy LM %s on %s", self.config.proxy_model, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.proxy_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.proxy_model,
            torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()

    def count_tokens(self, text: str) -> int:
        self._load_model()
        assert self._tokenizer is not None
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _segment_perplexity(self, segment: str) -> float:
        """Lower perplexity => higher importance (more predictable / structural)."""
        self._load_model()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        inputs = self._tokenizer(
            segment,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = inputs["input_ids"].to(self._device)
        if input_ids.shape[1] < 2:
            return float("inf")

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        return float(torch.exp(torch.tensor(loss)).item())

    def compress(self, prompt: str, compression_ratio: float | None = None) -> CompressionResult:
        """
        Compress prompt by dropping low-importance sentence segments.

        Args:
            prompt: Raw input prompt.
            compression_ratio: Target fraction of tokens to remove (0.25 = 25% reduction).
        """
        ratio = compression_ratio if compression_ratio is not None else self.config.compression_ratio
        ratio = max(0.0, min(ratio, 0.90))

        segments = split_sentences(prompt)
        if not segments:
            tokens = self.count_tokens(prompt)
            return CompressionResult(
                original=prompt,
                compressed=prompt,
                original_tokens=tokens,
                compressed_tokens=tokens,
                segments_kept=0,
                segments_total=0,
            )

        original_tokens = self.count_tokens(prompt)
        target_tokens = max(1, int(original_tokens * (1.0 - ratio)))

        scored: list[tuple[str, float, int]] = []
        for seg in segments:
            tok_count = self.count_tokens(seg)
            ppl = self._segment_perplexity(seg)
            scored.append((seg, ppl, tok_count))

        # Keep segments with lowest perplexity (most important / structural)
        scored.sort(key=lambda x: x[1])

        kept: list[str] = []
        kept_tokens = 0
        for seg, _, tok_count in scored:
            if kept_tokens + tok_count <= target_tokens or not kept:
                kept.append(seg)
                kept_tokens += tok_count
            if kept_tokens >= target_tokens:
                break

        kept_texts = set(kept)
        ordered = [seg for seg in segments if seg in kept_texts]

        compressed = " ".join(ordered)
        compressed_tokens = self.count_tokens(compressed)

        return CompressionResult(
            original=prompt,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            segments_kept=len(ordered),
            segments_total=len(segments),
        )
