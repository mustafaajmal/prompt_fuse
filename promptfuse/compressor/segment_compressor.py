"""Segment-level prompt compressor using proxy LM perplexity scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
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
    target_tokens: int = 0
    protected_segments_kept: int = 0

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
        self._protected_res = [re.compile(pat, re.IGNORECASE) for pat in self.config.preserve_patterns]

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
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
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

    def _segment_perplexities(self, segments: list[str]) -> list[float]:
        """
        Compute perplexity for all segments in one batched forward pass.

        Lower perplexity => lower importance (more redundant under the proxy LM).
        """
        self._load_model()
        assert self._tokenizer is not None and self._model is not None and self._device is not None
        if not segments:
            return []

        inputs = self._tokenizer(
            segments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        if input_ids.shape[1] < 2:
            return [float("inf")] * len(segments)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(shift_labels.shape[0], shift_labels.shape[1])
            valid = shift_labels != -100
            valid_counts = valid.sum(dim=1)
            summed_loss = (token_loss * valid).sum(dim=1)

            per_segment_loss = torch.full_like(summed_loss, fill_value=float("inf"), dtype=torch.float32)
            non_empty = valid_counts > 0
            per_segment_loss[non_empty] = summed_loss[non_empty] / valid_counts[non_empty]
            per_segment_ppl = torch.exp(per_segment_loss)

        return [float(x.item()) for x in per_segment_ppl]

    def _is_protected_segment(self, segment: str) -> bool:
        """Keep critical instruction-like segments even under aggressive compression."""
        return any(pattern.search(segment) for pattern in self._protected_res)

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
                target_tokens=tokens,
                protected_segments_kept=0,
            )

        token_cache: dict[str, int] = {}

        def token_count(text: str) -> int:
            if text not in token_cache:
                token_cache[text] = self.count_tokens(text)
            return token_cache[text]

        original_tokens = token_count(prompt)
        target_tokens = max(1, int(original_tokens * (1.0 - ratio)))

        segment_tokens = [token_count(seg) for seg in segments]
        segment_ppl = self._segment_perplexities(segments)

        protected_indices = [idx for idx, seg in enumerate(segments) if self._is_protected_segment(seg)]
        protected_set = set(protected_indices)

        # Keep protected segments first, then high-perplexity (informative) segments until budget is met.
        kept_indices: list[int] = list(protected_indices)
        kept_index_set = set(kept_indices)
        kept_tokens = sum(segment_tokens[idx] for idx in kept_indices)

        ranked_indices = sorted(
            (idx for idx in range(len(segments)) if idx not in protected_set),
            key=lambda idx: segment_ppl[idx],
            reverse=True,
        )
        for idx in ranked_indices:
            tok_count = segment_tokens[idx]
            if kept_tokens + tok_count <= target_tokens or not kept_indices:
                kept_indices.append(idx)
                kept_index_set.add(idx)
                kept_tokens += tok_count
            if kept_tokens >= target_tokens:
                break

        if not kept_indices:
            best_idx = max(range(len(segments)), key=lambda i: segment_ppl[i])
            kept_indices = [best_idx]
            kept_index_set = {best_idx}

        ordered = [seg for idx, seg in enumerate(segments) if idx in kept_index_set]

        compressed = " ".join(ordered)
        compressed_tokens = token_count(compressed)

        return CompressionResult(
            original=prompt,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            segments_kept=len(ordered),
            segments_total=len(segments),
            target_tokens=target_tokens,
            protected_segments_kept=sum(1 for idx in kept_indices if idx in protected_set),
        )
