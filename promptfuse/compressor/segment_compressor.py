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
_PROTECTED_SEGMENT_PATTERNS = [
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bdo not\b", re.IGNORECASE),
    re.compile(r"\bdon't\b", re.IGNORECASE),
    re.compile(r"\bonly\b", re.IGNORECASE),
    re.compile(r"\bexactly\b", re.IGNORECASE),
    re.compile(r"\breturn\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bjson\b", re.IGNORECASE),
    re.compile(r"\bxml\b", re.IGNORECASE),
    re.compile(r"\btable\b", re.IGNORECASE),
]
_TOKEN_PROTECT_KEYWORDS = {
    "not",
    "n't",
    "no",
    "never",
    "must",
    "should",
    "return",
    "format",
    "json",
    "xml",
}


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
    then drops lower-priority segments to approach the target compression ratio.

    Design goal: preserve grammatical and logical integrity by keeping full
    sentence segments and protecting constraint-bearing instructions.
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
        """Higher perplexity => more informative content (LLMLingua-style importance)."""
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
            return 0.0

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        return float(torch.exp(torch.tensor(loss)).item())

    def _token_importance_scores(self, prompt: str) -> list[tuple[int, float]]:
        """Per-token loss (higher = more important to keep)."""
        self._load_model()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = inputs["input_ids"].to(self._device)
        if input_ids.shape[1] < 2:
            return []

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            per_token = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        scores = [(i + 1, float(per_token[i].item())) for i in range(per_token.shape[0])]
        return scores

    def _compress_token_level(self, prompt: str, target_tokens: int) -> str:
        """Gentle token-level trim when segment compression cannot reduce enough.

        This is intentionally conservative and protects constraint keywords to
        avoid invalidating prompt logic.
        """
        self._load_model()
        assert self._tokenizer is not None

        token_scores = self._token_importance_scores(prompt)
        if not token_scores:
            return prompt

        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= target_tokens:
            return prompt

        token_strs = self._tokenizer.convert_ids_to_tokens(ids)
        protected_positions = set()
        for i, token in enumerate(token_strs):
            norm = token.lower().replace("Ġ", "").replace("▁", "")
            if any(key in norm for key in _TOKEN_PROTECT_KEYWORDS):
                protected_positions.add(i)
        # Protect a small head/tail window for structural coherence.
        protected_positions.update(range(min(4, len(ids))))
        protected_positions.update(range(max(0, len(ids) - 2), len(ids)))

        score_by_position = {pos: score for pos, score in token_scores}
        ranked_candidates = sorted(
            (
                (i, score_by_position.get(i, float("inf")))
                for i in range(len(ids))
                if i not in protected_positions
            ),
            key=lambda x: x[1],
        )
        needed = max(0, len(ids) - target_tokens)
        drop_positions = {pos for pos, _ in ranked_candidates[:needed]}

        kept_ids = [tid for i, tid in enumerate(ids) if i not in drop_positions]
        if len(kept_ids) > target_tokens:
            # Last-resort trim from non-protected tail positions.
            overflow = len(kept_ids) - target_tokens
            tail_drop = []
            for i in range(len(ids) - 1, -1, -1):
                if i in drop_positions or i in protected_positions:
                    continue
                tail_drop.append(i)
                if len(tail_drop) >= overflow:
                    break
            if tail_drop:
                drop_positions.update(tail_drop)
                kept_ids = [tid for i, tid in enumerate(ids) if i not in drop_positions]

        if not kept_ids:
            kept_ids = ids[:target_tokens]

        return self._tokenizer.decode(kept_ids, skip_special_tokens=True)

    @staticmethod
    def _is_protected_segment(segment: str) -> bool:
        return any(pattern.search(segment) for pattern in _PROTECTED_SEGMENT_PATTERNS)

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

        if len(segments) <= 1 and original_tokens > target_tokens:
            compressed = self._compress_token_level(prompt, target_tokens)
            compressed_tokens = self.count_tokens(compressed)
            return CompressionResult(
                original=prompt,
                compressed=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                segments_kept=len(segments),
                segments_total=len(segments),
            )

        scored: list[tuple[int, str, float, int]] = []
        for idx, seg in enumerate(segments):
            tok_count = self.count_tokens(seg)
            ppl = self._segment_perplexity(seg)
            scored.append((idx, seg, ppl, tok_count))

        # Keep highest-perplexity segments first (informative content) while
        # preserving instruction constraints and anchor structure.
        anchor_idx = 0
        kept_indices: set[int] = {anchor_idx}
        kept_tokens = next(tok for i, _s, _p, tok in scored if i == anchor_idx)

        protected_indices = {
            i for i, s, _p, _tok in scored if i != anchor_idx and self._is_protected_segment(s)
        }
        for i in sorted(protected_indices):
            tok_count = next(tok for idx, _s, _p, tok in scored if idx == i)
            if i not in kept_indices:
                kept_indices.add(i)
                kept_tokens += tok_count

        ranked = sorted(
            (item for item in scored if item[0] not in kept_indices),
            key=lambda x: x[2],  # perplexity
            reverse=True,
        )
        for idx, _seg, _ppl, tok_count in ranked:
            # Keep at least two segments when available to preserve flow.
            if len(kept_indices) < 2:
                kept_indices.add(idx)
                kept_tokens += tok_count
                continue
            if kept_tokens + tok_count <= target_tokens:
                kept_indices.add(idx)
                kept_tokens += tok_count
            if kept_tokens >= target_tokens:
                break

        ordered = [seg for idx, seg in enumerate(segments) if idx in kept_indices]

        compressed = " ".join(ordered)
        compressed_tokens = self.count_tokens(compressed)

        # Only use token fallback when sentence granularity cannot reduce enough
        # and we are still meaningfully over budget.
        if compressed_tokens > int(target_tokens * 1.35) and len(segments) <= 2:
            compressed = self._compress_token_level(prompt, target_tokens)
            compressed_tokens = self.count_tokens(compressed)

        return CompressionResult(
            original=prompt,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            segments_kept=len(ordered),
            segments_total=len(segments),
        )
