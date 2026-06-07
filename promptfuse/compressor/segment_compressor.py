"""Segment-level prompt compressor using proxy LM perplexity scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from promptfuse.config import CompressorConfig

logger = logging.getLogger(__name__)

# Join kept segments the same way for budgeting and final text.
SEGMENT_JOIN = " "

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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Outcome of a single compression pass."""

    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    segments_kept: int
    segments_total: int
    target_tokens: int = 0
    protected_segments_kept: int = 0
    dropped_segments: list[str] = field(default_factory=list)
    truncated_segments: list[int] = field(default_factory=list)

    @property
    def token_reduction(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


@dataclass
class _SegmentScore:
    """Internal bookkeeping for a single segment during scoring."""

    index: int
    text: str
    token_count: int = 0
    perplexity: float = float("inf")


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Fallback when NLTK is unavailable: split after sentence-ending punctuation.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _nltk_punkt_tokenize(text: str) -> list[str] | None:
    """Tokenize with NLTK punkt_tab if available."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        return tokenizer.tokenize(text)  # type: ignore[no-any-return]
    except Exception:
        return None


def _nltk_sent_tokenize(text: str) -> list[str] | None:
    """Tokenize with NLTK sent_tokenize if available."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize

        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return None


def _regex_sentence_split(text: str) -> list[str]:
    """Conservative regex split on [.!?] boundaries."""
    parts = _SENTENCE_BOUNDARY_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


_CACHED_USE_PUNKT: bool | None = None


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-level segments.

    Tries, in order: NLTK punkt_tab, NLTK sent_tokenize, regex sentence
    boundaries, then paragraph boundaries. Single-paragraph multi-sentence
    prompts still split when NLTK or regex applies.
    """
    global _CACHED_USE_PUNKT

    stripped = text.strip()
    if not stripped:
        return []

    if _CACHED_USE_PUNKT is not False:
        sentences = _nltk_punkt_tokenize(stripped)
        if sentences and len(sentences) > 1:
            return sentences
        if sentences and len(sentences) == 1:
            # Punkt returned one span; try sent_tokenize / regex before giving up.
            pass
        elif sentences is None and _CACHED_USE_PUNKT is None:
            _CACHED_USE_PUNKT = False

    sentences = _nltk_sent_tokenize(stripped)
    if sentences and len(sentences) > 1:
        return sentences

    regex_parts = _regex_sentence_split(stripped)
    if len(regex_parts) > 1:
        return regex_parts

    parts = _PARAGRAPH_RE.split(stripped)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    return [stripped]


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class SegmentCompressor:
    """Sentence-segment-level compressor inspired by LLMLingua.

    Scores each sentence segment by average token perplexity under a small proxy LM,
    then drops lower-priority segments to approach the target compression ratio.

    Design goal: preserve grammatical and logical integrity by keeping full
    sentence segments and protecting constraint-bearing instructions.
    """

    def __init__(
        self,
        config: CompressorConfig | None = None,
        *,
        lazy_load: bool = False,
    ) -> None:
        from promptfuse.config import CompressorConfig as _CompressorConfig

        self.config = config or _CompressorConfig()
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None
        self._device: torch.device | None = None
        self._protected_res: list[re.Pattern[str]] = [
            re.compile(pat, re.IGNORECASE) for pat in self.config.preserve_patterns
        ]

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
        """Idempotent model loader with atomic success guarantee."""
        if self._model is not None:
            return

        device = self._resolve_device()
        logger.info("Loading proxy LM %s on %s", self.config.proxy_model, device)

        tokenizer = AutoTokenizer.from_pretrained(self.config.proxy_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.proxy_model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _ensure_loaded(self) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None
        return self._tokenizer, self._model, self._device

    def count_tokens(self, text: str) -> int:
        tokenizer, _, _ = self._ensure_loaded()
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _score_segments(
        self,
        segments: list[str],
    ) -> tuple[list[_SegmentScore], list[int]]:
        """Score segments; return (scores, indices truncated at max_length)."""
        tokenizer, model, device = self._ensure_loaded()

        if not segments:
            return [], []

        encodings = [
            tokenizer.encode(seg, add_special_tokens=False, truncation=False)
            for seg in segments
        ]

        infos: list[_SegmentScore] = []
        truncated_indices: list[int] = []

        for idx, (seg, enc) in enumerate(zip(segments, encodings)):
            tok_count = len(enc)
            if tok_count > self.config.max_length:
                truncated_indices.append(idx)
                tok_count = self.config.max_length
            infos.append(_SegmentScore(index=idx, text=seg, token_count=tok_count))

        if truncated_indices:
            logger.warning(
                "Segments at indices %s exceed max_length (%d) and will be "
                "truncated; their perplexity scores may be inaccurate.",
                truncated_indices,
                self.config.max_length,
            )

        order = sorted(range(len(infos)), key=lambda i: infos[i].token_count)
        batch_size = self.config.batch_size

        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_texts = [infos[i].text for i in batch_indices]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            if input_ids.shape[1] < 2:
                for bi in batch_indices:
                    infos[bi].perplexity = float("inf")
                continue

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]

                token_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape(shift_labels.shape)

                valid_mask = shift_labels != -100
                valid_counts = valid_mask.sum(dim=1).float()
                summed_loss = (token_loss * valid_mask).sum(dim=1)

                ppl = torch.where(
                    valid_counts > 0,
                    torch.exp(summed_loss / valid_counts),
                    torch.tensor(float("inf"), device=device),
                )

            for j, bi in enumerate(batch_indices):
                infos[bi].perplexity = ppl[j].item()

        return infos, truncated_indices

    def _token_importance_scores(self, prompt: str) -> list[tuple[int, float]]:
        """Per-token loss (higher = more important to keep)."""
        tokenizer, model, device = self._ensure_loaded()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            return []

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            per_token = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        return [(i + 1, float(per_token[i].item())) for i in range(per_token.shape[0])]

    def _compress_token_level(self, prompt: str, target_tokens: int) -> str:
        """Gentle token-level trim when segment compression cannot reduce enough."""
        tokenizer, _, _ = self._ensure_loaded()

        token_scores = self._token_importance_scores(prompt)
        if not token_scores:
            return prompt

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= target_tokens:
            return prompt

        token_strs = tokenizer.convert_ids_to_tokens(ids)
        protected_positions = set()
        for i, token in enumerate(token_strs):
            norm = token.lower().replace("Ġ", "").replace("▁", "")
            if any(key in norm for key in _TOKEN_PROTECT_KEYWORDS):
                protected_positions.add(i)
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

        return tokenizer.decode(kept_ids, skip_special_tokens=True)

    def _is_protected(self, segment: str) -> bool:
        return any(pat.search(segment) for pat in self._protected_res)

    def compress(
        self,
        prompt: str,
        compression_ratio: float | None = None,
    ) -> CompressionResult:
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
            )

        infos, truncated_indices = self._score_segments(segments)

        original_tokens = sum(info.token_count for info in infos)
        if original_tokens == 0:
            original_tokens = 1
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
                target_tokens=target_tokens,
            )

        protected_idx: list[int] = []
        unprotected_idx: list[int] = []
        for info in infos:
            if self._is_protected(info.text):
                protected_idx.append(info.index)
            else:
                unprotected_idx.append(info.index)

        protected_tokens = sum(infos[i].token_count for i in protected_idx)
        if protected_tokens > target_tokens:
            logger.warning(
                "Protected segments alone use %d tokens, exceeding the "
                "target budget of %d tokens.  All protected segments will "
                "be kept; effective compression will be lower than requested.",
                protected_tokens,
                target_tokens,
            )

        remaining_budget = max(0, target_tokens - protected_tokens)
        ranked = sorted(unprotected_idx, key=lambda i: infos[i].perplexity, reverse=True)

        kept_unprotected: list[int] = []
        used = 0
        for idx in ranked:
            tok = infos[idx].token_count
            if used + tok <= remaining_budget:
                kept_unprotected.append(idx)
                used += tok
            if used >= remaining_budget and kept_unprotected:
                break

        kept_set = set(protected_idx) | set(kept_unprotected)
        if not kept_set:
            best = max(range(len(infos)), key=lambda i: infos[i].perplexity)
            kept_set.add(best)

        ordered = [segments[i] for i in sorted(kept_set)]
        dropped = [segments[i] for i in range(len(segments)) if i not in kept_set]

        compressed = SEGMENT_JOIN.join(ordered)
        compressed_tokens = self.count_tokens(compressed)

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
            target_tokens=target_tokens,
            protected_segments_kept=len(protected_idx),
            dropped_segments=dropped,
            truncated_segments=truncated_indices,
        )
