"""Semantic unifier: maps compressed prompts to canonical forms via bi-encoder retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from promptfuse.unifier.canonical_store import CanonicalEntry, CanonicalStore

if TYPE_CHECKING:
    from promptfuse.config import UnifierConfig

logger = logging.getLogger(__name__)


@dataclass
class UnificationResult:
    original: str
    unified: str
    cache_hit: bool
    similarity: float
    canonical_id: int | None
    token_count: int


class SemanticUnifier:
    """
    Embeds incoming prompts and retrieves canonical forms from FAISS inventory.

    If cosine similarity >= τ, replace with canonical form; otherwise add to inventory.
    """

    def __init__(self, config: UnifierConfig | None = None, *, lazy_load: bool = False):
        from promptfuse.config import UnifierConfig as _UnifierConfig

        self.config = config or _UnifierConfig()
        self._encoder = None
        self.store = CanonicalStore(
            embedding_dim=self.config.embedding_dim,
            inventory_path=self.config.inventory_path,
        )

        if not lazy_load:
            self._load_encoder()

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_name = self.config.fine_tuned_encoder or self.config.encoder_model
        logger.info("Loading bi-encoder %s", model_name)
        self._encoder = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        self._load_encoder()
        assert self._encoder is not None
        return self._encoder.encode(text, convert_to_numpy=True)

    def count_tokens_approx(self, text: str) -> int:
        """Whitespace token count as lightweight proxy when no tokenizer available."""
        return len(text.split())

    def unify(self, prompt: str, token_count: int | None = None) -> UnificationResult:
        """
        Map prompt to canonical form if semantically similar entry exists.

        Args:
            prompt: Compressed (or raw) prompt text.
            token_count: Optional precomputed token count from compressor.
        """
        tok_count = token_count if token_count is not None else self.count_tokens_approx(prompt)
        embedding = self.embed(prompt)

        matched, similarity = self.store.search(
            embedding,
            threshold=self.config.similarity_threshold,
        )

        if matched is not None:
            return UnificationResult(
                original=prompt,
                unified=matched.text,
                cache_hit=True,
                similarity=similarity,
                canonical_id=matched.id,
                token_count=matched.token_count,
            )

        entry = self.store.add(prompt, embedding, tok_count)
        return UnificationResult(
            original=prompt,
            unified=prompt,
            cache_hit=False,
            similarity=similarity,
            canonical_id=entry.id,
            token_count=tok_count,
        )

    def warmup(self, prompts: list[tuple[str, int]]) -> None:
        """
        Batch warmup over historical prompts to build canonical inventory.

        Selects shortest-token form per semantic cluster.
        """
        for text, tok_count in prompts:
            embedding = self.embed(text)
            matched, similarity = self.store.search(
                embedding,
                threshold=self.config.similarity_threshold,
            )
            self.store.maybe_add_shorter(
                text,
                embedding,
                tok_count,
                matched,
                similarity,
                self.config.similarity_threshold,
            )

    def save_inventory(self) -> None:
        self.store.save()
