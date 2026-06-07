"""Semantic unifier: maps compressed prompts to canonical forms via bi-encoder retrieval."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from promptfuse.unifier.canonical_store import CanonicalEntry, CanonicalStore

if TYPE_CHECKING:
    from promptfuse.config import UnifierConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnificationResult:
    """Outcome of a single unification pass."""

    original: str
    unified: str
    cache_hit: bool
    similarity: float          # cosine similarity to the matched canonical form,
                               # or best-below-threshold score on a miss (0.0 if
                               # the inventory was empty).
    canonical_id: int | None
    token_count: int
    is_new_canonical: bool = False  # True when this prompt created a new entry.


# ---------------------------------------------------------------------------
# Unifier
# ---------------------------------------------------------------------------

class SemanticUnifier:
    """Embeds incoming prompts and retrieves canonical forms from a FAISS index.

    If cosine similarity ≥ τ, the prompt is replaced with the canonical form.
    Otherwise the prompt itself is added to the inventory as a new canonical
    entry.

    Thread safety
    -------------
    All mutations to the canonical store are serialised by an internal lock so
    the unifier can be shared across async / threaded request handlers (e.g.
    FastAPI with a thread-pool executor).
    """

    def __init__(
        self,
        config: UnifierConfig | None = None,
        *,
        lazy_load: bool = False,
    ) -> None:
        from promptfuse.config import UnifierConfig as _UnifierConfig

        self.config = config or _UnifierConfig()
        self._encoder = None
        self._lock = threading.RLock()
        self.store = CanonicalStore(
            embedding_dim=self.config.embedding_dim,
            inventory_path=self.config.inventory_path,
        )

        if not lazy_load:
            self._load_encoder()

    # ---- encoder lifecycle -------------------------------------------------

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        from sentence_transformers import SentenceTransformer

        model_name = self.config.fine_tuned_encoder or self.config.encoder_model
        logger.info("Loading bi-encoder %s", model_name)
        self._encoder = SentenceTransformer(model_name)

    def _ensure_encoder(self):
        self._load_encoder()
        assert self._encoder is not None
        return self._encoder

    # ---- embedding ---------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single prompt and return an L2-normalised vector."""
        encoder = self._ensure_encoder()
        vec = encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        # encode() returns 1-D for a single string — keep it that way.
        return vec  # type: ignore[return-value]

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed multiple prompts and return L2-normalised vectors (N × D).

        Batching through the model is significantly faster than calling
        ``embed()`` in a loop because it amortises GPU kernel launches and
        benefits from data parallelism.
        """
        encoder = self._ensure_encoder()
        vecs = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
        )
        return vecs  # type: ignore[return-value]

    # ---- token counting ----------------------------------------------------

    @staticmethod
    def _whitespace_token_count(text: str) -> int:
        """Rough whitespace-based token estimate (last resort)."""
        return len(text.split())

    def _resolve_token_count(self, text: str, provided: int | None) -> int:
        """Use *provided* count when available; fall back with a warning."""
        if provided is not None:
            return provided
        logger.debug(
            "No token count provided for prompt (len=%d chars); using "
            "whitespace approximation.  Pass token_count from the "
            "compressor for accurate budget tracking.",
            len(text),
        )
        return self._whitespace_token_count(text)

    # ---- core unification --------------------------------------------------

    def unify(
        self,
        prompt: str,
        token_count: int | None = None,
    ) -> UnificationResult:
        """Map *prompt* to a canonical form if a close-enough entry exists.

        Parameters
        ----------
        prompt:
            Compressed (or raw) prompt text.
        token_count:
            Pre-computed token count from the compressor.  If omitted a rough
            whitespace estimate is used (with a log warning).
        """
        tok_count = self._resolve_token_count(prompt, token_count)
        embedding = self.embed(prompt)

        with self._lock:
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

            # No match — this prompt becomes a new canonical entry.
            entry = self.store.add(prompt, embedding, tok_count)

        return UnificationResult(
            original=prompt,
            unified=prompt,
            cache_hit=False,
            similarity=similarity,
            canonical_id=entry.id,
            token_count=tok_count,
            is_new_canonical=True,
        )

    # ---- batch warmup ------------------------------------------------------

    def warmup(
        self,
        prompts: list[tuple[str, int]],
        batch_size: int = 64,
    ) -> dict[str, int]:
        """Build the canonical inventory from a batch of historical prompts.

        Instead of processing prompts one-at-a-time (which makes the result
        depend on insertion order), this method:

        1. Embeds every prompt in one batched pass.
        2. Greedily clusters them: for each prompt, find the nearest existing
           cluster centroid.  If similarity ≥ τ, assign to that cluster;
           otherwise start a new cluster.
        3. Within each cluster, selects the *shortest-token* prompt as the
           canonical form.

        Parameters
        ----------
        prompts:
            List of ``(text, token_count)`` pairs.
        batch_size:
            Encoding batch size forwarded to the sentence-transformer.

        Returns
        -------
        stats : dict
            ``{"clusters": int, "prompts_processed": int}``
        """
        if not prompts:
            return {"clusters": 0, "prompts_processed": 0}

        texts = [t for t, _ in prompts]
        token_counts = [c for _, c in prompts]
        embeddings = self.embed_batch(texts, batch_size=batch_size)

        # Greedy single-pass clustering.
        # Each cluster is represented by its centroid (mean of member
        # embeddings) and the list of (index, text, token_count) members.
        threshold = self.config.similarity_threshold

        clusters: list[dict] = []
        # cluster = {"centroid": np.ndarray, "members": [(idx, text, tok_count), ...]}

        for i, (text, tok_count, emb) in enumerate(
            zip(texts, token_counts, embeddings)
        ):
            best_cluster = None
            best_sim = -1.0

            for ci, cluster in enumerate(clusters):
                sim = float(np.dot(emb, cluster["centroid"]))
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_cluster = ci

            if best_cluster is not None:
                c = clusters[best_cluster]
                c["members"].append((i, text, tok_count))
                # Update centroid as running mean, re-normalise.
                n = len(c["members"])
                c["centroid"] = c["centroid"] * ((n - 1) / n) + emb / n
                norm = np.linalg.norm(c["centroid"])
                if norm > 0:
                    c["centroid"] /= norm
            else:
                clusters.append({
                    "centroid": emb.copy(),
                    "members": [(i, text, tok_count)],
                })

        # For each cluster, pick the shortest-token member as canonical.
        with self._lock:
            for cluster in clusters:
                members = cluster["members"]
                # Sort by token count ascending; ties broken by original order.
                best_idx, best_text, best_tok = min(
                    members, key=lambda m: (m[2], m[0])
                )
                best_emb = embeddings[best_idx]
                self.store.add(best_text, best_emb, best_tok)

        logger.info(
            "Warmup complete: %d prompts → %d canonical clusters.",
            len(prompts),
            len(clusters),
        )

        return {"clusters": len(clusters), "prompts_processed": len(prompts)}

    # ---- persistence -------------------------------------------------------

    def save_inventory(self) -> None:
        """Persist the canonical store to disk."""
        with self._lock:
            self.store.save()

    def reload_inventory(self) -> None:
        """Reload the canonical store from disk (e.g. after an external update)."""
        with self._lock:
            self.store.load()