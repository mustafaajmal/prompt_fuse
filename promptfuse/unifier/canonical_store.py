"""FAISS-backed canonical prompt inventory for semantic unification."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CanonicalEntry:
    id: int
    text: str
    token_count: int


class CanonicalStore:
    """
    Maintains canonical prompt forms with FAISS approximate nearest-neighbor search.

    During warmup, the shortest-token prompt in each semantic cluster is chosen as canonical.
    """

    def __init__(self, embedding_dim: int = 384, inventory_path: str | Path | None = None):
        self.embedding_dim = embedding_dim
        self.inventory_path = Path(inventory_path) if inventory_path else None
        self._index = faiss.IndexFlatIP(embedding_dim)  # inner product on normalized vectors
        self._entries: list[CanonicalEntry] = []
        self._next_id = 0

        if self.inventory_path and self.inventory_path.exists():
            self.load()

    @property
    def size(self) -> int:
        return len(self._entries)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms

    def add(self, text: str, embedding: np.ndarray, token_count: int) -> CanonicalEntry:
        """Add a new canonical prompt to the inventory."""
        entry = CanonicalEntry(id=self._next_id, text=text, token_count=token_count)
        self._next_id += 1
        self._entries.append(entry)

        vec = self._normalize(embedding.reshape(1, -1).astype(np.float32))
        self._index.add(vec)
        return entry

    def search(
        self,
        embedding: np.ndarray,
        k: int = 1,
        threshold: float = 0.85,
    ) -> tuple[CanonicalEntry | None, float]:
        """
        Find nearest canonical prompt above similarity threshold.

        Returns (entry, similarity) or (None, best_similarity).
        """
        if self._index.ntotal == 0:
            return None, 0.0

        vec = self._normalize(embedding.reshape(1, -1).astype(np.float32))
        similarities, indices = self._index.search(vec, min(k, self._index.ntotal))

        best_sim = float(similarities[0][0])
        best_idx = int(indices[0][0])

        if best_sim >= threshold and best_idx >= 0:
            return self._entries[best_idx], best_sim
        return None, best_sim

    def maybe_add_shorter(
        self,
        text: str,
        embedding: np.ndarray,
        token_count: int,
        matched: CanonicalEntry | None,
        similarity: float,
        threshold: float,
    ) -> CanonicalEntry:
        """
        If a match exists but incoming prompt is shorter, replace canonical form.
        Otherwise register as new canonical if no match.
        """
        if matched is not None:
            if token_count < matched.token_count:
                matched.text = text
                matched.token_count = token_count
                logger.debug("Updated canonical %d with shorter form (%d tokens)", matched.id, token_count)
            return matched

        return self.add(text, embedding, token_count)

    def save(self, path: Path | None = None) -> None:
        path = Path(path or self.inventory_path or "data/canonical_inventory")
        path.mkdir(parents=True, exist_ok=True)

        entries_file = path / "entries.json"
        with open(entries_file, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

        if self._index.ntotal > 0:
            faiss.write_index(self._index, str(path / "index.faiss"))

        meta = {"embedding_dim": self.embedding_dim, "next_id": self._next_id}
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)

        logger.info("Saved %d canonical entries to %s", len(self._entries), path)

    def load(self, path: Path | None = None) -> None:
        path = Path(path or self.inventory_path)
        entries_file = path / "entries.json"
        index_file = path / "index.faiss"
        meta_file = path / "meta.json"

        if not entries_file.exists():
            return

        with open(entries_file) as f:
            raw = json.load(f)
        self._entries = [CanonicalEntry(**e) for e in raw]

        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            self.embedding_dim = meta.get("embedding_dim", self.embedding_dim)
            self._next_id = meta.get("next_id", len(self._entries))

        if index_file.exists():
            self._index = faiss.read_index(str(index_file))
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)

        logger.info("Loaded %d canonical entries from %s", len(self._entries), path)
