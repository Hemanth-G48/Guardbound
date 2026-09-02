"""Embedding extraction with deterministic caching.

Processes conversations incrementally, embedding each query and response
into R^768. Supports both all-mpnet-base-v2 (default) and
all-distilroberta-v1. Embeddings are cached to avoid recomputation.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..embeddings import SentenceEmbedder, PAPER_EMBEDDING_DIM
from ..logging_utils import get_logger
from ..schemas import Conversation, Turn, load_conversations_jsonl

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #

class EmbeddingCache:
    """Deterministic cache for text embeddings keyed by (model, text).

    Vectors are stored as individual .npy files; the index maps keys to
    filenames.  This avoids np.savez key-name issues with special chars.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._vectors_dir = self.cache_dir / "vectors"
        self._vectors_dir.mkdir(exist_ok=True)
        self._index_file = self.cache_dir / "embedding_index.json"
        self._vectors: dict[str, np.ndarray] = {}
        self._index: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._index_file.exists():
            try:
                self._index = json.loads(self._index_file.read_text())
                for key, meta in self._index.items():
                    npy_path = self._vectors_dir / f"{key}.npy"
                    if npy_path.exists():
                        self._vectors[key] = np.load(npy_path)
                logger.info("Loaded %d cached embeddings", len(self._vectors))
            except Exception as exc:
                logger.warning("Could not load embedding cache: %s", exc)

    def _save(self) -> None:
        if self._vectors:
            for key, vec in self._vectors.items():
                npy_path = self._vectors_dir / f"{key}.npy"
                np.save(npy_path, vec)
            self._index_file.write_text(json.dumps(self._index, indent=2))

    @staticmethod
    def _key(model_name: str, text: str) -> str:
        payload = f"{model_name}||{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model_name: str, text: str) -> np.ndarray | None:
        key = self._key(model_name, text)
        return self._vectors.get(key)

    def put(self, model_name: str, text: str, vector: np.ndarray) -> None:
        key = self._key(model_name, text)
        self._vectors[key] = vector
        self._index[key] = {
            "model": model_name,
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text_preview": text[:80],
        }

    def save(self) -> None:
        """Explicitly save the cache to disk."""
        self._save()

    def __len__(self) -> int:
        return len(self._vectors)

    def __bool__(self) -> bool:
        """Cache object is always truthy; check len() for emptiness."""
        return True


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #

@dataclass
class EmbeddingResult:
    """Result of embedding a single conversation."""
    conversation_idx: int
    num_turns: int
    num_cached: int
    num_computed: int


class ConversationEmbedder:
    """Embed all turns in conversations using sentence-transformers.

    Processes conversations incrementally with caching to avoid recomputation.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        expected_dim: int = PAPER_EMBEDDING_DIM,
        batch_size: int = 64,
        cache_dir: Path | None = None,
    ):
        self.model_name = model_name
        self.expected_dim = expected_dim
        self.batch_size = batch_size
        self._embedder = SentenceEmbedder(model_name, expected_dim)
        self._cache = EmbeddingCache(cache_dir) if cache_dir else None

    def embed_conversations(
        self,
        conversations: list[Conversation],
        dry_run: bool = False,
    ) -> list[EmbeddingResult]:
        """Embed all turns in all conversations.

        Returns embedding results with cache statistics.
        """
        results = []

        for idx, conv in enumerate(conversations):
            if dry_run:
                logger.info("[DRY RUN] Would embed conversation %d (%d turns)", idx, len(conv.turns))
                results.append(EmbeddingResult(idx, len(conv.turns), 0, 0))
                continue

            num_cached = 0
            num_computed = 0

            # Collect texts that need embedding
            texts_to_embed: list[tuple[int, str, str]] = []  # (turn_idx, "query"|"response", text)

            for t_idx, turn in enumerate(conv.turns):
                if turn.query_embedding is None:
                    query_cached = False
                    if self._cache:
                        cached = self._cache.get(self.model_name, turn.query)
                        if cached is not None:
                            turn.query_embedding = cached
                            num_cached += 1
                            query_cached = True
                    if not query_cached:
                        texts_to_embed.append((t_idx, "query", turn.query))

                if turn.response and turn.response_embedding is None:
                    resp_cached = False
                    if self._cache:
                        cached = self._cache.get(self.model_name, turn.response)
                        if cached is not None:
                            turn.response_embedding = cached
                            num_cached += 1
                            resp_cached = True
                    if not resp_cached:
                        texts_to_embed.append((t_idx, "response", turn.response))

            # Batch embed remaining texts
            if texts_to_embed:
                all_texts = [t[2] for t in texts_to_embed]
                for batch_start in range(0, len(all_texts), self.batch_size):
                    batch_texts = all_texts[batch_start:batch_start + self.batch_size]
                    batch_vecs = self._embedder.embed(batch_texts)

                    for i, (t_idx, field, text) in enumerate(
                        texts_to_embed[batch_start:batch_start + self.batch_size]
                    ):
                        vec = batch_vecs[i]
                        turn = conv.turns[t_idx]
                        if field == "query":
                            turn.query_embedding = vec
                        else:
                            turn.response_embedding = vec
                        num_computed += 1

                        if self._cache:
                            self._cache.put(self.model_name, text, vec)

            results.append(EmbeddingResult(idx, len(conv.turns), num_cached, num_computed))

            # Save cache periodically (every 10 conversations) and always at the end
            if self._cache and ((idx + 1) % 10 == 0 or idx == len(conversations) - 1):
                self._cache.save()

            if (idx + 1) % 100 == 0:
                logger.info("Embedded %d/%d conversations", idx + 1, len(conversations))

        return results

    def embed_conversations_from_file(
        self,
        input_path: Path,
        dry_run: bool = False,
    ) -> list[EmbeddingResult]:
        """Load conversations from JSONL and embed them."""
        conversations = load_conversations_jsonl(input_path)
        logger.info("Loaded %d conversations from %s", len(conversations), input_path)
        return self.embed_conversations(conversations, dry_run=dry_run)
