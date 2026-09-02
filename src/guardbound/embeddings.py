"""Sentence-embedding wrapper (Phase 1 interface, used heavily from Phase 2).

Paper (Sec. 5.1): pretrained sentence embedding models ``all-mpnet-base-v2``
(default) and ``all-distilroberta-v1``; both map sentences to R^768.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

PAPER_EMBEDDING_DIM = 768  # n = 768 (paper)


class SentenceEmbedder:
    """Lazy-loading wrapper around sentence-transformers.

    Importable without the dependency installed; the model loads on first call
    to :meth:`embed`.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 expected_dim: int = PAPER_EMBEDDING_DIM):
        self.model_name = model_name
        self.expected_dim = expected_dim
        self._model = None  # lazy

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The 'sentence-transformers' package is required for "
                    f"{self.__class__.__name__}. Install it with: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed sentences -> float32 array of shape (batch, expected_dim)."""
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True,
                               show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
            raise ValueError(
                f"Expected embeddings of shape (B, {self.expected_dim}); got "
                f"{vectors.shape}. The paper requires 768-dim sentence embeddings."
            )
        return vectors

    def embed_query_response(self, query: str, response: str) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: embed one (u_k, z_k) pair."""
        out = self.embed([query, response])
        return out[0], out[1]


# --------------------------------------------------------------------------- #
# Convenience factories
# --------------------------------------------------------------------------- #

_MODEL_ALIASES = {
    "mpnet": "all-mpnet-base-v2",
    "distilroberta": "all-distilroberta-v1",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "all-distilroberta-v1": "all-distilroberta-v1",
}


def get_embed_fn(model: str = "mpnet") -> Callable[[str], "torch.Tensor"]:
    """Return a callable ``text -> torch.Tensor[1, 768]`` for a model alias.

    The returned function is the format expected by
    :class:`SteeredLLMChat` and the adaptive attack.
    """
    model_name = _resolve(model)

    embedder = SentenceEmbedder(model_name=model_name)

    def _fn(text: str) -> "torch.Tensor":
        import torch
        arr = embedder.embed([text])
        return torch.as_tensor(arr, dtype=torch.float32)

    return _fn


def resolve_embedding_model(name: str) -> str:
    """Resolve a short alias or full HF name to the canonical model name."""
    return _resolve(name)


def _resolve(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)
