"""Tests for the embedding wrapper (Phase 1) — no model download required.

We inject a fake ``sentence_transformers`` module to verify the wrapper's
dimension contract without touching the network.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from guardbound.embeddings import PAPER_EMBEDDING_DIM, SentenceEmbedder


class FakeSTModel:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        return np.ones((len(texts), 768), dtype=np.float32)


def _install_fake_st(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda name: FakeSTModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)


def test_paper_embedding_dim_constant():
    assert PAPER_EMBEDDING_DIM == 768


def test_embed_returns_batch_times_768(monkeypatch):
    _install_fake_st(monkeypatch)
    emb = SentenceEmbedder("all-mpnet-base-v2")
    assert not emb.is_loaded
    vecs = emb.embed(["a", "b", "c"])
    assert vecs.shape == (3, 768)          # paper: n = 768
    assert vecs.dtype == np.float32
    assert emb.is_loaded                    # lazy load happened once


def test_embed_pair_helper(monkeypatch):
    _install_fake_st(monkeypatch)
    u_k, z_k = SentenceEmbedder("all-distilroberta-v1").embed_query_response("q", "r")
    assert u_k.shape == (768,)
    assert z_k.shape == (768,)


def test_dim_mismatch_raises(monkeypatch):
    class WrongDimModel:
        def encode(self, texts, **kw):
            return np.zeros((len(texts), 384), dtype=np.float32)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda name: WrongDimModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    emb = SentenceEmbedder("some-384d-model")
    with pytest.raises(ValueError, match="768"):
        emb.embed(["x"])
