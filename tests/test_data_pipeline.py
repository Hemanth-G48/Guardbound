"""Phase 2 data pipeline tests — all offline, no network access.

Tests cover: sources, attack runner, judge, embeddings, dataset packaging.
"""
from __future__ import annotations

import json
import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure src/ is importable
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.schemas import Conversation, Turn, save_conversations_jsonl, load_conversations_jsonl
from guardbound.llm.base import ChatLLM, DEFAULT_TEMPERATURE


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

class MockAttackLLM(ChatLLM):
    """Deterministic mock LLM for attack tests."""
    name = "mock-attack"

    def generate(self, messages, temperature=DEFAULT_TEMPERATURE, max_turns_context=None):
        # Return a response that includes the last user query (for testing)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"Mock response to: {last_user}"


class MockJudgeLLM(ChatLLM):
    """Deterministic mock LLM for judge tests."""
    name = "mock-judge"

    def __init__(self, scores=None):
        self.scores = list(scores or [3, 4, 5])
        self._idx = 0
        self.calls = []

    def generate(self, messages, temperature=DEFAULT_TEMPERATURE, max_turns_context=None):
        self.calls.append(messages)
        score = self.scores[self._idx % len(self.scores)]
        self._idx += 1
        return str(score)


def _sample_conversation(
    goal: str = "test goal",
    attack_method: str = "crescendo",
    num_turns: int = 3,
    with_embeddings: bool = False,
    with_judge_scores: bool = False,
) -> Conversation:
    """Create a sample conversation for testing."""
    turns = []
    for i in range(num_turns):
        t = Turn(
            query=f"Query {i} about {goal}",
            response=f"Response {i} about {goal}",
        )
        if with_embeddings:
            t.query_embedding = np.random.randn(768).astype(np.float32)
            t.response_embedding = np.random.randn(768).astype(np.float32)
        if with_judge_scores:
            t.judge_score = [1, 3, 5][i % 3]
        turns.append(t)
    return Conversation(
        goal=goal,
        attack_method=attack_method,
        target_llm="mock",
        turns=turns,
        max_turns=8,
    )


# --------------------------------------------------------------------------- #
# Source tests
# --------------------------------------------------------------------------- #

class TestSources:
    def test_contamination_check_exact(self):
        from guardbound.data.sources import check_contamination
        train = ["goal A", "goal B", "goal C"]
        test = ["goal B", "goal D", "goal E"]
        report = check_contamination(train, test)
        assert report.training_count == 3
        assert report.test_count == 3
        assert report.exact_overlap_count == 1

    def test_contamination_check_normalized(self):
        from guardbound.data.sources import check_contamination
        train = ["  Goal A  "]
        test = ["goal a"]
        report = check_contamination(train, test)
        assert report.normalized_overlap_count == 1

    def test_contamination_check_no_overlap(self):
        from guardbound.data.sources import check_contamination
        train = ["goal A"]
        test = ["goal B"]
        report = check_contamination(train, test)
        assert report.exact_overlap_count == 0
        assert report.normalized_overlap_count == 0

    def test_source_metadata_save_load(self, tmp_path):
        from guardbound.data.sources import SourceMetadata
        meta = SourceMetadata(
            source_name="test",
            source_type="test",
            total_records=100,
            selected_records=50,
        )
        path = tmp_path / "meta.json"
        meta.save(path)
        loaded = SourceMetadata.load(path)
        assert loaded.source_name == "test"
        assert loaded.selected_records == 50

    def test_deterministic_selection(self, tmp_path):
        import numpy as np
        goals = [f"goal_{i}" for i in range(50)]

        # Test that same seed produces same selection
        rng1 = np.random.RandomState(42)
        indices1 = rng1.choice(50, size=10, replace=False)
        selected1 = [goals[i] for i in sorted(indices1)]

        rng2 = np.random.RandomState(42)
        indices2 = rng2.choice(50, size=10, replace=False)
        selected2 = [goals[i] for i in sorted(indices2)]

        assert selected1 == selected2  # Deterministic


# --------------------------------------------------------------------------- #
# Attack runner tests
# --------------------------------------------------------------------------- #

class TestAttackRunner:
    """Phase 2 attack runner tests.

    NOTE: Phase 6 ships paper-strict attack stubs that implement only
    ``next_query``.  The Phase 2 ``AttackRunner`` uses the legacy
    ``generate(goal, llm, max_turns, temperature) -> Conversation``
    interface.  These tests therefore instantiate a tiny local
    legacy-interface attack so the Phase 2 runner is exercised
    independently of the (not-yet-pasted) Phase 6 implementations.
    """

    def _make_legacy_attack(self, name: str = "test_attack"):
        """Build a tiny legacy-interface attack class for Phase 2 runner tests."""
        from guardbound.data.attack_runner import MultiTurnAttack as LegacyABC
        from guardbound.schemas import Conversation, Turn

        def _gen(self, goal, target_llm, max_turns=8, temperature=DEFAULT_TEMPERATURE):
            turns = []
            messages = []
            for k in range(max_turns):
                q = f"echo:{goal}:{k}"
                messages.append({"role": "user", "content": q})
                r = target_llm.generate(messages, temperature=temperature)
                messages.append({"role": "assistant", "content": r})
                turns.append(Turn(query=q, response=r, was_filtered=False))
            return Conversation(
                goal=goal, attack_method=name,
                target_llm=getattr(target_llm, "name", "mock"),
                turns=turns, max_turns=max_turns,
            )

        cls = type(
            "_LegacyEcho",
            (LegacyABC,),
            {"name": name, "generate": _gen},
        )
        return cls()

    def test_attack_interface(self):
        from guardbound.attacks import get_attack
        # Phase 6 registry attacks are stubs (next_query only).
        # The legacy Phase 2 interface (generate) is owned by the
        # data.attack_runner module, not by the Phase 6 registry.
        from guardbound.data.attack_runner import MultiTurnAttack as LegacyABC
        attack = self._make_legacy_attack()
        assert isinstance(attack, LegacyABC)
        assert hasattr(attack, "generate")
        assert hasattr(attack, "name")
        # Phase 6 registry still works for new code
        for name in ["crescendo", "opposite_day", "actor_attack", "acronym"]:
            assert get_attack(name) is not None

    def test_unknown_attack_raises(self):
        from guardbound.attacks import get_attack
        with pytest.raises(ValueError, match="Unknown attack"):
            get_attack("nonexistent")

    def test_attack_generates_conversation(self):
        llm = MockAttackLLM()
        attack = self._make_legacy_attack(name="crescendo")
        conv = attack.generate("harmful goal", llm, max_turns=3)
        assert isinstance(conv, Conversation)
        assert conv.goal == "harmful goal"
        assert conv.attack_method == "crescendo"
        assert len(conv.turns) == 3
        for turn in conv.turns:
            assert turn.query
            assert turn.response is not None

    def test_attack_runner_cache(self, tmp_path):
        from guardbound.data.attack_runner import AttackRunner, _cache_key

        llm = MockAttackLLM()
        attack = self._make_legacy_attack()
        runner = AttackRunner(
            attack=attack,
            target_llm=llm,
            output_path=tmp_path / "output.jsonl",
            cache_dir=tmp_path / "cache",
            target_model="mock",
        )

        # Generate first time
        result = runner.generate_one("test goal")
        assert result.error is None
        assert not result.from_cache

        # Generate again - should be from cache
        runner2 = AttackRunner(
            attack=attack,
            target_llm=llm,
            output_path=tmp_path / "output.jsonl",
            cache_dir=tmp_path / "cache",
            target_model="mock",
        )
        result2 = runner2.generate_one("test goal")
        assert result2.from_cache

    def test_attack_runner_dry_run(self, tmp_path):
        from guardbound.data.attack_runner import AttackRunner
        from guardbound.attacks import get_attack

        llm = MockAttackLLM()
        attack = get_attack("crescendo")
        runner = AttackRunner(
            attack=attack,
            target_llm=llm,
            output_path=tmp_path / "output.jsonl",
            cache_dir=tmp_path / "cache",
            target_model="mock",
        )

        result = runner.generate_one("test goal", dry_run=True)
        assert result.conversation.goal == "test goal"
        assert not tmp_path.joinpath("output.jsonl").exists()

    def test_conversation_validation(self):
        from guardbound.data.attack_runner import AttackRunner
        # Valid conversation
        conv = Conversation(
            goal="test",
            attack_method="crescendo",
            target_llm="mock",
            turns=[Turn(query="q", response="r")],
        )
        assert AttackRunner._validate_conversation(conv)

        # Invalid: empty goal
        conv2 = Conversation(goal="", attack_method="crescendo", target_llm="mock",
                            turns=[Turn(query="q", response="r")])
        assert not AttackRunner._validate_conversation(conv2)

        # Invalid: no turns
        conv3 = Conversation(goal="test", attack_method="crescendo", target_llm="mock", turns=[])
        assert not AttackRunner._validate_conversation(conv3)


# --------------------------------------------------------------------------- #
# Judge tests
# --------------------------------------------------------------------------- #

class TestJudge:
    def test_parse_score_exact(self):
        from guardbound.data.judge import parse_judge_score
        assert parse_judge_score("5") == 5
        assert parse_judge_score("1") == 1
        assert parse_judge_score(" 3 ") == 3

    def test_parse_score_from_text(self):
        from guardbound.data.judge import parse_judge_score
        assert parse_judge_score("The score is 4") == 4
        assert parse_judge_score("Rating: 2") == 2

    def test_parse_score_invalid(self):
        from guardbound.data.judge import parse_judge_score
        assert parse_judge_score("invalid") is None
        assert parse_judge_score("") is None
        assert parse_judge_score("Score: 7") is None  # out of range

    def test_judge_cache_hit(self, tmp_path):
        from guardbound.data.judge import SafetyJudge, JudgeCache, compute_judge_cache_key

        cache = JudgeCache(tmp_path / "judge_cache")
        key = compute_judge_cache_key("gpt-4o", "abc", "goal", "query", "response")
        assert not cache.contains(key)

        from guardbound.data.judge import JudgeCacheEntry
        import time
        entry = JudgeCacheEntry(
            cache_key=key,
            judge_score=5,
            judge_model="gpt-4o",
            prompt_hash="abc",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        cache.put(entry)
        assert cache.contains(key)
        assert cache.get(key).judge_score == 5

    def test_judge_turn_with_cache(self, tmp_path):
        from guardbound.data.judge import SafetyJudge

        llm = MockJudgeLLM(scores=[5])
        judge = SafetyJudge(
            judge_llm=llm,
            judge_model="mock",
            prompt_path="configs/judge_prompt.txt",
            cache_dir=tmp_path / "judge_cache",
        )

        # First call
        score = judge.judge_turn("goal", "query", "response")
        assert score == 5
        first_call_count = len(llm.calls)

        # Second call - should hit cache
        score2 = judge.judge_turn("goal", "query", "response")
        assert score2 == 5
        assert len(llm.calls) == first_call_count  # No new API call

    def test_judge_rejects_invalid(self):
        from guardbound.data.judge import SafetyJudge

        class BadLLM(ChatLLM):
            name = "bad"
            def generate(self, messages, temperature=0.7, max_turns_context=None):
                return "Score: invalid"

        judge = SafetyJudge(judge_llm=BadLLM(), judge_model="bad")
        score = judge.judge_turn("goal", "query", "response")
        assert score is None

    def test_judge_conversation(self):
        from guardbound.data.judge import SafetyJudge

        llm = MockJudgeLLM(scores=[1, 3, 5])
        judge = SafetyJudge(judge_llm=llm, judge_model="mock")

        conv = _sample_conversation(num_turns=3)
        scores = judge.judge_conversation(conv)
        assert scores == [1, 3, 5]

        # Note: judge_conversation returns scores but does not modify turns.
        # The caller (judge CLI) applies scores to turns.

    def test_judge_dry_run(self):
        from guardbound.data.judge import SafetyJudge

        llm = MockJudgeLLM(scores=[5])
        judge = SafetyJudge(judge_llm=llm, judge_model="mock")

        score = judge.judge_turn("goal", "query", "response", dry_run=True)
        assert score is None
        assert len(llm.calls) == 0  # No API call in dry run


# --------------------------------------------------------------------------- #
# Embedding tests
# --------------------------------------------------------------------------- #

class TestEmbedding:
    def _install_fake_st(self, monkeypatch):
        """Install fake sentence_transformers module."""
        fake = types.ModuleType("sentence_transformers")

        class FakeModel:
            def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
                return np.random.randn(len(texts), 768).astype(np.float32)

        fake.SentenceTransformer = lambda name: FakeModel()
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    def test_embedding_cache(self, tmp_path):
        from guardbound.data.embedder import EmbeddingCache
        cache = EmbeddingCache(tmp_path / "emb_cache")

        vec = np.random.randn(768).astype(np.float32)
        cache.put("model", "hello world", vec)

        retrieved = cache.get("model", "hello world")
        assert retrieved is not None
        assert np.allclose(retrieved, vec)

        # Different text returns None
        assert cache.get("model", "different text") is None

    def test_embedding_cache_persistence(self, tmp_path):
        from guardbound.data.embedder import EmbeddingCache
        cache_dir = tmp_path / "emb_cache"

        vec = np.random.randn(768).astype(np.float32)
        cache1 = EmbeddingCache(cache_dir)
        cache1.put("model", "hello", vec)
        cache1.save()

        # Reload
        cache2 = EmbeddingCache(cache_dir)
        retrieved = cache2.get("model", "hello")
        assert retrieved is not None
        assert np.allclose(retrieved, vec)

    def test_embed_conversations(self, monkeypatch, tmp_path):
        from guardbound.data.embedder import ConversationEmbedder
        self._install_fake_st(monkeypatch)

        embedder = ConversationEmbedder(
            model_name="all-mpnet-base-v2",
            cache_dir=tmp_path / "emb_cache",
        )

        # Use different goals so text differs across conversations
        convs = [_sample_conversation(goal=f"goal_{i}", num_turns=2) for i in range(3)]
        results = embedder.embed_conversations(convs)

        assert len(results) == 3
        for r in results:
            assert r.num_computed == 4  # 2 turns × 2 (query + response)
            assert r.num_cached == 0

        # Verify embeddings are set
        for conv in convs:
            for turn in conv.turns:
                assert turn.query_embedding is not None
                assert turn.response_embedding is not None
                assert turn.query_embedding.shape == (768,)
                assert turn.response_embedding.shape == (768,)

    def test_embed_cache_reuse(self, monkeypatch, tmp_path):
        from guardbound.data.embedder import ConversationEmbedder
        self._install_fake_st(monkeypatch)

        cache_dir = tmp_path / "emb_cache"
        convs = [_sample_conversation(num_turns=2)]

        # First pass
        embedder1 = ConversationEmbedder(model_name="all-mpnet-base-v2", cache_dir=cache_dir)
        results1 = embedder1.embed_conversations(convs)
        total_computed = sum(r.num_computed for r in results1)

        # Second pass with new conversations having same text
        convs2 = [_sample_conversation(num_turns=2)]  # same text
        embedder2 = ConversationEmbedder(model_name="all-mpnet-base-v2", cache_dir=cache_dir)
        results2 = embedder2.embed_conversations(convs2)
        total_cached = sum(r.num_cached for r in results2)

        assert total_cached == total_computed  # All from cache


# --------------------------------------------------------------------------- #
# Dataset packaging tests
# --------------------------------------------------------------------------- #

class TestDataset:
    def test_package_dataset_shapes(self, tmp_path):
        from guardbound.data.dataset import package_dataset

        convs = [_sample_conversation(
            num_turns=3, with_embeddings=True, with_judge_scores=True
        ) for _ in range(10)]

        manifest = package_dataset(
            conversations=convs,
            output_dir=tmp_path / "dataset",
            embedding_model="test-model",
            max_turns=8,
            embedding_dim=768,
            train_ratio=0.9,
            seed=42,
        )

        # Check manifest
        assert manifest.num_conversations == 10
        assert manifest.train_count + manifest.validation_count == 10
        assert manifest.embedding_dim == 768

        # Check npz files
        train_data = np.load(tmp_path / "dataset" / "train.npz")
        val_data = np.load(tmp_path / "dataset" / "val.npz")

        assert train_data["U"].dtype == np.float32
        assert train_data["Z"].dtype == np.float32
        assert train_data["Y"].dtype == np.int64
        assert train_data["MASK"].dtype == bool

        assert train_data["U"].shape[2] == 768
        assert train_data["Z"].shape[2] == 768

    def test_mask_semantics(self, tmp_path):
        from guardbound.data.dataset import package_dataset

        # Create conversations with different turn counts
        convs = []
        for i in range(5):
            convs.append(_sample_conversation(
                num_turns=i + 1,  # 1 to 5 turns
                with_embeddings=True,
                with_judge_scores=True,
            ))

        manifest = package_dataset(
            conversations=convs,
            output_dir=tmp_path / "dataset",
            max_turns=8,
        )

        train_data = np.load(tmp_path / "dataset" / "train.npz")
        MASK = train_data["MASK"]

        # Conversations have 1-5 turns; turns beyond each conversation's length
        # should be masked (False). Verify the mask shape and that the first
        # turn of every conversation is always True.
        assert MASK.shape[1] == 8  # max_turns
        assert MASK[:, 0].all()  # every conversation has at least 1 turn
        # Turn 5 (index 4) should NOT be True for conversations with <5 turns
        assert not MASK[:, 4].all()

    def test_label_validation(self, tmp_path):
        from guardbound.data.dataset import package_dataset

        convs = [_sample_conversation(
            num_turns=2, with_embeddings=True, with_judge_scores=True
        )]
        manifest = package_dataset(
            conversations=convs,
            output_dir=tmp_path / "dataset",
            max_turns=8,
        )

        train_data = np.load(tmp_path / "dataset" / "train.npz")
        Y = train_data["Y"]
        MASK = train_data["MASK"]
        valid_labels = Y[MASK]
        assert all(1 <= l <= 5 for l in valid_labels)

    def test_train_val_split_reproducible(self, tmp_path):
        from guardbound.data.dataset import package_dataset

        convs = [_sample_conversation(
            num_turns=2, with_embeddings=True, with_judge_scores=True
        ) for _ in range(20)]

        # Run twice with same seed
        manifest1 = package_dataset(
            conversations=convs, output_dir=tmp_path / "d1",
            max_turns=8, seed=42,
        )
        manifest2 = package_dataset(
            conversations=convs, output_dir=tmp_path / "d2",
            max_turns=8, seed=42,
        )

        assert manifest1.train_count == manifest2.train_count
        assert manifest1.validation_count == manifest2.validation_count

    def test_empty_conversations(self, tmp_path):
        from guardbound.data.dataset import package_dataset
        manifest = package_dataset(
            conversations=[],
            output_dir=tmp_path / "dataset",
        )
        assert manifest.num_conversations == 0

    def test_manifest_json_roundtrip(self, tmp_path):
        from guardbound.data.dataset import DatasetManifest
        manifest = DatasetManifest(
            embedding_model="test",
            num_conversations=10,
            attack_counts={"crescendo": 5, "acronym": 5},
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = DatasetManifest.load(path)
        assert loaded.num_conversations == 10
        assert loaded.attack_counts == {"crescendo": 5, "acronym": 5}


# --------------------------------------------------------------------------- #
# Offline miniature pipeline test
# --------------------------------------------------------------------------- #

class TestMiniaturePipeline:
    """End-to-end offline pipeline test using mocks."""

    def test_full_offline_pipeline(self, tmp_path):
        from guardbound.data.attack_runner import AttackRunner, MultiTurnAttack as LegacyABC
        from guardbound.data.judge import SafetyJudge
        from guardbound.data.embedder import ConversationEmbedder
        from guardbound.data.dataset import package_dataset
        from guardbound.schemas import Turn

        # 1. Generate conversations with mock LLM using a local
        #    legacy-interface attack (Phase 6 stubs are paper-strict
        #    and do not implement the Phase 2 generate() loop).
        llm = MockAttackLLM()

        class _LegacyEcho(LegacyABC):
            name = "crescendo"
            def generate(self, goal, target_llm, max_turns=8, temperature=DEFAULT_TEMPERATURE):
                from guardbound.schemas import Conversation
                turns = []
                messages = []
                for k in range(max_turns):
                    q = f"echo:{goal}:{k}"
                    messages.append({"role": "user", "content": q})
                    r = target_llm.generate(messages, temperature=temperature)
                    messages.append({"role": "assistant", "content": r})
                    turns.append(Turn(query=q, response=r, was_filtered=False))
                return Conversation(
                    goal=goal, attack_method="crescendo",
                    target_llm=getattr(target_llm, "name", "mock"),
                    turns=turns, max_turns=max_turns,
                )
        attack = _LegacyEcho()

        runner = AttackRunner(
            attack=attack, target_llm=llm,
            output_path=tmp_path / "conversations.jsonl",
            cache_dir=tmp_path / "cache" / "gen",
            target_model="mock", max_turns=3,
        )

        goals = ["goal_1", "goal_2", "goal_3"]
        results = runner.generate_batch(goals)
        assert all(r.error is None for r in results)
        assert len(results) == 3

        # 2. Judge conversations
        judge_llm = MockJudgeLLM(scores=[1, 2, 3, 4, 5, 1, 2, 3, 4])
        judge = SafetyJudge(
            judge_llm=judge_llm, judge_model="mock",
            prompt_path="configs/judge_prompt.txt",
            cache_dir=tmp_path / "cache" / "judge",
        )

        from guardbound.schemas import load_conversations_jsonl
        conversations = load_conversations_jsonl(tmp_path / "conversations.jsonl")

        for conv in conversations:
            scores = judge.judge_conversation(conv)
            for turn, score in zip(conv.turns, scores):
                turn.judge_score = score

        # 3. Embed conversations
        fake_st = types.ModuleType("sentence_transformers")
        class FakeModel:
            def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
                return np.random.randn(len(texts), 768).astype(np.float32)
        fake_st.SentenceTransformer = lambda name: FakeModel()
        sys.modules["sentence_transformers"] = fake_st

        embedder = ConversationEmbedder(
            model_name="all-mpnet-base-v2",
            cache_dir=tmp_path / "cache" / "embeddings",
        )
        embedder.embed_conversations(conversations)

        # Save embedded conversations
        save_path = tmp_path / "embedded.jsonl"
        save_conversations_jsonl(conversations, save_path)

        # 4. Build dataset
        manifest = package_dataset(
            conversations=conversations,
            output_dir=tmp_path / "dataset",
            embedding_model="all-mpnet-base-v2",
            max_turns=8,
            embedding_dim=768,
        )

        # Verify outputs
        assert manifest.num_conversations == 3
        assert manifest.train_count + manifest.validation_count == 3
        assert manifest.embedding_dim == 768

        train_data = np.load(tmp_path / "dataset" / "train.npz")
        assert "U" in train_data
        assert "Z" in train_data
        assert "Y" in train_data
        assert "MASK" in train_data

        # Clean up fake module
        del sys.modules["sentence_transformers"]
