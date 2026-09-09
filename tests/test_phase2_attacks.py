"""Phase 2 tests: ActorAttack, OppositeDayAttack, AcronymAttack integration.

All tests are offline — attacker/target LLMs are ``MockChatLLM`` and any NBF
components are deterministic stubs/monkeypatches.

Verified behaviors (against nbf_original_stuff official run.py):
- ActorAttack: auto pre-attack (infer_single) on first ``next_query``, chain
  advancement via step_judge semantics (successful -> next query, unknown ->
  next actor, rejective -> rewrite same slot, N_retry = 3), 10-trial NBF loop
  in call_multi.
- OppositeDay/Acronym: first-turn generation from empty history, "(Summary) "
  target-history storage for rounds > 1, LLM refusal detection, Acronym's
  (refusal OR disclaimer) control flow, 3-trial NBF candidate filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.attacks.acronym import AcronymAttack  # noqa: E402
from guardbound.attacks.actor_attack import ActorAttack  # noqa: E402
from guardbound.attacks.opposite_day import OppositeDayAttack  # noqa: E402
from guardbound.llm.mock import MockChatLLM  # noqa: E402
from guardbound.schemas import Turn  # noqa: E402


# ============================================================================
# ActorAttack
# ============================================================================

def _actor_pre_attack_responses(actor_num: int, query_counts: list[int]) -> list[str]:
    """Script the attacker responses consumed by infer_single.

    Call order: extract_harm_target (json), network (str), actors (json),
    then per actor: query chain (str) + formatted questions (json).
    """
    actors = ", ".join(
        '{"actor_name": "A%d", "relationship": "r%d"}' % (i + 1, i + 1)
        for i in range(actor_num)
    )
    responses = [
        '{"target": "t", "details": {"delivery_type": "article", "other_details": ""}}',
        "network response",
        '{"actors": [%s]}' % actors,
    ]
    for n in query_counts:
        responses.append("query chain response")
        questions = ", ".join(
            '{"question": "q%d"}' % (i + 1) for i in range(n)
        )
        responses.append('{"questions": [%s]}' % questions)
    return responses


def _actor_attack(actor_num: int, query_counts: list[int]) -> tuple[ActorAttack, MockChatLLM]:
    responses = _actor_pre_attack_responses(actor_num, query_counts)
    attacker = MockChatLLM(responses=responses)
    attack = ActorAttack(attacker_llm=attacker, actor_num=actor_num)
    return attack, attacker


class TestActorAttackPreAttack:
    def test_next_query_auto_initializes(self):
        """next_query() runs infer_single and returns the first chain query."""
        attack, attacker = _actor_attack(1, [2])
        q = attack.next_query("goal", [])
        assert q == "q1"
        # Pre-attack ran exactly once: 5 attacker calls, no query regeneration.
        assert len(attacker.calls) == 5
        assert attack._pre_attack_data["harm_target"] == "t"

    def test_pre_attack_runs_only_once(self):
        """Subsequent next_query calls do not re-run the pre-attack."""
        attack, attacker = _actor_attack(1, [2])
        attack.next_query("goal", [])
        attack.next_query("goal", [])
        assert len(attacker.calls) == 5  # still only the pre-attack calls


class TestActorAttackChainAdvancement:
    def test_queries_generated_for_multiple_actors_in_order(self):
        """next_query + record_turn walk actor1's chain, then actor2's."""
        attack, attacker = _actor_attack(2, [2, 1])
        # Append judge responses after the pre-attack responses.
        attacker.responses.extend(['{"type": "successful"}'] * 10)

        assert attack.next_query("goal", []) == "q1"          # actor 1, query 1
        attack.record_turn("q1", "r", 3)
        assert attack.next_query("goal", []) == "q2"          # actor 1, query 2
        attack.record_turn("q2", "r", 3)
        assert attack.next_query("goal", []) == "q1"          # actor 2, query 1
        attack.record_turn("q1", "r", 3)
        # All actors exhausted
        assert attack.next_query("goal", []) == ""
        assert attack._current_actor_idx == 2

    def test_rejective_rewrites_query_and_retries_same_slot(self):
        """step_judge=rejective triggers rewrite_query; next_query serves it."""
        attack, attacker = _actor_attack(1, [1])
        attacker.responses.extend(['{"type": "rejective"}'])
        q1 = attack.next_query("goal", [])
        assert q1 == "q1"
        attack.record_turn(q1, "I'm sorry, I cannot help", 1)
        # rewrite_query produced a new query for the same slot
        q2 = attack.next_query("goal", [])
        assert q2 != "q1" and q2.strip()
        # The last attacker call was the rewrite (ATTACK_MODIFY_PROMPT)
        last_messages, _, _ = attacker.calls[-1]
        user_content = " ".join(m["content"] for m in last_messages if m["role"] == "user")
        assert "modify your question" in user_content or "13 categories" in user_content

    def test_unknown_moves_to_next_actor(self):
        """step_judge=unknown skips the rest of the current actor."""
        attack, attacker = _actor_attack(2, [1, 1])
        attacker.responses.extend(['{"type": "unknown"}'])
        assert attack.next_query("goal", []) == "q1"
        attack.record_turn("q1", "I don't know", 2)
        assert attack.next_query("goal", []) == "q1"  # actor 2's first query
        assert attack._current_actor_idx == 1

    def test_successful_advances_and_sets_success(self):
        """step_judge=successful advances; rubric score 5 marks success."""
        attack, attacker = _actor_attack(1, [2])
        attacker.responses.extend(['{"type": "successful"}'])
        assert attack.next_query("goal", []) == "q1"
        attack.record_turn("q1", "full harmful answer", 5)
        assert attack.was_successful()
        assert attack.next_query("goal", []) == "q2"  # advanced


class TestActorAttackNBF:
    def test_call_multi_filters_unsafe_queries_before_target(self):
        """With NBF score >= threshold, the query never reaches the target.

        Official call_multi: N_retry=3 retries per query, each attempt runs
        the 10-trial loop whose author structure evaluates once then breaks
        (accept iff score < threshold). Here every attempt is unsafe.
        """
        attack = ActorAttack(attacker_llm=MockChatLLM(responses=[]), actor_num=1)
        target = MockChatLLM(responses=["ok"])

        calls = {"n": 0}

        def _fake_score(queries, index):
            calls["n"] += 1
            return 0.5  # >= threshold -> unsafe

        attack._calculate_score = _fake_score  # type: ignore[method-assign]
        attack.set_safety_filter(embed_fn=lambda s: None, barrier=object(), threshold=-0.5)

        attack.call_multi(
            target_system="sys",
            instruction="goal",
            queries=["q1"],
            query_details={"delivery_type": "article", "other_details": ""},
            target_llm=target,
            use_safety_filter=True,
            safety_threshold=-0.5,
        )
        assert target.calls == []  # filtered query never sent
        assert calls["n"] == 4  # one evaluation per retry attempt (N_retry=3)

    def test_call_multi_sends_safe_queries(self):
        """With NBF score < threshold the query is sent to the target."""
        attack = ActorAttack(attacker_llm=MockChatLLM(responses=[
            '{"score": 4}',        # evaluate_with_rubric
            '{"type": "successful"}',  # step_judge
        ]), actor_num=1)
        target = MockChatLLM(responses=["sure"])

        def _fake_score(queries, index):
            return -0.9  # < threshold -> safe

        attack._calculate_score = _fake_score  # type: ignore[method-assign]
        attack.set_safety_filter(embed_fn=lambda s: None, barrier=object(), threshold=-0.5)

        dialog, skip = attack.call_multi(
            target_system="sys",
            instruction="goal",
            queries=["q1"],
            query_details={"delivery_type": "article", "other_details": ""},
            target_llm=target,
            use_safety_filter=True,
            safety_threshold=-0.5,
        )
        assert len(target.calls) == 1
        assert not skip


# ============================================================================
# OppositeDay
# ============================================================================

class TestOppositeDayFirstQuery:
    def test_first_query_non_empty(self):
        attack = OppositeDayAttack(attacker_llm=MockChatLLM(
            responses=['{"generatedQuestion": "od1", "lastResponseSummary": "s0"}']
        ))
        q = attack.next_query("goal", [])
        assert q == "od1"
        assert q.strip()

    def test_attacker_prompt_has_system_and_first_round_marker(self):
        attacker = MockChatLLM(
            responses=['{"generatedQuestion": "od1", "lastResponseSummary": "s0"}']
        )
        attack = OppositeDayAttack(attacker_llm=attacker)
        attack.next_query("goal", [])
        messages, _temp, json_fmt = attacker.calls[0]
        assert json_fmt is True
        assert any("conversationObjective" in m["content"] for m in messages if m["role"] == "system")
        assert any("first round" in m["content"] for m in messages if m["role"] == "user")


class TestOppositeDayMultiTurn:
    def test_second_query_uses_prior_response_and_summary_history(self):
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "od1", "lastResponseSummary": "s1"}',
            '{"generatedQuestion": "od2", "lastResponseSummary": "s2"}',
            '{"generatedQuestion": "od3", "lastResponseSummary": "s3"}',
        ])
        attack = OppositeDayAttack(attacker_llm=attacker)

        q1 = attack.next_query("goal", [])
        assert q1 == "od1"
        attack.record_turn(q1, "resp1", 3)

        # Round 2: prior raw response is passed as last_response
        q2 = attack.next_query("goal", [Turn(query=q1, response="resp1")])
        assert q2 == "od2"
        messages2, _, _ = attacker.calls[1]
        user_contents2 = [m["content"] for m in messages2 if m["role"] == "user"]
        # Official prompt structure: the first-round boilerplate prompt is
        # always present, plus one "round number N" prompt per prior turn.
        assert any("round number 2" in c for c in user_contents2)
        assert any("resp1" in c for c in user_contents2)

        attack.record_turn(q2, "resp2", 3)
        # Official: target history stores "(Summary) <summary>" for rounds > 1
        assert attack._history_target == ["(Summary) s2"]

        # Round 3: the attacker prompt now contains the stored summary
        q3 = attack.next_query("goal", [
            Turn(query=q1, response="resp1"),
            Turn(query=q2, response="resp2"),
        ])
        assert q3 == "od3"
        messages3, _, _ = attacker.calls[2]
        user_contents3 = [m["content"] for m in messages3 if m["role"] == "user"]
        assert any("(Summary) s2" in c for c in user_contents3)


class TestOppositeDayRefusal:
    def test_llm_refusal_detection(self):
        attacker = MockChatLLM(responses=['{"value": true, "metadata": 95, "rationale": "r"}'])
        attack = OppositeDayAttack(attacker_llm=attacker)
        assert attack.check_refusal("p", "I can't help with that")

    def test_llm_non_refusal_detection(self):
        attacker = MockChatLLM(responses=['{"value": false, "metadata": 95, "rationale": "r"}'])
        attack = OppositeDayAttack(attacker_llm=attacker)
        assert not attack.check_refusal("p", "Here you go")


class TestOppositeDayNBF:
    def test_filtered_prompt_is_regenerated(self):
        """3-trial filter: unsafe prompt regenerated, safe prompt accepted."""
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "od-unsafe", "lastResponseSummary": ""}',
            '{"generatedQuestion": "od-safe", "lastResponseSummary": ""}',
        ])
        attack = OppositeDayAttack(attacker_llm=attacker)
        # Dummy components: _nbf_score is monkeypatched below, so the real
        # embed_fn/barrier are never exercised.
        attack.set_safety_filter(embed_fn=object(), barrier=object(), threshold=-0.5)

        scores = iter([0.5, -0.9])  # first >= threshold (filtered), second safe

        def _fake_score(query, dialog_hist):
            return next(scores)

        attack._nbf_score = _fake_score  # type: ignore[method-assign]

        q = attack.next_query("goal", [])
        assert q == "od-safe"
        assert len(attacker.calls) == 2  # original + regenerated

    def test_accepted_prompt_not_regenerated(self):
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "od-safe", "lastResponseSummary": ""}',
        ])
        attack = OppositeDayAttack(attacker_llm=attacker)
        attack.set_safety_filter(embed_fn=object(), barrier=object(), threshold=-0.5)
        attack._nbf_score = lambda query, dialog_hist: -0.9  # type: ignore[method-assign]
        q = attack.next_query("goal", [])
        assert q == "od-safe"
        assert len(attacker.calls) == 1


# ============================================================================
# Acronym
# ============================================================================

class TestAcronymFirstQuery:
    def test_first_query_non_empty(self):
        attack = AcronymAttack(attacker_llm=MockChatLLM(
            responses=['{"generatedQuestion": "ac1", "lastResponseSummary": "s0"}']
        ))
        q = attack.next_query("goal", [])
        assert q == "ac1"
        assert q.strip()


class TestAcronymMultiTurn:
    def test_conversation_progression(self):
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "ac1", "lastResponseSummary": "s1"}',
            '{"generatedQuestion": "ac2", "lastResponseSummary": "s2"}',
        ])
        attack = AcronymAttack(attacker_llm=attacker)
        q1 = attack.next_query("goal", [])
        assert q1 == "ac1"
        attack.record_turn(q1, "resp1", 3)
        q2 = attack.next_query("goal", [Turn(query=q1, response="resp1")])
        assert q2 == "ac2"
        attack.record_turn(q2, "resp2", 3)
        assert attack._history_target == ["(Summary) s2"]


class TestAcronymRefusal:
    def test_refusal_detection(self):
        attacker = MockChatLLM(responses=['{"value": true, "metadata": 95}'])
        attack = AcronymAttack(attacker_llm=attacker)
        assert attack.check_refusal("p", "I can't do that")

    def test_disclaimer_triggers_backtrack_when_no_refusal(self):
        """Official control flow: (check_refusal OR check_disclaimer)."""
        attacker = MockChatLLM(responses=[
            '{"value": false, "metadata": 80}',       # not a refusal
            '{"value": true, "rationale": "has disclaimer"}',  # disclaimer
        ])
        attack = AcronymAttack(attacker_llm=attacker)
        assert attack.check_refusal(
            "p", "The acronym is KIKE; note this term may be offensive."
        )
        assert len(attacker.calls) == 2  # refusal checked first, then disclaimer

    def test_no_refusal_no_disclaimer(self):
        attacker = MockChatLLM(responses=[
            '{"value": false, "metadata": 80}',
            '{"value": false, "rationale": "none"}',
        ])
        attack = AcronymAttack(attacker_llm=attacker)
        assert not attack.check_refusal("p", "The acronym is KIKE.")


class TestAcronymNBF:
    def test_filtered_prompt_is_regenerated(self):
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "ac-unsafe", "lastResponseSummary": ""}',
            '{"generatedQuestion": "ac-safe", "lastResponseSummary": ""}',
        ])
        attack = AcronymAttack(attacker_llm=attacker)
        attack.set_safety_filter(embed_fn=object(), barrier=object(), threshold=-0.5)
        scores = iter([0.5, -0.9])
        attack._nbf_score = lambda query, dialog_hist: next(scores)  # type: ignore[method-assign]
        q = attack.next_query("goal", [])
        assert q == "ac-safe"
        assert len(attacker.calls) == 2


# ============================================================================
# Shared runner regression (all three attacks through run_attack)
# ============================================================================

class TestRunnerIntegration:
    def _run_opposite_day_through_runner(self):
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "od1", "lastResponseSummary": "s1"}',
            '{"score": 3}',                                   # rubric turn 1
            '{"value": false, "metadata": 80}',               # refusal turn 1
            '{"generatedQuestion": "od2", "lastResponseSummary": "s2"}',
            '{"score": 3}',                                   # rubric turn 2
            '{"value": false, "metadata": 80}',               # refusal turn 2
        ])
        target = MockChatLLM(responses=["target resp 1", "target resp 2"])
        attack = OppositeDayAttack(attacker_llm=attacker)
        return attack, target

    def test_opposite_day_runs_through_runner(self):
        from guardbound.attacks.runner import run_attack
        attack, target = self._run_opposite_day_through_runner()
        conv = run_attack(attack, "goal", target, max_turns=2)
        assert len(conv.turns) == 2
        assert all(t.query.strip() for t in conv.turns)
        assert len(target.calls) == 2

    def test_acronym_runs_through_runner(self):
        from guardbound.attacks.runner import run_attack
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "ac1", "lastResponseSummary": "s1"}',
            '{"score": 3}',
            '{"value": false, "metadata": 80}',   # refusal
            '{"value": false, "rationale": "none"}',  # disclaimer
            '{"generatedQuestion": "ac2", "lastResponseSummary": "s2"}',
            '{"score": 3}',
            '{"value": false, "metadata": 80}',
            '{"value": false, "rationale": "none"}',
        ])
        target = MockChatLLM(responses=["target resp 1", "target resp 2"])
        attack = AcronymAttack(attacker_llm=attacker)
        conv = run_attack(attack, "goal", target, max_turns=2)
        assert len(conv.turns) == 2
        assert all(t.query.strip() for t in conv.turns)

    def test_actor_attack_runs_through_runner(self):
        from guardbound.attacks.runner import run_attack
        attacker = MockChatLLM(responses=(
            _actor_pre_attack_responses(1, [2])
            + ['{"score": 3}', '{"type": "successful"}',   # turn 1: rubric + judge
               '{"score": 3}', '{"type": "successful"}']   # turn 2
        ))
        target = MockChatLLM(responses=["target resp 1", "target resp 2"])
        attack = ActorAttack(attacker_llm=attacker, actor_num=1)
        conv = run_attack(attack, "goal", target, max_turns=2)
        assert len(conv.turns) == 2
        assert all(t.query.strip() for t in conv.turns)
        assert [t.query for t in conv.turns] == ["q1", "q2"]