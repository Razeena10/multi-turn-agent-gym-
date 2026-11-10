"""Wave 6: Comprehensive Testing and Validation.

Covers edge cases, boundary conditions, and integration scenarios not already
exercised by Waves 1–5. All 322 existing tests are preserved unchanged.

Sections:
    1. Environment edge cases
    2. Agent robustness
    3. Reward validation — progress-aware penalty, non-ASK repeats, APOLOGIZE
    4. Evaluator edge cases and validation
    5. Metrics edge cases and boundary conditions
    6. Integration: 100-episode full run with all verifications
    7. Code quality and reproducibility
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from src import config
from src.baseline_agent import BaselineSupportAgent
from src.evaluator import evaluate_agent
from src.metrics import compute_metrics, compute_scenario_breakdown
from src.support_env import CustomerSupportEnv
from src.support_scenarios import (
    SCENARIOS,
    NUM_SCENARIOS,
    ConversationState,
    UserStateMachine,
    init_state,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _env_at_scenario(scenario_idx: int, seed_start: int = 42) -> tuple:
    """Return (env, obs, info) with the env seeded to a specific scenario."""
    env = CustomerSupportEnv()
    for seed in range(seed_start, seed_start + 500):
        obs, info = env.reset(seed=seed)
        if info["scenario_id"] == scenario_idx:
            return env, obs, info
    raise RuntimeError(f"Could not land on scenario {scenario_idx}")


def _run_full_episode(
    env: CustomerSupportEnv,
    agent: BaselineSupportAgent,
    seed: int,
) -> tuple[float, int, str]:
    """Run one episode; return (total_reward, turns, termination_reason)."""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    total = 0.0
    turns = 0
    while True:
        action = agent.select_action(obs)
        obs, r, term, trunc, info = env.step(action)
        total += r
        turns += 1
        if term or trunc:
            return total, turns, info.get("termination_reason", "unknown")


# ===========================================================================
# 1. Environment Edge Cases
# ===========================================================================


class TestEnvEdgeCases:
    """Edge cases not covered by existing environment tests."""

    # --- Turn boundary ---

    def test_first_step_turn_count_is_1(self):
        """Turn count is exactly 1 after the very first step."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(config.ASK_ACCOUNT_INFO)
        assert info["turn_count"] == 1

    def test_turn_count_at_max_turns_minus_1_not_truncated(self):
        """One turn before the limit does not truncate."""
        env = CustomerSupportEnv(max_turns=3)
        env.reset(seed=42)
        env.step(config.ASK_ACCOUNT_INFO)  # turn 1
        _, _, term, trunc, _ = env.step(config.ASK_ACCOUNT_INFO)  # turn 2
        assert trunc is False  # not yet at max_turns

    def test_max_turns_1_truncates_immediately(self):
        """max_turns=1 truncates after a single step."""
        env = CustomerSupportEnv(max_turns=1)
        env.reset(seed=42)
        _, _, term, trunc, info = env.step(config.ASK_ACCOUNT_INFO)
        assert trunc is True
        assert term is False
        assert info["termination_reason"] == "truncated"

    def test_resolution_on_final_turn_not_truncated(self):
        """Resolving on the last allowed turn is 'resolved', not 'truncated'.

        Deterministic setup:
        - Scenario 0 (Internet Connection Problem) requires exactly 5 turns to
          resolve optimally: 3 info-gathering steps + DIAGNOSE + PROPOSE_SOLUTION.
        - max_turns is set to 5 — the exact number of turns needed.
        - The episode must therefore resolve on turn 5, which is simultaneously the
          final permitted turn.
        - The environment checks resolution/escalation BEFORE truncation, so the
          episode must return terminated=True, truncated=False, reason='resolved'.
        - If the implementation incorrectly applied truncation first, this test
          would fail because truncated would be True and terminated would be False.
        """
        # Find a seed that lands on scenario 0
        scenario_idx = 0  # Internet Connection Problem — 3 fields, 5 turns
        turns_to_resolve = len(SCENARIOS[scenario_idx].required_fields) + 2  # 5
        env = CustomerSupportEnv(max_turns=turns_to_resolve)
        agent = BaselineSupportAgent()

        target_seed = None
        for s in range(42, 600):
            _, info = env.reset(seed=s)
            if info["scenario_id"] == scenario_idx:
                target_seed = s
                break

        assert target_seed is not None, (
            "Could not find a seed that produces scenario 0 within 600 attempts"
        )

        # Run the episode with the deterministic seed
        obs, info = env.reset(seed=target_seed)
        agent.reset()
        assert info["scenario_id"] == scenario_idx

        final_term = False
        final_trunc = False
        final_info: dict = {}
        turns_taken = 0

        for _ in range(turns_to_resolve):
            action = agent.select_action(obs)
            obs, reward, final_term, final_trunc, final_info = env.step(action)
            turns_taken += 1
            if final_term or final_trunc:
                break

        # The episode must have ended exactly on turn 5
        assert turns_taken == turns_to_resolve, (
            f"Expected resolution on turn {turns_to_resolve}, "
            f"but episode ended on turn {turns_taken}"
        )

        # Unconditional assertions — must all be true for the test to pass
        assert final_term is True, (
            "terminated must be True: episode resolved on the final turn"
        )
        assert final_trunc is False, (
            "truncated must be False: resolution takes priority over truncation"
        )
        assert final_info.get("termination_reason") == "resolved", (
            f"termination_reason must be 'resolved', got "
            f"'{final_info.get('termination_reason')}'"
        )
        assert final_info.get("success") is True, (
            "success must be True for a resolved episode"
        )

    # --- Invalid action edge cases ---

    def test_action_exactly_num_actions_is_invalid(self):
        """Action == NUM_ACTIONS (one past the max) is invalid."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(config.NUM_ACTIONS)

    def test_action_float_raises(self):
        """Float action raises ValueError."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        with pytest.raises(ValueError, match="must be an integer"):
            env.step(3.0)

    def test_action_none_raises(self):
        """None action raises ValueError."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        with pytest.raises((ValueError, TypeError)):
            env.step(None)

    # --- Reset edge cases ---

    def test_reset_after_truncation_works(self):
        """reset() after truncation returns a valid observation."""
        env = CustomerSupportEnv(max_turns=1)
        env.reset(seed=42)
        env.step(config.ASK_ACCOUNT_INFO)  # truncates
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert info["turn_count"] == 0

    def test_reset_returns_turn_0(self):
        """Turn count in info is 0 after reset."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        for _ in range(3):
            env.step(config.ASK_ACCOUNT_INFO)
        _, info = env.reset(seed=99)
        assert info["turn_count"] == 0

    def test_step_without_reset_raises(self):
        """step() before any reset raises RuntimeError."""
        env = CustomerSupportEnv()
        with pytest.raises(RuntimeError, match="Episode has ended"):
            env.step(config.ASK_ACCOUNT_INFO)

    # --- Observation validity ---

    def test_frustration_clamped_at_zero_after_apology(self):
        """Frustration never goes below 0.0 after apology at near-zero."""
        env, obs, _ = _env_at_scenario(0)
        env._state.frustration_level = 0.05
        env.step(config.APOLOGIZE)
        # Frustration delta is -0.3, clamped to 0.0
        assert env._state.frustration_level == pytest.approx(0.0)

    def test_frustration_clamped_at_one(self):
        """Frustration never exceeds 1.0 even with stacked increases."""
        env, _, _ = _env_at_scenario(0)
        env._state.frustration_level = 0.95
        # Trigger irrelevant action which adds 0.15 frustration
        env.step(config.ASK_BILLING_INFO)  # irrelevant for scenario 0
        assert env._state.frustration_level <= 1.0

    def test_collected_information_all_zeros_at_reset(self):
        """collected_information is all-zero immediately after reset."""
        env = CustomerSupportEnv()
        obs, _ = env.reset(seed=42)
        assert all(v == 0 for v in obs["collected_information"])

    def test_missing_information_all_ones_at_reset(self):
        """missing_information is all-one (or matches required fields) at reset."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        scenario = SCENARIOS[info["scenario_id"]]
        n_fields = len(scenario.required_fields)
        # First n_fields positions in missing should be 1
        for i in range(n_fields):
            assert obs["missing_information"][i] == 1

    def test_obs_and_info_consistent_scenario_id(self):
        """Scenario ID in obs matches scenario ID in info after reset."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert int(obs["problem_type"]) == info["scenario_id"]

    # --- Invalid max_turns ---

    def test_invalid_max_turns_string(self):
        """String max_turns raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            CustomerSupportEnv(max_turns="five")

    def test_invalid_max_turns_none(self):
        """None max_turns raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            CustomerSupportEnv(max_turns=None)


# ===========================================================================
# 2. Agent Robustness
# ===========================================================================


class TestAgentRobustness:
    """Agent edge cases and boundary conditions not in existing tests."""

    # --- Frustration exact boundary ---

    def test_frustration_exactly_high_threshold_no_apologize(self):
        """frustration == FRUSTRATION_HIGH_THRESHOLD (0.7) does NOT apologize.

        The agent's policy check is `frustration > FRUSTRATION_HIGH_THRESHOLD`
        (strict greater-than, not >=). At exactly 0.7, the condition is False,
        so the agent proceeds to the next priority level (gather info) rather
        than apologising. This boundary test confirms the strict > semantics
        and guards against accidental >= comparisons in future refactors.
        """
        agent = BaselineSupportAgent()
        obs = {
            "customer_issue": np.int64(0),
            "collected_information": np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8),
            "missing_information": np.ones(config.MAX_INFO_FIELDS, dtype=np.int8),
            "turn_count": np.int64(1),
            "frustration": np.array(config.FRUSTRATION_HIGH_THRESHOLD, dtype=np.float32),
            "previous_customer_response": np.int64(config.MSG_INFO_PROVIDED),
            "available_actions": np.ones(config.NUM_ACTIONS, dtype=np.int8),
            "problem_type": np.int64(0),
        }
        action = agent.select_action(obs)
        # At exactly 0.7 (not above), the agent must NOT apologize
        assert action != config.APOLOGIZE, (
            f"Agent apologized at frustration=={config.FRUSTRATION_HIGH_THRESHOLD}, "
            "but the check is strictly > so it should not apologize at the boundary"
        )

    def test_frustration_zero_after_reset(self):
        """Agent can handle frustration=0.0 without errors."""
        agent = BaselineSupportAgent()
        env, obs, _ = _env_at_scenario(0)
        assert float(obs["frustration"]) == pytest.approx(0.0)
        action = agent.select_action(obs)
        assert 0 <= action < config.NUM_ACTIONS

    # --- All keys must be present ---

    @pytest.mark.parametrize("missing_key", list(BaselineSupportAgent._OBSERVATION_KEYS))
    def test_each_required_key_missing_raises(self, missing_key: str):
        """Removing any single required key raises KeyError."""
        agent = BaselineSupportAgent()
        env, obs, _ = _env_at_scenario(0)
        del obs[missing_key]
        with pytest.raises(KeyError):
            agent.select_action(obs)

    # --- Scenario-specific action paths ---

    def test_agent_asks_billing_for_billing_scenario(self):
        """For Billing Issue (id=4), agent first asks ASK_ACCOUNT_INFO."""
        agent = BaselineSupportAgent()
        env, obs, info = _env_at_scenario(4)
        assert info["scenario_id"] == 4
        action = agent.select_action(obs)
        # First field is account_id → ASK_ACCOUNT_INFO
        assert action == config.ASK_ACCOUNT_INFO

    def test_agent_asks_technical_for_router_scenario(self):
        """For Router Malfunction (id=1), agent asks ASK_ACCOUNT_INFO first."""
        agent = BaselineSupportAgent()
        env, obs, info = _env_at_scenario(1)
        action = agent.select_action(obs)
        # First required field is account_id
        assert action == config.ASK_ACCOUNT_INFO

    # --- Reset between episodes is idempotent ---

    def test_double_reset_is_safe(self):
        """Calling reset() twice does not raise and leaves clean state."""
        agent = BaselineSupportAgent()
        agent._has_diagnosed = True
        agent.reset()
        agent.reset()
        assert agent._has_diagnosed is False
        assert agent._has_proposed is False

    # --- Action selection returns an int, not numpy int ---

    def test_select_action_returns_plain_int(self):
        """select_action returns a plain Python int."""
        agent = BaselineSupportAgent()
        env, obs, _ = _env_at_scenario(0)
        action = agent.select_action(obs)
        assert isinstance(action, int)


# ===========================================================================
# 3. Reward Validation — Progress-Aware Penalty
# ===========================================================================


class TestRewardValidation:
    """Comprehensive reward tests covering all progress-aware scenarios."""

    # --- DIAGNOSE repeated: does it get penalised? ---

    def test_repeated_diagnose_no_progress_penalised(self):
        """Repeated DIAGNOSE (no info gained) triggers repeat penalty."""
        env, obs, info = _env_at_scenario(0)
        # First DIAGNOSE (incomplete info)
        _, r1, _, _, _ = env.step(config.DIAGNOSE)
        # Second DIAGNOSE (still no info progress — diagnosed=True already)
        _, r2, _, _, _ = env.step(config.DIAGNOSE)
        # r2 should include the repeated-action penalty
        # diagnose gathers no info → penalty fires on 2nd consecutive
        assert r2 == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_REPEATED_ACTION
        )

    # --- ESCALATE repeated: does it terminate before repeating? ---

    def test_escalate_terminates_before_repeat(self):
        """ESCALATE terminates the episode so repeat penalty never fires."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, r, term, _, _ = env.step(config.ESCALATE)
        assert term is True
        # Can't repeat — episode is over

    # --- APOLOGIZE multiple times: always exempt ---

    def test_apologize_repeated_never_penalised(self):
        """APOLOGIZE can be repeated indefinitely without penalty."""
        env, _, _ = _env_at_scenario(0)
        env._state.frustration_level = 0.8
        # Step 1 apologize
        _, r1, _, _, _ = env.step(config.APOLOGIZE)
        env._state.frustration_level = 0.8  # reset frustration manually
        # Step 2 apologize (same action, but exempt)
        _, r2, _, _, _ = env.step(config.APOLOGIZE)
        # r2 should be only turn_penalty + apologize_high, no repeat penalty
        assert r2 == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_APOLOGIZE_HIGH_FRUSTRATION
        )
        env._state.frustration_level = 0.8
        # Step 3 apologize
        _, r3, _, _, _ = env.step(config.APOLOGIZE)
        assert r3 == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_APOLOGIZE_HIGH_FRUSTRATION
        )

    # --- Escalate reward exact value ---

    def test_escalate_reward_exact(self):
        """Escalation reward is exactly turn_penalty + escalate."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(config.ESCALATE)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_ESCALATE
        )

    # --- Resolution reward exact value ---

    def test_resolution_reward_exact(self):
        """Resolution reward is exactly turn_penalty + resolve."""
        env, obs, info = _env_at_scenario(0)
        agent = BaselineSupportAgent()
        agent.reset()
        for _ in range(20):
            action = agent.select_action(obs)
            obs, reward, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        assert term is True
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_RESOLVE
        )

    # --- Progress-aware: 3-field same-action scenario ---

    def test_scenario_3_three_consecutive_ask_no_penalty(self):
        """All 3 ASK_ACCOUNT_INFO calls in Password Reset get +0.5 (no penalty)."""
        env, obs, info = _env_at_scenario(3)
        assert info["scenario_id"] == 3
        # Turn 1: gathers account_id
        _, r1, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert r1 == pytest.approx(config.REWARD_TURN_PENALTY + config.REWARD_INFO_GATHERED)
        # Turn 2: gathers email (productive repeat)
        _, r2, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert r2 == pytest.approx(config.REWARD_TURN_PENALTY + config.REWARD_INFO_GATHERED)
        # Turn 3: gathers security_question (still productive)
        _, r3, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert r3 == pytest.approx(config.REWARD_TURN_PENALTY + config.REWARD_INFO_GATHERED)

    # --- Unproductive repeat of ASK after exhausting all fields ---

    def test_scenario_0_ask_account_after_exhausted_penalised(self):
        """ASK_ACCOUNT_INFO on scenario 0 after account_id gathered is penalised."""
        env, _, info = _env_at_scenario(0)
        env.step(config.ASK_ACCOUNT_INFO)   # gathers account_id
        _, r2, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)  # no more account fields
        assert r2 == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_REPEATED_ACTION
        )

    # --- Cumulative reward across a full optimal episode ---

    def test_optimal_episode_cumulative_reward(self):
        """Full optimal episode (5 turns, 3 fields) totals +10.5."""
        for scenario_idx in range(NUM_SCENARIOS):
            env, obs, info = _env_at_scenario(scenario_idx)
            agent = BaselineSupportAgent()
            agent.reset()
            total = 0.0
            for _ in range(20):
                action = agent.select_action(obs)
                obs, r, term, trunc, _ = env.step(action)
                total += r
                if term or trunc:
                    break
            assert total == pytest.approx(10.5), (
                f"Scenario {scenario_idx} ({SCENARIOS[scenario_idx].name}): "
                f"expected +10.5, got {total:.4f}"
            )

    # --- Truncation reward includes REWARD_TRUNCATE ---

    def test_truncation_reward_includes_truncate_penalty(self):
        """Truncation reward includes REWARD_TRUNCATE (-5.0)."""
        env = CustomerSupportEnv(max_turns=1)
        env.reset(seed=42)
        # DIAGNOSE gathers no info → reward = turn_penalty + truncate_penalty
        _, reward, term, trunc, _ = env.step(config.DIAGNOSE)
        assert trunc is True
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_TRUNCATE
        )

    # --- Wrong solution reward does NOT include resolve ---

    def test_wrong_solution_does_not_include_resolve(self):
        """Premature PROPOSE_SOLUTION does not include REWARD_RESOLVE."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(config.PROPOSE_SOLUTION)
        assert config.REWARD_RESOLVE not in [reward]
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_WRONG_SOLUTION
        )

    # --- Apologize at exactly 0.5 (boundary of APOLOGY_THRESHOLD) ---

    def test_apologize_at_exactly_apology_threshold_no_high_reward(self):
        """Apologize at frustration == 0.5 gets low reward (threshold is >0.5)."""
        env, _, _ = _env_at_scenario(0)
        env._state.frustration_level = config.FRUSTRATION_APOLOGY_THRESHOLD  # 0.5
        _, reward, _, _, _ = env.step(config.APOLOGIZE)
        # threshold check is > 0.5, so 0.5 is NOT above threshold → low reward
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_APOLOGIZE_LOW_FRUSTRATION
        )

    def test_apologize_just_above_apology_threshold_high_reward(self):
        """Apologize at frustration == 0.51 gets high reward."""
        env, _, _ = _env_at_scenario(0)
        env._state.frustration_level = 0.51
        _, reward, _, _, _ = env.step(config.APOLOGIZE)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_APOLOGIZE_HIGH_FRUSTRATION
        )


# ===========================================================================
# 4. Evaluator Edge Cases
# ===========================================================================


class TestEvaluatorEdgeCases:
    """Evaluator edge cases and boundary conditions."""

    def test_single_episode_record_has_all_keys(self):
        """Single episode record contains all 8 required keys."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=1, seed=42)
        required = {
            "episode_id", "scenario_id", "total_reward",
            "number_of_turns", "success", "terminated",
            "truncated", "termination_reason",
        }
        assert set(records[0].keys()) == required

    def test_episode_id_starts_at_zero(self):
        """First episode always has episode_id == 0."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=3, seed=42)
        assert records[0]["episode_id"] == 0

    def test_success_false_when_escalated(self):
        """success is False for escalated episodes."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        for rec in records:
            if rec["termination_reason"] == "escalated":
                assert rec["success"] is False

    def test_success_false_when_truncated(self):
        """success is False for truncated episodes."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        for rec in records:
            if rec["truncated"]:
                assert rec["success"] is False

    def test_terminated_and_truncated_mutually_exclusive(self):
        """No episode has both terminated and truncated == True."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=50, seed=42)
        for rec in records:
            assert not (rec["terminated"] and rec["truncated"])

    def test_num_episodes_float_raises(self):
        """Float num_episodes raises ValueError."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        with pytest.raises(ValueError, match="num_episodes"):
            evaluate_agent(env, agent, num_episodes=5.0)

    def test_seed_float_raises(self):
        """Float seed raises ValueError."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        with pytest.raises(ValueError, match="seed"):
            evaluate_agent(env, agent, num_episodes=5, seed=42.0)

    def test_seed_none_raises(self):
        """None seed raises ValueError (seed must be an int)."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        with pytest.raises((ValueError, TypeError)):
            evaluate_agent(env, agent, num_episodes=5, seed=None)

    def test_max_steps_1_records_one_turn(self):
        """max_steps=1 means each episode runs exactly 1 step."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=5, seed=42, max_steps=1)
        for rec in records:
            assert rec["number_of_turns"] == 1

    def test_additive_seeding_is_reproducible(self):
        """Per-episode seed (base + episode_id) is deterministic."""
        env1 = CustomerSupportEnv()
        agent1 = BaselineSupportAgent()
        records1 = evaluate_agent(env1, agent1, num_episodes=5, seed=100)

        env2 = CustomerSupportEnv()
        agent2 = BaselineSupportAgent()
        records2 = evaluate_agent(env2, agent2, num_episodes=5, seed=100)

        for r1, r2 in zip(records1, records2):
            assert r1 == r2

    def test_10_episode_all_fields_valid(self):
        """10-episode run: all required fields present and valid types."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=10, seed=42)
        for rec in records:
            assert isinstance(rec["episode_id"], int)
            assert isinstance(rec["scenario_id"], int)
            assert isinstance(rec["total_reward"], float)
            assert isinstance(rec["number_of_turns"], int)
            assert isinstance(rec["success"], bool)
            assert isinstance(rec["terminated"], bool)
            assert isinstance(rec["truncated"], bool)
            assert isinstance(rec["termination_reason"], str)
            assert rec["number_of_turns"] >= 1


# ===========================================================================
# 5. Metrics Edge Cases
# ===========================================================================


def _rec(
    episode_id: int = 0,
    scenario_id: int = 0,
    total_reward: float = 10.5,
    number_of_turns: int = 5,
    success: bool = True,
    terminated: bool = True,
    truncated: bool = False,
    termination_reason: str = "resolved",
) -> dict:
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "total_reward": total_reward,
        "number_of_turns": number_of_turns,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": termination_reason,
    }


class TestMetricsEdgeCases:
    """Metrics edge cases and boundary conditions."""

    def test_single_record_metrics(self):
        """Single record produces valid metrics."""
        records = [_rec(total_reward=10.5, number_of_turns=5)]
        m = compute_metrics(records)
        assert m["total_episodes"] == 1
        assert m["successful_resolutions"] == 1
        assert m["failed_resolutions"] == 0
        assert m["success_rate"] == pytest.approx(1.0)
        assert m["mean_reward"] == pytest.approx(10.5)
        assert m["std_reward"] == pytest.approx(0.0)
        assert m["mean_turns"] == pytest.approx(5.0)

    def test_std_reward_all_identical(self):
        """All identical rewards → std = 0."""
        records = [_rec(total_reward=10.5) for _ in range(10)]
        m = compute_metrics(records)
        assert m["std_reward"] == pytest.approx(0.0)

    def test_negative_rewards_handled(self):
        """Negative rewards are handled correctly."""
        records = [
            _rec(0, total_reward=-5.0, success=False, terminated=False,
                 truncated=True, termination_reason="truncated"),
            _rec(1, total_reward=-10.0, success=False, terminated=False,
                 truncated=True, termination_reason="truncated"),
        ]
        m = compute_metrics(records)
        assert m["min_reward"] == pytest.approx(-10.0)
        assert m["max_reward"] == pytest.approx(-5.0)
        assert m["mean_reward"] == pytest.approx(-7.5)

    def test_all_truncated_rates(self):
        """All truncated: truncation_rate would be 1.0 of terminated+truncated."""
        records = [
            _rec(i, success=False, terminated=False, truncated=True,
                 termination_reason="truncated")
            for i in range(5)
        ]
        m = compute_metrics(records)
        assert m["truncated_episodes"] == 5
        assert m["terminated_episodes"] == 0
        assert m["success_rate"] == pytest.approx(0.0)

    def test_mean_turns_exact(self):
        """Mean turns computed exactly for mixed values."""
        records = [
            _rec(0, number_of_turns=3),
            _rec(1, number_of_turns=7),
            _rec(2, number_of_turns=5),
        ]
        m = compute_metrics(records)
        assert m["mean_turns"] == pytest.approx(5.0)

    def test_scenario_breakdown_uneven_counts(self):
        """Breakdown handles uneven distribution across scenarios."""
        records = [_rec(i, scenario_id=0) for i in range(8)]
        records += [_rec(8, scenario_id=4, success=False, terminated=False,
                         truncated=True, termination_reason="truncated")]
        breakdown = compute_scenario_breakdown(records)
        assert breakdown[0]["count"] == 8
        assert breakdown[4]["count"] == 1
        for sid in [1, 2, 3]:
            assert breakdown[sid]["count"] == 0
            assert breakdown[sid]["success_rate"] == pytest.approx(0.0)

    def test_compute_metrics_returns_native_types(self):
        """All metric values are plain Python float/int, not numpy scalars."""
        records = [_rec(i, scenario_id=i % NUM_SCENARIOS) for i in range(10)]
        m = compute_metrics(records)
        assert isinstance(m["mean_reward"], float)
        assert isinstance(m["std_reward"], float)
        assert isinstance(m["total_episodes"], int)
        assert isinstance(m["successful_resolutions"], int)

    def test_total_reward_is_float_in_record(self):
        """total_reward in episode record is always a float."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=5, seed=42)
        for rec in records:
            assert isinstance(rec["total_reward"], float)

    def test_scenario_breakdown_mean_reward_exact(self):
        """Scenario breakdown mean reward is correct with 3 records."""
        records = [
            _rec(0, scenario_id=2, total_reward=10.5),
            _rec(1, scenario_id=2, total_reward=10.5),
            _rec(2, scenario_id=2, total_reward=10.5),
        ]
        breakdown = compute_scenario_breakdown(records)
        assert breakdown[2]["mean_reward"] == pytest.approx(10.5)
        assert breakdown[2]["mean_turns"] == pytest.approx(5.0)


# ===========================================================================
# 6. Integration: 100-Episode Full Run
# ===========================================================================


class TestIntegration100Episodes:
    """Full 100-episode integration tests — real agent-environment runs."""

    @pytest.fixture(scope="class")
    @classmethod
    def records_100(cls):
        """Run 100 episodes once and share across all tests in this class."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        return evaluate_agent(env, agent, num_episodes=100, seed=42)

    @pytest.fixture(scope="class")
    @classmethod
    def metrics_100(cls, records_100):
        return compute_metrics(records_100)

    def test_exactly_100_records(self, records_100):
        """Exactly 100 episode records are produced."""
        assert len(records_100) == 100

    def test_all_episodes_terminate_or_truncate(self, records_100):
        """Every episode terminated or truncated."""
        for rec in records_100:
            assert rec["terminated"] or rec["truncated"], (
                f"Episode {rec['episode_id']} neither terminated nor truncated"
            )

    def test_turn_limits_respected(self, records_100):
        """All episode turns within [1, DEFAULT_MAX_TURNS]."""
        for rec in records_100:
            assert 1 <= rec["number_of_turns"] <= config.DEFAULT_MAX_TURNS

    def test_all_rewards_finite(self, records_100):
        """All rewards are finite (not NaN or infinite)."""
        for rec in records_100:
            assert math.isfinite(rec["total_reward"])

    def test_all_scenario_ids_valid(self, records_100):
        """All scenario IDs are in [0, NUM_SCENARIOS)."""
        for rec in records_100:
            assert 0 <= rec["scenario_id"] < NUM_SCENARIOS

    def test_all_five_scenarios_appear(self, records_100):
        """All five scenarios appear across 100 episodes."""
        seen = {rec["scenario_id"] for rec in records_100}
        assert seen == set(range(NUM_SCENARIOS))

    def test_consistent_success_failure_counts(self, records_100):
        """successful + failed == 100 in episode records."""
        successes = sum(1 for r in records_100 if r["success"])
        failures = sum(1 for r in records_100 if not r["success"])
        assert successes + failures == 100

    def test_metrics_total_equals_100(self, metrics_100):
        """Metrics total_episodes == 100."""
        assert metrics_100["total_episodes"] == 100

    def test_metrics_sum_constraint(self, metrics_100):
        """successful_resolutions + failed_resolutions == 100."""
        assert (metrics_100["successful_resolutions"]
                + metrics_100["failed_resolutions"]) == 100

    def test_metrics_rates_sum_to_one(self, metrics_100):
        """success_rate + failure_rate == 1.0."""
        assert (metrics_100["success_rate"]
                + metrics_100["failure_rate"]) == pytest.approx(1.0)

    def test_metrics_mean_reward_correct(self, metrics_100):
        """Mean reward is +10.5 (all episodes resolve optimally)."""
        assert metrics_100["mean_reward"] == pytest.approx(10.5)

    def test_metrics_std_reward_zero(self, metrics_100):
        """Std reward is 0.0 (all episodes identical with baseline agent)."""
        assert metrics_100["std_reward"] == pytest.approx(0.0)

    def test_metrics_mean_turns_correct(self, metrics_100):
        """Mean turns is 5.0 (all scenarios need 5 turns)."""
        assert metrics_100["mean_turns"] == pytest.approx(5.0)

    def test_hidden_state_never_in_records(self, records_100):
        """No hidden state keys appear in any episode record."""
        hidden = {
            "hidden_issue", "diagnosed", "solution_proposed",
            "solution_rejected", "last_action", "consecutive_action_count",
            "scenario", "state", "conversation_state",
        }
        for rec in records_100:
            for key in hidden:
                assert key not in rec

    def test_scenario_breakdown_counts_sum_to_100(self, metrics_100):
        """Per-scenario counts sum to 100."""
        total = sum(
            v["count"] for v in metrics_100["scenario_breakdown"].values()
        )
        assert total == 100

    def test_scenario_breakdown_all_success(self, metrics_100):
        """All scenarios achieve 100% success rate."""
        for sid in range(NUM_SCENARIOS):
            sb = metrics_100["scenario_breakdown"][sid]
            if sb["count"] > 0:
                assert sb["success_rate"] == pytest.approx(1.0), (
                    f"Scenario {sid} ({SCENARIOS[sid].name}): "
                    f"success_rate={sb['success_rate']:.2f}"
                )

    def test_scenario_breakdown_all_reward_optimal(self, metrics_100):
        """All scenarios achieve +10.5 mean reward."""
        for sid in range(NUM_SCENARIOS):
            sb = metrics_100["scenario_breakdown"][sid]
            if sb["count"] > 0:
                assert sb["mean_reward"] == pytest.approx(10.5), (
                    f"Scenario {sid} ({SCENARIOS[sid].name}): "
                    f"mean_reward={sb['mean_reward']:.4f}"
                )

    def test_episode_ids_are_sequential(self, records_100):
        """Episode IDs are 0, 1, 2, ..., 99 in order."""
        ids = [r["episode_id"] for r in records_100]
        assert ids == list(range(100))

    def test_reproducibility_full_run(self):
        """Two independent 100-episode runs with same seed produce identical results."""
        env1 = CustomerSupportEnv()
        a1 = BaselineSupportAgent()
        r1 = evaluate_agent(env1, a1, num_episodes=100, seed=42)

        env2 = CustomerSupportEnv()
        a2 = BaselineSupportAgent()
        r2 = evaluate_agent(env2, a2, num_episodes=100, seed=42)

        for rec1, rec2 in zip(r1, r2):
            assert rec1["scenario_id"] == rec2["scenario_id"]
            assert rec1["total_reward"] == pytest.approx(rec2["total_reward"])
            assert rec1["number_of_turns"] == rec2["number_of_turns"]
            assert rec1["termination_reason"] == rec2["termination_reason"]


# ===========================================================================
# 7. Code Quality and Reproducibility
# ===========================================================================


class TestCodeQuality:
    """Code quality and reproducibility checks."""

    def test_no_hardcoded_reward_in_compute_metrics(self):
        """compute_metrics uses actual data, not hardcoded values."""
        # Verify it respects different reward values
        records_high = [_rec(total_reward=99.0) for _ in range(3)]
        records_low = [_rec(total_reward=-99.0, success=False,
                            terminated=False, truncated=True,
                            termination_reason="truncated") for _ in range(3)]
        m_high = compute_metrics(records_high)
        m_low = compute_metrics(records_low)
        assert m_high["mean_reward"] == pytest.approx(99.0)
        assert m_low["mean_reward"] == pytest.approx(-99.0)

    def test_env_state_none_before_reset(self):
        """Environment _state is None before first reset."""
        env = CustomerSupportEnv()
        assert env._state is None
        assert env._episode_active is False

    def test_env_state_populated_after_reset(self):
        """Environment _state is a ConversationState after reset."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        assert env._state is not None
        assert isinstance(env._state, ConversationState)

    def test_env_state_none_after_close(self):
        """Environment _state is None after close()."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        env.close()
        assert env._state is None
        assert env._episode_active is False

    def test_render_no_error_active_episode(self, capsys):
        """render() runs without error during an active episode."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        env.render()
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_render_no_error_before_reset(self, capsys):
        """render() runs without error before any reset."""
        env = CustomerSupportEnv()
        env.render()
        out = capsys.readouterr().out
        assert "No active episode" in out

    def test_agent_state_transitions_correctly(self):
        """Agent internal state transitions: not diagnosed → diagnosed → proposed."""
        agent = BaselineSupportAgent()
        assert agent._has_diagnosed is False
        assert agent._has_proposed is False
        # Simulate diagnosing
        agent._update_tracking(config.DIAGNOSE)
        assert agent._has_diagnosed is True
        assert agent._has_proposed is False
        # Simulate proposing
        agent._update_tracking(config.PROPOSE_SOLUTION)
        assert agent._has_proposed is True

    def test_consecutive_count_increments_on_repeat(self):
        """Agent consecutive count increments on same action."""
        agent = BaselineSupportAgent()
        agent._last_action = config.ASK_ACCOUNT_INFO
        agent._consecutive_count = 1
        agent._update_tracking(config.ASK_ACCOUNT_INFO)
        assert agent._consecutive_count == 2

    def test_consecutive_count_resets_on_different_action(self):
        """Agent consecutive count resets to 1 on different action."""
        agent = BaselineSupportAgent()
        agent._last_action = config.ASK_ACCOUNT_INFO
        agent._consecutive_count = 3
        agent._update_tracking(config.DIAGNOSE)
        assert agent._consecutive_count == 1
        assert agent._last_action == config.DIAGNOSE

    def test_config_constants_are_immutable_types(self):
        """Config constants are Python primitives (int/float/str), not mutable."""
        assert isinstance(config.NUM_ACTIONS, int)
        assert isinstance(config.REWARD_RESOLVE, float)
        assert isinstance(config.TERMINATION_RESOLVED, str)
        assert isinstance(config.DEFAULT_MAX_TURNS, int)
        assert isinstance(config.FRUSTRATION_HIGH_THRESHOLD, float)

    def test_scenarios_tuple_is_frozen(self):
        """SCENARIOS tuple is immutable and all entries are frozen dataclasses."""
        from src.support_scenarios import SCENARIOS
        assert isinstance(SCENARIOS, tuple)
        for s in SCENARIOS:
            # frozen=True means we can't set attributes
            with pytest.raises((AttributeError, TypeError)):
                s.id = 999

    def test_evaluate_agent_records_length_matches_num_episodes(self):
        """Length of records always equals num_episodes exactly."""
        for n in [1, 5, 10, 25]:
            env = CustomerSupportEnv()
            agent = BaselineSupportAgent()
            records = evaluate_agent(env, agent, num_episodes=n, seed=42)
            assert len(records) == n

    def test_observation_space_contains_after_every_step(self):
        """observation_space.contains(obs) is True after every step."""
        env = CustomerSupportEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if term or trunc:
                break
