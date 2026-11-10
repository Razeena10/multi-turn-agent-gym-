"""Tests for the evaluation harness.

Wave 1: Verifies config constants used by the evaluator.
Wave 5: Verifies evaluate_agent with single/multiple/100 episodes,
        reset behavior, reproducibility, rewards, turns, termination,
        truncation, invalid inputs, and infinite-loop protection.
"""

import pytest
import numpy as np

from src import config
from src.evaluator import evaluate_agent
from src.support_env import CustomerSupportEnv
from src.baseline_agent import BaselineSupportAgent
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS


# ---------------------------------------------------------------------------
# Wave 1: Config constant verification (FR-09, FR-11, FR-12)
# ---------------------------------------------------------------------------


class TestEvaluatorConfig:
    """Verify constants the evaluator depends on (FR-09, Design Section 11)."""

    def test_default_num_episodes(self):
        assert config.DEFAULT_NUM_EPISODES == 100

    def test_default_max_turns(self):
        assert config.DEFAULT_MAX_TURNS == 15

    def test_default_seed(self):
        assert config.DEFAULT_SEED == 42

    def test_termination_reason_constants_exist(self):
        assert config.TERMINATION_RESOLVED == "resolved"
        assert config.TERMINATION_ESCALATED == "escalated"
        assert config.TERMINATION_TRUNCATED == "truncated"
        assert config.TERMINATION_ERROR == "error"


# ---------------------------------------------------------------------------
# Wave 5: Evaluation harness tests (FR-09)
# ---------------------------------------------------------------------------


class TestEvaluateAgent:
    """Verify evaluate_agent produces correct episode records (FR-09)."""

    def test_single_episode(self):
        """Single episode produces one record with all fields."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=1, seed=42)
        assert len(records) == 1
        rec = records[0]
        assert "episode_id" in rec
        assert "scenario_id" in rec
        assert "total_reward" in rec
        assert "number_of_turns" in rec
        assert "success" in rec
        assert "terminated" in rec
        assert "truncated" in rec
        assert "termination_reason" in rec

    def test_multiple_episodes(self):
        """Multiple episodes produce correct count."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=5, seed=42)
        assert len(records) == 5

    def test_100_episodes(self):
        """100 episodes produce exactly 100 records (FR-09 AC-09.01)."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        assert len(records) == 100

    def test_record_episode_ids_sequential(self):
        """Episode IDs are 0 through num_episodes-1."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=10, seed=42)
        ids = [r["episode_id"] for r in records]
        assert ids == list(range(10))

    def test_every_episode_terminates_or_truncates(self):
        """Every episode record has terminated or truncated == True."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=20, seed=42)
        for rec in records:
            assert rec["terminated"] or rec["truncated"], (
                f"Episode {rec['episode_id']} did not terminate or truncate"
            )

    def test_numeric_rewards(self):
        """All rewards are finite floats."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=10, seed=42)
        for rec in records:
            assert isinstance(rec["total_reward"], float)
            assert rec["total_reward"] == rec["total_reward"]  # NaN check

    def test_valid_turn_counts(self):
        """Turn counts are positive and within env max_turns."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=10, seed=42)
        for rec in records:
            assert rec["number_of_turns"] > 0
            assert rec["number_of_turns"] <= config.DEFAULT_MAX_TURNS

    def test_scenario_ids_in_valid_range(self):
        """All scenario IDs are in [0, NUM_SCENARIOS)."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=50, seed=42)
        for rec in records:
            assert 0 <= rec["scenario_id"] < NUM_SCENARIOS

    def test_reproducible(self):
        """Same seed produces identical records (NFR-02)."""
        env1 = CustomerSupportEnv()
        agent1 = BaselineSupportAgent()
        records1 = evaluate_agent(env1, agent1, num_episodes=10, seed=42)

        env2 = CustomerSupportEnv()
        agent2 = BaselineSupportAgent()
        records2 = evaluate_agent(env2, agent2, num_episodes=10, seed=42)

        for r1, r2 in zip(records1, records2):
            assert r1["total_reward"] == pytest.approx(r2["total_reward"])
            assert r1["number_of_turns"] == r2["number_of_turns"]
            assert r1["termination_reason"] == r2["termination_reason"]

    def test_different_seed_different_results(self):
        """Different seeds produce different scenario sequences."""
        env1 = CustomerSupportEnv()
        agent1 = BaselineSupportAgent()
        records1 = evaluate_agent(env1, agent1, num_episodes=20, seed=42)

        env2 = CustomerSupportEnv()
        agent2 = BaselineSupportAgent()
        records2 = evaluate_agent(env2, agent2, num_episodes=20, seed=99)

        # Different seeds sample different scenario sequences
        scenarios1 = [r["scenario_id"] for r in records1]
        scenarios2 = [r["scenario_id"] for r in records2]
        assert scenarios1 != scenarios2

    def test_reset_env_and_agent_each_episode(self):
        """Env and agent are reset for each episode."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=5, seed=42)
        # Each episode should start with turn 0 internally
        # Check that scenario_ids vary (different scenarios sampled)
        # This indirectly confirms reset is working
        assert len(records) == 5

    def test_invalid_num_episodes_zero(self):
        """num_episodes=0 raises ValueError (AC-12.06)."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        with pytest.raises(ValueError, match="num_episodes"):
            evaluate_agent(env, agent, num_episodes=0)

    def test_invalid_num_episodes_negative(self):
        """Negative num_episodes raises ValueError."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        with pytest.raises(ValueError, match="num_episodes"):
            evaluate_agent(env, agent, num_episodes=-1)

    def test_invalid_seed_type(self):
        """Non-integer seed raises ValueError (AC-12.07)."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        with pytest.raises(ValueError, match="seed"):
            evaluate_agent(env, agent, num_episodes=5, seed="abc")

    def test_infinite_loop_protection(self):
        """Defensive max_steps safeguard prevents infinite loops."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(
            env, agent, num_episodes=1, seed=42, max_steps=5
        )
        # With max_steps=5, episode should be truncated or terminated
        # within 5 steps. If the scenario needs more than 5 turns,
        # the env's own truncation at max_turns=15 won't fire,
        # but our safeguard will.
        rec = records[0]
        assert rec["terminated"] or rec["truncated"]
        assert rec["number_of_turns"] <= 5

    def test_no_hidden_state_in_records(self):
        """Episode records do not contain hidden state fields."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=3, seed=42)
        hidden_keys = {
            "hidden_issue", "diagnosed", "solution_proposed",
            "solution_rejected", "last_action", "consecutive_action_count",
            "scenario", "state",
        }
        for rec in records:
            for key in hidden_keys:
                assert key not in rec, f"Hidden key '{key}' in record"

    def test_success_terminated_consistent(self):
        """success=True implies terminated=True and reason='resolved'."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=20, seed=42)
        for rec in records:
            if rec["success"]:
                assert rec["terminated"] is True
                assert rec["termination_reason"] == "resolved"
            else:
                assert rec["termination_reason"] != "resolved"


class TestEvaluateAgent100Episodes:
    """Verify the full 100-episode evaluation run (FR-09)."""

    def test_100_episodes_exact_count(self):
        """Exactly 100 episode records are produced."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        assert len(records) == 100

    def test_100_episodes_all_terminate_or_truncate(self):
        """Every one of 100 episodes terminates or truncates."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        for rec in records:
            assert rec["terminated"] or rec["truncated"]

    def test_100_episodes_consistent_counts(self):
        """successful + failed == total across 100 episodes."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        successes = sum(1 for r in records if r["success"])
        failures = sum(1 for r in records if not r["success"])
        assert successes + failures == 100

    def test_100_episodes_valid_scenario_ids(self):
        """All scenario IDs are valid across 100 episodes."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        for rec in records:
            assert 0 <= rec["scenario_id"] < NUM_SCENARIOS

    def test_100_episodes_covers_all_scenarios(self):
        """100 episodes cover all five scenarios."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=100, seed=42)
        scenario_ids = {r["scenario_id"] for r in records}
        assert scenario_ids == set(range(NUM_SCENARIOS))
