"""Tests for metrics computation.

Wave 1: Verifies reward constants used in metrics.
Wave 5: Verifies compute_metrics() and compute_scenario_breakdown()
        with real and synthetic data, empty inputs, rate constraints,
        and scenario-level breakdowns.
"""

import pytest
import numpy as np

from src import config
from src.metrics import compute_metrics, compute_scenario_breakdown
from src.evaluator import evaluate_agent
from src.support_env import CustomerSupportEnv
from src.baseline_agent import BaselineSupportAgent
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS


# ---------------------------------------------------------------------------
# Wave 1: Reward constant verification (FR-05 AC-05.01 through AC-05.08)
# ---------------------------------------------------------------------------


class TestRewardConstants:
    """Verify all reward constant values match the specification (FR-05)."""

    def test_reward_resolve(self):
        """AC-05.01: Correct resolution yields +10.0."""
        assert config.REWARD_RESOLVE == 10.0

    def test_reward_info_gathered(self):
        """AC-05.02: Gathering info yields +1.0."""
        assert config.REWARD_INFO_GATHERED == 1.0

    def test_reward_turn_penalty(self):
        """AC-05.03: Each turn incurs -0.5."""
        assert config.REWARD_TURN_PENALTY == -0.5

    def test_reward_wrong_solution(self):
        """AC-05.04: Wrong solution yields -3.0."""
        assert config.REWARD_WRONG_SOLUTION == -3.0

    def test_reward_repeated_action(self):
        """AC-05.05: Repeated action yields -2.0."""
        assert config.REWARD_REPEATED_ACTION == -2.0

    def test_reward_escalate(self):
        """AC-05.06: Escalation yields +2.0."""
        assert config.REWARD_ESCALATE == 2.0

    def test_reward_truncate(self):
        """AC-05.07: Turn limit exceeded yields -5.0."""
        assert config.REWARD_TRUNCATE == -5.0

    def test_reward_apologize_high_frustration(self):
        """AC-05.08: Apologize at high frustration yields +0.5."""
        assert config.REWARD_APOLOGIZE_HIGH_FRUSTRATION == 0.5

    def test_reward_apologize_low_frustration(self):
        """AC-05.08: Apologize at low frustration yields 0.0."""
        assert config.REWARD_APOLOGIZE_LOW_FRUSTRATION == 0.0


class TestFrustrationConstants:
    """Verify frustration thresholds and deltas (Design Section 9.5)."""

    def test_frustration_thresholds_in_range(self):
        assert 0.0 < config.FRUSTRATION_APOLOGY_THRESHOLD < 1.0
        assert 0.0 < config.FRUSTRATION_HIGH_THRESHOLD < 1.0

    def test_apology_threshold_below_high_threshold(self):
        assert config.FRUSTRATION_APOLOGY_THRESHOLD <= config.FRUSTRATION_HIGH_THRESHOLD

    def test_all_reward_values_are_finite(self):
        """AC-05.09: All reward values must be finite floats."""
        rewards = [
            config.REWARD_RESOLVE,
            config.REWARD_INFO_GATHERED,
            config.REWARD_TURN_PENALTY,
            config.REWARD_WRONG_SOLUTION,
            config.REWARD_REPEATED_ACTION,
            config.REWARD_ESCALATE,
            config.REWARD_TRUNCATE,
            config.REWARD_APOLOGIZE_HIGH_FRUSTRATION,
            config.REWARD_APOLOGIZE_LOW_FRUSTRATION,
        ]
        for r in rewards:
            assert isinstance(r, float)
            assert r == r  # NaN check (NaN != NaN)


# ---------------------------------------------------------------------------
# Wave 5: compute_metrics tests (FR-10)
# ---------------------------------------------------------------------------

# Synthetic records helpers

def _make_record(
    episode_id: int = 0,
    scenario_id: int = 0,
    total_reward: float = 5.0,
    number_of_turns: int = 6,
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


class TestComputeMetricsStructure:
    """Verify compute_metrics returns all required keys (FR-10 AC-10.01)."""

    REQUIRED_KEYS = {
        "total_episodes", "successful_resolutions", "failed_resolutions",
        "success_rate", "failure_rate", "mean_reward", "min_reward",
        "max_reward", "std_reward", "mean_turns",
        "terminated_episodes", "truncated_episodes", "scenario_breakdown",
    }

    def test_returns_all_required_keys(self):
        records = [_make_record()]
        metrics = compute_metrics(records)
        for key in self.REQUIRED_KEYS:
            assert key in metrics, f"Missing key: '{key}'"

    def test_empty_input_has_all_keys(self):
        """Empty input still returns all required keys."""
        metrics = compute_metrics([])
        for key in self.REQUIRED_KEYS:
            assert key in metrics, f"Missing key: '{key}'"

    def test_values_are_finite(self):
        """All numeric metric values are finite (AC-05.09)."""
        records = [_make_record(i, i % 5) for i in range(10)]
        metrics = compute_metrics(records)
        numeric_keys = [
            "success_rate", "failure_rate", "mean_reward",
            "min_reward", "max_reward", "std_reward", "mean_turns",
        ]
        for key in numeric_keys:
            assert metrics[key] == metrics[key], f"NaN in '{key}'"


class TestComputeMetricsCounts:
    """Verify count metrics are correct (FR-10 AC-10.01, AC-10.03)."""

    def test_total_episodes(self):
        records = [_make_record(i) for i in range(7)]
        assert compute_metrics(records)["total_episodes"] == 7

    def test_empty_total(self):
        assert compute_metrics([])["total_episodes"] == 0

    def test_successful_count_all_success(self):
        records = [_make_record(i, success=True) for i in range(5)]
        m = compute_metrics(records)
        assert m["successful_resolutions"] == 5
        assert m["failed_resolutions"] == 0

    def test_successful_count_all_fail(self):
        records = [_make_record(i, success=False, terminated=False,
                                truncated=True, termination_reason="truncated")
                   for i in range(4)]
        m = compute_metrics(records)
        assert m["successful_resolutions"] == 0
        assert m["failed_resolutions"] == 4

    def test_successful_plus_failed_equals_total(self):
        """AC-10.03: successful + failed == total (always)."""
        records = [
            _make_record(0, success=True),
            _make_record(1, success=False, terminated=False,
                         truncated=True, termination_reason="truncated"),
            _make_record(2, success=True),
        ]
        m = compute_metrics(records)
        assert m["successful_resolutions"] + m["failed_resolutions"] == m["total_episodes"]

    def test_terminated_count(self):
        records = [
            _make_record(0, terminated=True, truncated=False),
            _make_record(1, terminated=False, truncated=True,
                         success=False, termination_reason="truncated"),
        ]
        m = compute_metrics(records)
        assert m["terminated_episodes"] == 1
        assert m["truncated_episodes"] == 1


class TestComputeMetricsRates:
    """Verify rate metrics are correct (FR-10)."""

    def test_success_rate_all_success(self):
        records = [_make_record(i, success=True) for i in range(4)]
        m = compute_metrics(records)
        assert m["success_rate"] == pytest.approx(1.0)
        assert m["failure_rate"] == pytest.approx(0.0)

    def test_success_rate_all_fail(self):
        records = [_make_record(i, success=False, terminated=False,
                                truncated=True, termination_reason="truncated")
                   for i in range(4)]
        m = compute_metrics(records)
        assert m["success_rate"] == pytest.approx(0.0)
        assert m["failure_rate"] == pytest.approx(1.0)

    def test_success_rate_partial(self):
        """3 out of 4 successes = 0.75 success rate."""
        records = [
            _make_record(0, success=True),
            _make_record(1, success=True),
            _make_record(2, success=True),
            _make_record(3, success=False, terminated=False,
                         truncated=True, termination_reason="truncated"),
        ]
        m = compute_metrics(records)
        assert m["success_rate"] == pytest.approx(0.75)
        assert m["failure_rate"] == pytest.approx(0.25)

    def test_success_plus_failure_rate_equals_one(self):
        """success_rate + failure_rate == 1.0."""
        records = [_make_record(i, success=(i % 2 == 0)) for i in range(6)]
        m = compute_metrics(records)
        assert m["success_rate"] + m["failure_rate"] == pytest.approx(1.0)

    def test_empty_rates_are_zero(self):
        m = compute_metrics([])
        assert m["success_rate"] == pytest.approx(0.0)
        assert m["failure_rate"] == pytest.approx(0.0)


class TestComputeMetricsRewards:
    """Verify reward metrics are correctly computed (FR-10)."""

    def test_mean_reward(self):
        records = [
            _make_record(0, total_reward=10.0),
            _make_record(1, total_reward=4.0, success=False,
                         terminated=False, truncated=True,
                         termination_reason="truncated"),
        ]
        m = compute_metrics(records)
        assert m["mean_reward"] == pytest.approx(7.0)

    def test_min_reward(self):
        records = [_make_record(i, total_reward=float(i + 1)) for i in range(5)]
        assert compute_metrics(records)["min_reward"] == pytest.approx(1.0)

    def test_max_reward(self):
        records = [_make_record(i, total_reward=float(i + 1)) for i in range(5)]
        assert compute_metrics(records)["max_reward"] == pytest.approx(5.0)

    def test_std_reward_single(self):
        """Single episode: std = 0."""
        records = [_make_record(0, total_reward=7.5)]
        assert compute_metrics(records)["std_reward"] == pytest.approx(0.0)

    def test_std_reward_two_episodes(self):
        records = [
            _make_record(0, total_reward=4.0),
            _make_record(1, total_reward=6.0, success=False,
                         terminated=False, truncated=True,
                         termination_reason="truncated"),
        ]
        m = compute_metrics(records)
        assert m["std_reward"] == pytest.approx(1.0)

    def test_empty_rewards_are_zero(self):
        m = compute_metrics([])
        assert m["mean_reward"] == pytest.approx(0.0)
        assert m["min_reward"] == pytest.approx(0.0)
        assert m["max_reward"] == pytest.approx(0.0)
        assert m["std_reward"] == pytest.approx(0.0)


class TestComputeMetricsTurns:
    """Verify turn metrics are correctly computed."""

    def test_mean_turns(self):
        records = [
            _make_record(0, number_of_turns=5),
            _make_record(1, number_of_turns=9, success=False,
                         terminated=False, truncated=True,
                         termination_reason="truncated"),
        ]
        assert compute_metrics(records)["mean_turns"] == pytest.approx(7.0)

    def test_empty_turns_zero(self):
        assert compute_metrics([])["mean_turns"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Wave 5: compute_scenario_breakdown tests (FR-10)
# ---------------------------------------------------------------------------


class TestScenarioBreakdown:
    """Verify scenario-level breakdown covers all five scenarios."""

    def test_breakdown_contains_all_scenarios(self):
        """Breakdown has an entry for every scenario ID 0-4."""
        records = [_make_record(i, scenario_id=i % NUM_SCENARIOS) for i in range(10)]
        breakdown = compute_scenario_breakdown(records)
        assert set(breakdown.keys()) == set(range(NUM_SCENARIOS))

    def test_breakdown_empty_input(self):
        """Empty input still has entries for all scenarios."""
        breakdown = compute_scenario_breakdown([])
        assert set(breakdown.keys()) == set(range(NUM_SCENARIOS))
        for sid in range(NUM_SCENARIOS):
            assert breakdown[sid]["count"] == 0
            assert breakdown[sid]["success_rate"] == pytest.approx(0.0)

    def test_breakdown_scenario_name(self):
        """Breakdown includes scenario name from SCENARIOS."""
        breakdown = compute_scenario_breakdown([])
        for sid in range(NUM_SCENARIOS):
            assert breakdown[sid]["name"] == SCENARIOS[sid].name

    def test_breakdown_success_count(self):
        """Successful episodes are counted per scenario."""
        records = [
            _make_record(0, scenario_id=0, success=True),
            _make_record(1, scenario_id=0, success=True),
            _make_record(2, scenario_id=0, success=False,
                         terminated=False, truncated=True,
                         termination_reason="truncated"),
        ]
        breakdown = compute_scenario_breakdown(records)
        assert breakdown[0]["successes"] == 2
        assert breakdown[0]["failures"] == 1
        assert breakdown[0]["count"] == 3
        assert breakdown[0]["success_rate"] == pytest.approx(2 / 3)

    def test_breakdown_mean_reward_per_scenario(self):
        """Mean reward is computed per scenario."""
        records = [
            _make_record(0, scenario_id=1, total_reward=8.0),
            _make_record(1, scenario_id=1, total_reward=4.0, success=False,
                         terminated=False, truncated=True,
                         termination_reason="truncated"),
        ]
        breakdown = compute_scenario_breakdown(records)
        assert breakdown[1]["mean_reward"] == pytest.approx(6.0)

    def test_breakdown_in_compute_metrics(self):
        """compute_metrics includes scenario_breakdown."""
        records = [_make_record(i, scenario_id=i % NUM_SCENARIOS) for i in range(10)]
        m = compute_metrics(records)
        assert "scenario_breakdown" in m
        assert set(m["scenario_breakdown"].keys()) == set(range(NUM_SCENARIOS))

    def test_breakdown_required_fields(self):
        """Each scenario entry has all required fields."""
        breakdown = compute_scenario_breakdown([])
        required = {"name", "count", "successes", "failures",
                    "success_rate", "mean_reward", "mean_turns"}
        for sid in range(NUM_SCENARIOS):
            for key in required:
                assert key in breakdown[sid], (
                    f"Scenario {sid} missing key '{key}'"
                )

    def test_breakdown_successes_plus_failures_equals_count(self):
        """For each scenario: successes + failures == count."""
        records = [
            _make_record(0, scenario_id=0, success=True),
            _make_record(1, scenario_id=0, success=False,
                         terminated=False, truncated=True,
                         termination_reason="truncated"),
            _make_record(2, scenario_id=2, success=True),
        ]
        breakdown = compute_scenario_breakdown(records)
        for sid in range(NUM_SCENARIOS):
            b = breakdown[sid]
            assert b["successes"] + b["failures"] == b["count"]


# ---------------------------------------------------------------------------
# Wave 5: Integration metrics tests (real evaluation data)
# ---------------------------------------------------------------------------


class TestMetricsWithRealData:
    """Verify metrics on real 20-episode evaluation data."""

    def _get_records(self, n: int = 20) -> list[dict]:
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        return evaluate_agent(env, agent, num_episodes=n, seed=42)

    def test_real_data_structure(self):
        records = self._get_records(5)
        m = compute_metrics(records)
        assert m["total_episodes"] == 5
        assert m["successful_resolutions"] + m["failed_resolutions"] == 5

    def test_real_data_success_rate_in_range(self):
        records = self._get_records(20)
        m = compute_metrics(records)
        assert 0.0 <= m["success_rate"] <= 1.0
        assert 0.0 <= m["failure_rate"] <= 1.0

    def test_real_data_mean_reward_finite(self):
        records = self._get_records(20)
        m = compute_metrics(records)
        assert m["mean_reward"] == m["mean_reward"]  # NaN check

    def test_real_data_turns_positive(self):
        records = self._get_records(20)
        m = compute_metrics(records)
        assert m["mean_turns"] > 0

    def test_real_data_breakdown_counts_sum_to_total(self):
        """Sum of per-scenario counts equals total episodes."""
        records = self._get_records(30)
        m = compute_metrics(records)
        total_from_breakdown = sum(
            v["count"] for v in m["scenario_breakdown"].values()
        )
        assert total_from_breakdown == m["total_episodes"]

    def test_real_data_scenario_breakdown_all_five(self):
        """50 episodes cover all 5 scenarios."""
        records = self._get_records(50)
        m = compute_metrics(records)
        covered = {
            sid for sid, v in m["scenario_breakdown"].items()
            if v["count"] > 0
        }
        assert covered == set(range(NUM_SCENARIOS))
