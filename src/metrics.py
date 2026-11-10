"""Performance metrics computation from per-episode evaluation records.

This module computes aggregate metrics and per-scenario breakdowns from
episode records produced by the evaluation harness. All metrics are
computed from real data — no hardcoded or fabricated statistics.

Metrics include:
    - Total, successful, and failed episode counts
    - Success and failure rates
    - Average, minimum, maximum, and std of rewards
    - Average turns per episode
    - Terminated and truncated episode counts
    - Per-scenario breakdown for all five scenarios

Constraints:
    - successful + failed == total (always)
    - Empty results are handled safely (all metrics return 0)
"""

from __future__ import annotations

import numpy as np

from src.config import TERMINATION_RESOLVED
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS


def compute_metrics(episode_records: list[dict]) -> dict:
    """Compute aggregate metrics from per-episode records.

    Args:
        episode_records: List of episode record dicts, each containing
            episode_id, scenario_id, total_reward, number_of_turns,
            success, terminated, truncated, termination_reason.

    Returns:
        A dict with all aggregate metrics and scenario breakdown.
        For empty input, all numeric metrics are 0.0 and counts are 0.
    """
    total = len(episode_records)

    if total == 0:
        return _empty_metrics()

    rewards = np.array([r["total_reward"] for r in episode_records])
    turns = np.array([r["number_of_turns"] for r in episode_records])
    successes = sum(1 for r in episode_records if r["success"])
    failures = total - successes
    terminated_count = sum(1 for r in episode_records if r["terminated"])
    truncated_count = sum(1 for r in episode_records if r["truncated"])

    return {
        "total_episodes": total,
        "successful_resolutions": successes,
        "failed_resolutions": failures,
        "success_rate": float(successes / total),
        "failure_rate": float(failures / total),
        "mean_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_turns": float(np.mean(turns)),
        "terminated_episodes": terminated_count,
        "truncated_episodes": truncated_count,
        "scenario_breakdown": compute_scenario_breakdown(episode_records),
    }


def compute_scenario_breakdown(episode_records: list[dict]) -> dict:
    """Compute per-scenario metrics for all five scenarios.

    Every scenario ID (0 through NUM_SCENARIOS-1) is included in the
    breakdown, even if no episodes used that scenario.

    Args:
        episode_records: List of episode record dicts.

    Returns:
        A dict mapping scenario_id to a metrics dict containing:
            name, count, successes, failures, success_rate,
            mean_reward, mean_turns.
    """
    breakdown: dict[int, dict] = {}

    for scenario_id in range(NUM_SCENARIOS):
        scenario_records = [
            r for r in episode_records if r["scenario_id"] == scenario_id
        ]
        count = len(scenario_records)
        scenario = SCENARIOS[scenario_id]

        if count == 0:
            breakdown[scenario_id] = {
                "name": scenario.name,
                "count": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_turns": 0.0,
            }
        else:
            successes = sum(1 for r in scenario_records if r["success"])
            rewards = [r["total_reward"] for r in scenario_records]
            turns = [r["number_of_turns"] for r in scenario_records]
            breakdown[scenario_id] = {
                "name": scenario.name,
                "count": count,
                "successes": successes,
                "failures": count - successes,
                "success_rate": float(successes / count),
                "mean_reward": float(np.mean(rewards)),
                "mean_turns": float(np.mean(turns)),
            }

    return breakdown


def _empty_metrics() -> dict:
    """Return a metrics dict with all zeros for empty input.

    Returns:
        A metrics dict with zero counts and 0.0 rates.
    """
    return {
        "total_episodes": 0,
        "successful_resolutions": 0,
        "failed_resolutions": 0,
        "success_rate": 0.0,
        "failure_rate": 0.0,
        "mean_reward": 0.0,
        "min_reward": 0.0,
        "max_reward": 0.0,
        "std_reward": 0.0,
        "mean_turns": 0.0,
        "terminated_episodes": 0,
        "truncated_episodes": 0,
        "scenario_breakdown": compute_scenario_breakdown([]),
    }
