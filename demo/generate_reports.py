"""Generate JSON and Markdown evaluation reports.

Runs the full evaluation pipeline using evaluate_agent(), compute_metrics(),
and compute_scenario_breakdown(), then writes:
    reports/evaluation_report.json
    reports/evaluation_summary.md

All numbers in the reports come directly from the evaluation output.
No results are hardcoded.

Usage (from repository root):
    python -m demo.generate_reports [--episodes N] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from src.baseline_agent import BaselineSupportAgent
from src.config import DEFAULT_NUM_EPISODES, DEFAULT_SEED
from src.evaluator import evaluate_agent
from src.metrics import compute_metrics, compute_scenario_breakdown
from src.support_env import CustomerSupportEnv
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS

_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def build_report(
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the evaluation and build the full report dict.

    Args:
        num_episodes: Number of episodes to evaluate.
        seed: Base random seed.

    Returns:
        A JSON-serializable dict with timestamp, config, metrics,
        scenario breakdown, and per-episode results.
    """
    env = CustomerSupportEnv()
    agent = BaselineSupportAgent()

    print(f"  Running {num_episodes} episodes (seed={seed}) …")
    records = evaluate_agent(env, agent, num_episodes=num_episodes, seed=seed)
    metrics = compute_metrics(records)

    # Convert scenario_breakdown keys from int to str (JSON requires str keys)
    raw_breakdown = metrics.pop("scenario_breakdown")
    scenario_breakdown: dict[str, dict] = {
        str(k): v for k, v in raw_breakdown.items()
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "CustomerSupportEnv",
        "agent": "BaselineSupportAgent",
        "config": {
            "num_episodes": num_episodes,
            "seed": seed,
            "max_turns": env.max_turns,
            "num_scenarios": NUM_SCENARIOS,
        },
        "metrics": metrics,
        "scenario_breakdown": scenario_breakdown,
        "episodes": records,
    }


def write_json_report(report: dict, path: str) -> None:
    """Write the report dict to a JSON file.

    Args:
        report: The report dict to serialise.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def validate_json_report(path: str) -> dict:
    """Read the JSON file back and verify its top-level structure.

    Args:
        path: Path to the JSON report file.

    Returns:
        The parsed dict.

    Raises:
        AssertionError: If required top-level keys are missing.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    required_keys = {
        "generated_at", "environment", "agent",
        "config", "metrics", "scenario_breakdown", "episodes",
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"JSON report missing keys: {missing}"

    required_metrics = {
        "total_episodes", "successful_resolutions", "failed_resolutions",
        "success_rate", "failure_rate", "mean_reward", "min_reward",
        "max_reward", "std_reward", "mean_turns",
        "terminated_episodes", "truncated_episodes",
    }
    missing_metrics = required_metrics - set(data["metrics"].keys())
    assert not missing_metrics, f"metrics dict missing keys: {missing_metrics}"

    assert len(data["episodes"]) == data["config"]["num_episodes"], (
        "Episode count mismatch between 'episodes' list and config"
    )

    return data


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(report: dict, path: str) -> None:
    """Write a human-readable Markdown summary of the evaluation.

    All numbers come directly from the report dict — nothing is hardcoded.

    Args:
        report: The full report dict (must have metrics and scenario_breakdown).
        path: Destination .md file path.
    """
    m = report["metrics"]
    cfg = report["config"]
    generated_at = report["generated_at"]
    sb = report["scenario_breakdown"]
    env_name = report["environment"]
    agent_name = report["agent"]

    lines: list[str] = []
    a = lines.append

    a("# Evaluation Summary — multi-turn-agent-gym")
    a("")
    a(f"Generated: {generated_at}")
    a("")

    a("## Configuration")
    a("")
    a(f"| Parameter | Value |")
    a(f"|---|---|")
    a(f"| Environment | `{env_name}` |")
    a(f"| Agent | `{agent_name}` |")
    a(f"| Episodes | {cfg['num_episodes']} |")
    a(f"| Seed | {cfg['seed']} |")
    a(f"| Max turns per episode | {cfg['max_turns']} |")
    a(f"| Scenarios | {cfg['num_scenarios']} |")
    a("")

    a("## Aggregate Metrics")
    a("")
    a(f"| Metric | Value |")
    a(f"|---|---|")
    a(f"| Total episodes | {m['total_episodes']} |")
    a(f"| Successful resolutions | {m['successful_resolutions']} |")
    a(f"| Failed resolutions | {m['failed_resolutions']} |")
    a(f"| **Success rate** | **{m['success_rate']:.1%}** |")
    a(f"| Failure rate | {m['failure_rate']:.1%} |")
    a(f"| Mean reward | {m['mean_reward']:+.4f} |")
    a(f"| Min reward | {m['min_reward']:+.4f} |")
    a(f"| Max reward | {m['max_reward']:+.4f} |")
    a(f"| Std reward | {m['std_reward']:.4f} |")
    a(f"| Mean turns | {m['mean_turns']:.2f} |")
    a(f"| Terminated episodes | {m['terminated_episodes']} |")
    a(f"| Truncated episodes | {m['truncated_episodes']} |")
    a("")

    a("## Per-Scenario Results")
    a("")
    a("| # | Scenario | Episodes | Resolved | Success Rate | Mean Reward | Mean Turns |")
    a("|---|---|---|---|---|---|---|")
    for sid_str, s in sb.items():
        if s["count"] > 0:
            a(
                f"| {sid_str} | {s['name']} | {s['count']} "
                f"| {s['successes']} | {s['success_rate']:.1%} "
                f"| {s['mean_reward']:+.2f} | {s['mean_turns']:.1f} |"
            )
        else:
            a(f"| {sid_str} | {s['name']} | 0 | — | — | — | — |")
    a("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate both reports from the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate JSON and Markdown evaluation reports."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help=f"Number of episodes (default: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    json_path = os.path.join(_REPORTS_DIR, "evaluation_report.json")
    md_path = os.path.join(_REPORTS_DIR, "evaluation_summary.md")

    print("multi-turn-agent-gym — Report Generator")
    print(f"  Episodes : {args.episodes}")
    print(f"  Seed     : {args.seed}")
    print(f"  Output   : {_REPORTS_DIR}")
    print()

    report = build_report(num_episodes=args.episodes, seed=args.seed)

    print("  Writing evaluation_report.json …")
    write_json_report(report, json_path)

    print("  Validating JSON …")
    validate_json_report(json_path)
    print("  JSON validation passed.")

    print("  Writing evaluation_summary.md …")
    write_markdown_report(report, md_path)

    m = report["metrics"]
    print()
    print("  ── Results ─────────────────────────────────────")
    print(f"  Success rate : {m['success_rate']:.1%}")
    print(f"  Mean reward  : {m['mean_reward']:+.4f}")
    print(f"  Mean turns   : {m['mean_turns']:.2f}")
    print(f"  ────────────────────────────────────────────────")
    print(f"  Reports saved to {_REPORTS_DIR}/")


if __name__ == "__main__":
    main()
