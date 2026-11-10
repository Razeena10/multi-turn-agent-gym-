"""Interactive demo for multi-turn-agent-gym.

Runs the CustomerSupportEnv with the BaselineSupportAgent and displays
each turn clearly. Only agent-visible information is shown — the hidden
root cause, expected resolution, and internal state are never exposed.

Usage (from repository root):
    python -m demo.run_demo
"""

from __future__ import annotations

import sys

from src.baseline_agent import BaselineSupportAgent
from src.config import (
    ACTION_NAMES,
    DEFAULT_MAX_TURNS,
    DEFAULT_SEED,
    DEFAULT_NUM_EPISODES,
    MESSAGE_TYPE_NAMES,
)
from src.evaluator import evaluate_agent
from src.metrics import compute_metrics
from src.support_env import CustomerSupportEnv
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP_WIDE = "=" * 70
_SEP_THIN = "-" * 70


def _banner(text: str) -> None:
    print(f"\n{_SEP_WIDE}")
    print(f"  {text}")
    print(_SEP_WIDE)


def _section(text: str) -> None:
    print(f"\n{_SEP_THIN}")
    print(f"  {text}")
    print(_SEP_THIN)


def _run_episode(
    env: CustomerSupportEnv,
    agent: BaselineSupportAgent,
    seed: int,
    verbose: bool = True,
) -> dict:
    """Run one episode and return the episode record.

    Displays each turn's action, customer response, reward, and running
    totals. Only agent-visible observation data is displayed — hidden
    state (hidden_issue, expected_resolution, diagnosed flag, etc.) is
    never accessed or printed.

    Args:
        env: The environment instance.
        agent: The agent instance.
        seed: Seed for this episode.
        verbose: Whether to print turn-by-turn output.

    Returns:
        Episode record dict with total_reward, turns, termination_reason,
        success, and scenario_id.
    """
    obs, info = env.reset(seed=seed)
    agent.reset()

    scenario_id = info["scenario_id"]
    scenario = SCENARIOS[scenario_id]

    if verbose:
        _section(f"Scenario: {scenario.name}")
        print(f"  Customer: \"{scenario.customer_issue}\"")
        print(f"  Max turns: {env.max_turns}  |  Seed: {seed}")

    total_reward = 0.0
    turn = 0
    terminated = False
    truncated = False
    final_info: dict = {}

    while not terminated and not truncated:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += reward
        turn += 1

        if verbose:
            frustration = float(obs["frustration"])
            customer_msg = MESSAGE_TYPE_NAMES.get(
                int(obs["previous_customer_response"]), "Unknown"
            )
            print(
                f"\n  Turn {turn:2d}  |  Action: {ACTION_NAMES[action]:<22}"
                f"  Reward: {reward:+.2f}  Cumulative: {total_reward:+.2f}"
            )
            print(
                f"          |  Customer: {customer_msg:<30}"
                f"  Frustration: {frustration:.2f}"
            )

    reason = final_info.get("termination_reason", "unknown")
    success = terminated and reason == "resolved"

    if verbose:
        status = "RESOLVED ✓" if success else f"ENDED ({reason})"
        print(f"\n  ── Episode over: {status}")
        print(f"     Total reward: {total_reward:+.4f}  |  Turns: {turn}")

    return {
        "scenario_id": scenario_id,
        "total_reward": float(total_reward),
        "turns": turn,
        "termination_reason": reason,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Menu modes
# ---------------------------------------------------------------------------


def _mode_single_episode(env: CustomerSupportEnv, agent: BaselineSupportAgent) -> None:
    """Run one episode with the default seed."""
    _banner("Single Episode (seed=42)")
    _run_episode(env, agent, seed=DEFAULT_SEED, verbose=True)


def _mode_select_scenario(
    env: CustomerSupportEnv, agent: BaselineSupportAgent
) -> None:
    """Let the user pick a scenario and run one episode with it."""
    _banner("Select Scenario")
    print("\n  Available scenarios:")
    for sid in range(NUM_SCENARIOS):
        print(f"    [{sid}] {SCENARIOS[sid].name}")
    print(f"    [b] Back to menu")

    raw = input("\n  Enter scenario number: ").strip().lower()
    if raw == "b":
        return

    try:
        chosen = int(raw)
        if not 0 <= chosen < NUM_SCENARIOS:
            raise ValueError
    except ValueError:
        print(f"  ✗ Invalid choice '{raw}'. Please enter 0–{NUM_SCENARIOS - 1}.")
        return

    # Find the first seed ≥ 42 that produces the chosen scenario
    for s in range(DEFAULT_SEED, DEFAULT_SEED + 600):
        _, info = env.reset(seed=s)
        if info["scenario_id"] == chosen:
            _run_episode(env, agent, seed=s, verbose=True)
            return

    print(f"  ✗ Could not find a seed for scenario {chosen}. Try again.")


def _mode_multiple_episodes(
    env: CustomerSupportEnv, agent: BaselineSupportAgent
) -> None:
    """Run N episodes and print a summary table."""
    _banner("Multiple Episodes")
    raw = input(
        f"  How many episodes? [default: {DEFAULT_NUM_EPISODES}]: "
    ).strip()

    if raw == "":
        n = DEFAULT_NUM_EPISODES
    else:
        try:
            n = int(raw)
            if n <= 0:
                raise ValueError
        except ValueError:
            print(f"  ✗ Invalid number '{raw}'. Must be a positive integer.")
            return

    raw_seed = input(
        f"  Random seed? [default: {DEFAULT_SEED}]: "
    ).strip()
    if raw_seed == "":
        seed = DEFAULT_SEED
    else:
        try:
            seed = int(raw_seed)
        except ValueError:
            print(f"  ✗ Invalid seed '{raw_seed}'. Must be an integer.")
            return

    print(f"\n  Running {n} episode(s) with seed={seed} …")
    records = evaluate_agent(env, agent, num_episodes=n, seed=seed)
    metrics = compute_metrics(records)

    _section("Results Summary")
    print(f"  Episodes:        {metrics['total_episodes']}")
    print(f"  Resolved:        {metrics['successful_resolutions']}")
    print(f"  Success rate:    {metrics['success_rate']:.1%}")
    print(f"  Mean reward:     {metrics['mean_reward']:+.4f}")
    print(f"  Mean turns:      {metrics['mean_turns']:.2f}")

    _section("Per-Scenario Breakdown")
    for sid, sb in metrics["scenario_breakdown"].items():
        if sb["count"] > 0:
            print(
                f"  [{sid}] {sb['name']:<32}"
                f"  n={sb['count']:3d}  "
                f"rate={sb['success_rate']:5.1%}  "
                f"reward={sb['mean_reward']:+.2f}"
            )


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the interactive demo."""
    env = CustomerSupportEnv(max_turns=DEFAULT_MAX_TURNS)
    agent = BaselineSupportAgent()

    _banner("multi-turn-agent-gym  —  Interactive Demo")
    print(
        "\n  A custom Gymnasium environment where a rule-based agent resolves"
        "\n  simulated customer-support problems over multi-turn conversations."
        "\n"
        "\n  Five scenarios are available:"
    )
    for s in SCENARIOS:
        print(f"    • {s.name}")

    while True:
        _section("Main Menu")
        print("  [1] Run one episode (default seed)")
        print("  [2] Select a specific scenario")
        print("  [3] Run multiple episodes with summary")
        print("  [q] Quit")

        choice = input("\n  Enter choice: ").strip().lower()

        if choice == "1":
            _mode_single_episode(env, agent)
        elif choice == "2":
            _mode_select_scenario(env, agent)
        elif choice == "3":
            _mode_multiple_episodes(env, agent)
        elif choice == "q":
            print("\n  Goodbye.\n")
            sys.exit(0)
        else:
            print(f"  ✗ Unknown option '{choice}'. Enter 1, 2, 3, or q.")


if __name__ == "__main__":
    main()
