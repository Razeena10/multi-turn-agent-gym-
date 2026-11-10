"""Evaluation harness for running episodes and collecting per-episode data.

This module implements the evaluation orchestrator that runs a specified
number of episodes using an agent against the environment, collecting
per-episode records with rewards, turns, and termination information.

The evaluation uses reproducible per-episode seeding (seed + episode_id)
and a defensive maximum-step safeguard to prevent infinite loops.

Episode records contain only observable data — no hidden state is accessed.
"""

from __future__ import annotations

from src.config import (
    DEFAULT_NUM_EPISODES,
    DEFAULT_SEED,
    DEFAULT_MAX_TURNS,
    TERMINATION_RESOLVED,
)
from src.support_env import CustomerSupportEnv
from src.baseline_agent import BaselineSupportAgent


def evaluate_agent(
    env: CustomerSupportEnv,
    agent: BaselineSupportAgent,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed: int = DEFAULT_SEED,
    max_steps: int = 1000,
) -> list[dict]:
    """Run evaluation episodes and collect per-episode records.

    For each episode, resets both the environment and the agent, then
    runs the agent-environment interaction loop until the episode
    terminates, truncates, or the defensive step safeguard is reached.

    Args:
        env: The CustomerSupportEnv instance to evaluate against.
        agent: The agent with a select_action(observation) -> int method.
        num_episodes: Number of episodes to run. Must be positive.
        seed: Base random seed. Each episode uses seed + episode_id.
        max_steps: Defensive safeguard against infinite loops.

    Returns:
        A list of episode record dicts, each containing:
            episode_id, scenario_id, total_reward, number_of_turns,
            success, terminated, truncated, termination_reason.

    Raises:
        ValueError: If num_episodes is not a positive integer.
        ValueError: If seed is not an integer.
    """
    if not isinstance(num_episodes, int) or num_episodes <= 0:
        raise ValueError(
            f"num_episodes must be a positive integer, got {num_episodes}"
        )
    if not isinstance(seed, int):
        raise ValueError(
            f"seed must be an integer, got {type(seed).__name__}"
        )

    records: list[dict] = []

    for episode_id in range(num_episodes):
        episode_seed = seed + episode_id
        obs, reset_info = env.reset(seed=episode_seed)
        agent.reset()

        total_reward = 0.0
        num_turns = 0
        terminated = False
        truncated = False
        termination_reason: str | None = None

        while not terminated and not truncated and num_turns < max_steps:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            num_turns += 1

        if terminated or truncated:
            termination_reason = step_info.get(
                "termination_reason", "unknown"
            )
        else:
            termination_reason = "max_steps_reached"
            truncated = True

        scenario_id = reset_info.get("scenario_id", -1)
        success = bool(
            terminated and termination_reason == TERMINATION_RESOLVED
        )

        records.append({
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "total_reward": float(total_reward),
            "number_of_turns": num_turns,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": termination_reason,
        })

    return records
