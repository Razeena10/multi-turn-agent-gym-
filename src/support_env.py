"""CustomerSupportEnv: Custom Gymnasium environment for multi-turn support conversations.

This module implements the core Gymnasium environment class that simulates a
customer-support agent resolving user problems over multiple conversation turns.

The environment follows Gymnasium conventions:
    - reset() -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)
    - render() -> None
    - close() -> None

The observation is a Dict space containing only agent-visible information.
Hidden state (hidden_issue, diagnosed, solution flags) is never exposed.
"""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

from src.config import (
    NUM_ACTIONS,
    NUM_MESSAGE_TYPES,
    MAX_INFO_FIELDS,
    DEFAULT_MAX_TURNS,
    PROPOSE_SOLUTION,
    ESCALATE,
    APOLOGIZE,
    REWARD_RESOLVE,
    REWARD_INFO_GATHERED,
    REWARD_TURN_PENALTY,
    REWARD_WRONG_SOLUTION,
    REWARD_REPEATED_ACTION,
    REWARD_ESCALATE,
    REWARD_TRUNCATE,
    REWARD_APOLOGIZE_HIGH_FRUSTRATION,
    REWARD_APOLOGIZE_LOW_FRUSTRATION,
    FRUSTRATION_APOLOGY_THRESHOLD,
    TERMINATION_RESOLVED,
    TERMINATION_ESCALATED,
    TERMINATION_TRUNCATED,
)
from src.support_scenarios import (
    ConversationState,
    ScenarioRegistry,
    UserResponse,
    UserStateMachine,
    init_state,
    NUM_SCENARIOS,
)


class CustomerSupportEnv(gymnasium.Env):
    """Custom Gymnasium environment for customer-support conversations.

    The environment simulates a multi-turn conversation between an agent and a
    simulated customer. The agent selects actions to gather information, diagnose,
    and resolve the customer's problem. The environment provides structured
    observations, scalar rewards, and clear termination signals.

    Hidden state (hidden_issue, diagnosed, solution flags) is maintained
    internally and never exposed in observations or info dicts.

    Attributes:
        max_turns: Maximum number of turns per episode.
        action_space: Discrete(NUM_ACTIONS) space for agent actions.
        observation_space: Dict space with agent-visible information only.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Initialize the environment.

        Args:
            max_turns: Maximum number of turns per episode. Must be positive.

        Raises:
            ValueError: If max_turns is not a positive integer.
        """
        super().__init__()
        if not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError(
                f"max_turns must be a positive integer, got {max_turns}"
            )

        self.max_turns: int = max_turns
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict({
            "customer_issue": spaces.Discrete(NUM_SCENARIOS, start=0),
            "collected_information": spaces.MultiBinary(MAX_INFO_FIELDS),
            "missing_information": spaces.MultiBinary(MAX_INFO_FIELDS),
            "turn_count": spaces.Discrete(max_turns + 1, start=0),
            "frustration": spaces.Box(
                low=0.0, high=1.0, shape=(), dtype=np.float32
            ),
            "previous_customer_response": spaces.Discrete(
                NUM_MESSAGE_TYPES, start=0
            ),
            "available_actions": spaces.MultiBinary(NUM_ACTIONS),
            "problem_type": spaces.Discrete(NUM_SCENARIOS, start=0),
        })

        self._state: ConversationState | None = None
        self._episode_active: bool = False
        self._episode_reward: float = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment to start a new episode.

        Calls the parent reset for proper RNG seeding, samples a scenario,
        initializes conversation state, and returns the initial observation.

        Args:
            seed: Optional seed for reproducible scenario sampling.
            options: Optional configuration dict (unused).

        Returns:
            Tuple of (observation, info) for the initial state.
        """
        super().reset(seed=seed)

        registry = ScenarioRegistry(self.np_random)
        scenario = registry.sample()
        self._state = init_state(scenario)
        self._episode_active = True
        self._episode_reward = 0.0

        observation = self._build_observation()
        info: dict = {
            "scenario_id": scenario.id,
            "turn_count": 0,
        }
        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute one step in the environment.

        Validates the action, increments the turn counter, processes the action
        through the UserStateMachine, computes the reward, and checks termination.
        Validation occurs before any state mutation (AC-12.08).

        Args:
            action: The agent's action integer (must be in action_space).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            terminated and truncated are never both True.

        Raises:
            RuntimeError: If the episode has ended (call reset() first).
            ValueError: If the action is not a valid integer in the action space.
        """
        # --- Validation phase (before any state mutation) ---
        if not self._episode_active:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )
        if not isinstance(action, (int, np.integer)):
            raise ValueError(
                f"Action must be an integer, got {type(action).__name__}"
            )
        if action not in self.action_space:
            raise ValueError(
                f"Invalid action {action}. Must be in [0, {self.action_space.n})."
            )

        # --- State update phase ---
        state = self._state
        assert state is not None
        state.turn_count += 1

        response = UserStateMachine.process_action(state, int(action))

        # Update action tracking before reward (for repeat detection)
        if action == state.last_action:
            state.consecutive_action_count += 1
        else:
            state.last_action = int(action)
            state.consecutive_action_count = 1

        # Compute reward before applying frustration delta so that
        # the apologize reward checks pre-apology frustration level
        reward = self._compute_reward(int(action), response)        # Apply frustration delta with clamping [0.0, 1.0]
        state.frustration_level = max(
            0.0,
            min(1.0, state.frustration_level + response.frustration_delta),
        )
        state.user_message_type = response.message_type
        self._episode_reward += reward

        # --- Termination check (mutually exclusive: never both True) ---
        terminated = False
        truncated = False
        termination_reason: str | None = None

        if action == PROPOSE_SOLUTION and response.solution_accepted:
            terminated = True
            termination_reason = TERMINATION_RESOLVED
        elif action == ESCALATE:
            terminated = True
            termination_reason = TERMINATION_ESCALATED
        elif state.turn_count >= self.max_turns:
            truncated = True
            termination_reason = TERMINATION_TRUNCATED
            reward += REWARD_TRUNCATE
            self._episode_reward += REWARD_TRUNCATE

        if terminated or truncated:
            self._episode_active = False

        # --- Build outputs ---
        observation = self._build_observation()
        info: dict = {
            "scenario_id": state.scenario.id,
            "turn_count": state.turn_count,
            "success": bool(
                terminated and termination_reason == TERMINATION_RESOLVED
            ),
        }
        if terminated or truncated:
            assert termination_reason is not None
            info["termination_reason"] = termination_reason
            info["episode_reward"] = self._episode_reward

        return observation, float(reward), terminated, truncated, info

    def _compute_reward(
        self, action: int, response: UserResponse
    ) -> float:
        """Compute the scalar reward for the current step.

        Uses config.py reward constants. The turn penalty is always applied.
        Action-specific rewards are added based on the action and response.

        Repeated-action penalty is progress-aware: REWARD_REPEATED_ACTION is
        only applied when the same action is repeated AND no meaningful progress
        was made (no new information gathered). This allows productive multi-field
        ASK actions (e.g., ASK_ACCOUNT_INFO gathering three fields in the Password
        Reset scenario) without penalty, while still penalising unproductive loops
        such as repeating PROPOSE_SOLUTION or repeating an ASK after all its fields
        are already collected. APOLOGIZE is always exempt.

        Args:
            action: The agent's action integer.
            response: The user's response from UserStateMachine.

        Returns:
            The scalar reward for this step (always finite).
        """
        reward = REWARD_TURN_PENALTY

        # Progress-aware repeated action penalty (except APOLOGIZE).
        # Fires only when the same action is repeated AND no new info was gathered.
        if (
            action == self._state.last_action
            and action != APOLOGIZE
            and self._state.consecutive_action_count >= 2
            and not response.info_gathered
        ):
            reward += REWARD_REPEATED_ACTION

        # Info gathering bonus
        if response.info_gathered:
            reward += REWARD_INFO_GATHERED

        # Propose solution outcomes
        if action == PROPOSE_SOLUTION:
            if response.solution_accepted:
                reward += REWARD_RESOLVE
            else:
                reward += REWARD_WRONG_SOLUTION

        # Escalation
        if action == ESCALATE:
            reward += REWARD_ESCALATE

        # Apologize (checks pre-apology frustration level)
        if action == APOLOGIZE:
            if self._state.frustration_level > FRUSTRATION_APOLOGY_THRESHOLD:
                reward += REWARD_APOLOGIZE_HIGH_FRUSTRATION
            else:
                reward += REWARD_APOLOGIZE_LOW_FRUSTRATION

        return float(reward)

    def _build_observation(self) -> dict:
        """Build the agent-visible observation from the current state.

        Only includes fields the agent is allowed to see. Hidden state
        (hidden_issue, diagnosed, solution flags, last_action) is never
        included in the observation.

        Returns:
            A dict matching the observation_space.
        """
        state = self._state
        assert state is not None
        scenario = state.scenario

        # Collected information as fixed-length binary vector
        collected = np.zeros(MAX_INFO_FIELDS, dtype=np.int8)
        for i, field_name in enumerate(scenario.required_fields):
            collected[i] = 1 if state.info_collected.get(field_name, False) else 0

        # Missing information (inverse of collected)
        missing = (1 - collected).astype(np.int8)

        # Available actions as binary mask
        available = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for action_id in scenario.available_actions:
            available[action_id] = 1

        return {
            "customer_issue": np.int64(scenario.id),
            "collected_information": collected,
            "missing_information": missing,
            "turn_count": np.int64(state.turn_count),
            "frustration": np.array(state.frustration_level, dtype=np.float32),
            "previous_customer_response": np.int64(state.user_message_type),
            "available_actions": available,
            "problem_type": np.int64(scenario.id),
        }

    def render(self) -> None:
        """Print a human-readable summary of the current conversation state.

        This is for debugging and human inspection only. The output is never
        passed to the agent as an observation.
        """
        if self._state is None:
            print("No active episode.")
            return

        state = self._state
        scenario = state.scenario
        from src.config import ACTION_NAMES, MESSAGE_TYPE_NAMES

        collected_count = sum(state.info_collected.values())
        total_count = len(state.info_collected)

        print("=== Conversation State ===")
        print(f"  Problem:        {scenario.name}")
        print(f"  Turn:           {state.turn_count}")
        print(f"  Info Collected: {collected_count}/{total_count}")
        print(f"  Frustration:    {state.frustration_level:.2f}")
        msg_name = MESSAGE_TYPE_NAMES.get(
            state.user_message_type, "Unknown"
        )
        print(f"  Last User Msg:  {msg_name}")
        last_name = (
            ACTION_NAMES.get(state.last_action, "None")
            if state.last_action is not None
            else "None"
        )
        print(f"  Last Action:    {last_name}")

    def close(self) -> None:
        """Clean up the environment. No external resources to release."""
        self._episode_active = False
        self._state = None
