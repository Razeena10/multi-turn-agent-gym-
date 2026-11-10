"""Rule-based baseline agent for the customer support environment.

This module implements a deterministic rule-based agent that follows a
priority-cascade policy to resolve customer problems. It uses only
observation data and public scenario definitions — never accessing
environment internals, hidden state, or private attributes.

Policy priority order:
    1. If frustration > 0.7, apologize to de-escalate.
    2. If solution was rejected, escalate to human agent.
    3. If information is missing, ask for the first uncollected field.
    4. If all info collected and not yet diagnosed, diagnose.
    5. If diagnosed and not yet proposed, propose solution.
    6. Fallback: escalate.

Anti-repetition logic prevents infinite loops while allowing multi-field
actions to repeat productively (e.g., ASK_ACCOUNT_INFO may gather 3 fields
in the Password Reset scenario).
"""

from __future__ import annotations

import numpy as np

from src.config import (
    APOLOGIZE,
    DIAGNOSE,
    ESCALATE,
    PROPOSE_SOLUTION,
    FRUSTRATION_HIGH_THRESHOLD,
    FIELD_TO_ACTION,
    MSG_SOLUTION_REJECTED,
    NUM_ACTIONS,
)
from src.support_scenarios import SCENARIOS


class BaselineSupportAgent:
    """Deterministic rule-based baseline agent for customer support.

    Uses a priority-cascade policy to select actions based on the current
    observation. Maintains minimal internal state (diagnosis/proposal flags
    and action tracking) that is reset between episodes.

    The agent never accesses environment internals, hidden state, or
    private attributes. It uses only the observation dict and public
    scenario definitions (required_fields, ask_field_mapping, FIELD_TO_ACTION).

    Attributes:
        _has_diagnosed: Whether the agent has performed DIAGNOSE this episode.
        _has_proposed: Whether the agent has proposed a solution this episode.
        _last_action: The last action taken (for anti-repetition).
        _consecutive_count: How many times _last_action was repeated.
    """

    _OBSERVATION_KEYS: frozenset[str] = frozenset({
        "customer_issue", "collected_information", "missing_information",
        "turn_count", "frustration", "previous_customer_response",
        "available_actions", "problem_type",
    })

    def __init__(self) -> None:
        """Initialize the agent with cleared internal state."""
        self.reset()

    def reset(self) -> None:
        """Clear internal state for a new episode.

        Resets diagnosis/proposal flags and action tracking to their
        initial values. Must be called before each new episode.
        """
        self._has_diagnosed: bool = False
        self._has_proposed: bool = False
        self._last_action: int | None = None
        self._consecutive_count: int = 0

    def select_action(self, observation: dict) -> int:
        """Select an action given the current observation.

        Args:
            observation: A dict with the 8 agent-visible observation fields.
                Must not include any hidden state.

        Returns:
            An integer action ID in [0, NUM_ACTIONS).

        Raises:
            KeyError: If the observation is missing required keys.
        """
        self._validate_observation(observation)
        action = self._decide_action(observation)
        action = self._check_repetition(action, observation)
        self._update_tracking(action)
        return action

    def _validate_observation(self, observation: dict) -> None:
        """Validate that the observation contains all required keys.

        Args:
            observation: The observation dict to validate.

        Raises:
            KeyError: If any required key is missing.
        """
        for key in self._OBSERVATION_KEYS:
            if key not in observation:
                raise KeyError(f"Observation missing required key: '{key}'")

    def _decide_action(self, observation: dict) -> int:
        """Determine the action via priority cascade.

        Args:
            observation: Validated observation dict.

        Returns:
            The selected action integer.
        """
        frustration = float(observation["frustration"])
        previous_response = int(observation["previous_customer_response"])
        problem_type = int(observation["problem_type"])
        collected = observation["collected_information"]

        # 1. High frustration → apologize to de-escalate
        if frustration > FRUSTRATION_HIGH_THRESHOLD:
            return APOLOGIZE

        # 2. Solution was rejected → escalate
        if previous_response == MSG_SOLUTION_REJECTED:
            return ESCALATE

        # 3. Missing info → ask for first uncollected field
        scenario = SCENARIOS[problem_type]
        for i, field_name in enumerate(scenario.required_fields):
            if i < len(collected) and int(collected[i]) == 0:
                return FIELD_TO_ACTION[field_name]

        # 4. All info collected, not diagnosed → diagnose
        if not self._has_diagnosed:
            return DIAGNOSE

        # 5. Diagnosed, not proposed → propose solution
        if not self._has_proposed:
            return PROPOSE_SOLUTION

        # 6. Fallback → escalate
        return ESCALATE

    def _check_repetition(
        self, action: int, observation: dict
    ) -> int:
        """Override action if it would be repeated unproductively.

        If the same non-apologize action would be taken for the 3rd
        consecutive time AND it can no longer gather new info, escalate
        instead to prevent infinite loops. Multi-field actions (like
        ASK_ACCOUNT_INFO in the Password Reset scenario) are allowed to
        repeat while they remain productive.

        Args:
            action: The proposed action.
            observation: The current observation.

        Returns:
            The action to take (possibly overridden to ESCALATE).
        """
        if (
            action == self._last_action
            and action != APOLOGIZE
            and self._consecutive_count >= 2
        ):
            if not self._is_action_productive(action, observation):
                return ESCALATE
        return action

    def _is_action_productive(
        self, action: int, observation: dict
    ) -> bool:
        """Check if the action can still gather new info.

        Looks up the scenario's ask_field_mapping to see if the action
        has any uncollected mapped fields remaining.

        Args:
            action: The action to check.
            observation: The current observation.

        Returns:
            True if the action can gather at least one uncollected field.
        """
        problem_type = int(observation["problem_type"])
        collected = observation["collected_information"]
        scenario = SCENARIOS[problem_type]

        mapping = scenario.ask_field_mapping.get(action, ())
        for field_name in mapping:
            if field_name in scenario.required_fields:
                pos = scenario.required_fields.index(field_name)
                if pos < len(collected) and int(collected[pos]) == 0:
                    return True
        return False

    def _update_tracking(self, action: int) -> None:
        """Update internal tracking state after selecting an action.

        Args:
            action: The final action that was selected.
        """
        if action == self._last_action:
            self._consecutive_count += 1
        else:
            self._last_action = action
            self._consecutive_count = 1

        if action == DIAGNOSE:
            self._has_diagnosed = True
        elif action == PROPOSE_SOLUTION:
            self._has_proposed = True
