"""Customer support scenarios: problem types, conversation state, and simulated user.

This module defines the scenario data model, conversation state tracking,
scenario registry, and the user state machine that simulates customer responses
to agent actions. The environment maintains a hidden state (hidden_issue, diagnosed,
solution_proposed, solution_rejected) that the agent cannot directly access.

Scenario data is kept separate from environment logic (support_env.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    ASK_ACCOUNT_INFO,
    ASK_BILLING_INFO,
    ASK_TECHNICAL_INFO,
    DIAGNOSE,
    PROPOSE_SOLUTION,
    ESCALATE,
    APOLOGIZE,
    MSG_INITIAL_PROBLEM,
    MSG_INFO_PROVIDED,
    MSG_NO_RELEVANT_INFO,
    MSG_SOLUTION_ACCEPTED,
    MSG_SOLUTION_REJECTED,
    MSG_FRUSTRATED,
    MSG_ESCALATION_CONFIRM,
    MSG_DIAGNOSIS_ACK,
    FRUSTRATION_INCREASE_IRRELEVANT,
    FRUSTRATION_INCREASE_REPEATED,
    FRUSTRATION_INCREASE_WRONG_SOLUTION,
    FRUSTRATION_INCREASE_PREMATURE_DIAGNOSIS,
    FRUSTRATION_DECREASE_DIAGNOSIS,
    FRUSTRATION_DECREASE_APOLOGIZE,
    FIELD_TO_ACTION,
)


# ---------------------------------------------------------------------------
# Scenario data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Defines a customer support scenario for the environment.

    Attributes:
        id: Unique integer identifier for the scenario.
        name: Human-readable scenario name.
        customer_issue: Description of the customer's reported problem.
        initial_state: Dict of info the customer provides initially.
        hidden_issue: The actual root cause, hidden from the agent.
        required_fields: Ordered tuple of info field names that must be gathered.
        available_actions: Tuple of action IDs relevant to this scenario.
        ask_field_mapping: Maps action ID to tuple of field names that action gathers.
        resolution_action: The action that resolves the problem (always PROPOSE_SOLUTION).
        resolution_conditions: Tuple of conditions that must be met for resolution.
        expected_resolution: Description of the expected resolution path.
    """

    id: int
    name: str
    customer_issue: str
    initial_state: dict[str, str]
    hidden_issue: str
    required_fields: tuple[str, ...]
    available_actions: tuple[int, ...]
    ask_field_mapping: dict[int, tuple[str, ...]]
    resolution_action: int
    resolution_conditions: tuple[str, ...]
    expected_resolution: str


# ---------------------------------------------------------------------------
# Scenario definitions (5 scenarios)
# ---------------------------------------------------------------------------

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id=0,
        name="Internet Connection Problem",
        customer_issue="My internet keeps dropping every few minutes",
        initial_state={"reported_symptom": "intermittent disconnection"},
        hidden_issue="DSL line degradation causing intermittent sync loss",
        required_fields=("account_id", "device_type", "connection_type"),
        available_actions=(
            ASK_ACCOUNT_INFO, ASK_TECHNICAL_INFO, DIAGNOSE,
            PROPOSE_SOLUTION, ESCALATE, APOLOGIZE,
        ),
        ask_field_mapping={
            ASK_ACCOUNT_INFO: ("account_id",),
            ASK_TECHNICAL_INFO: ("device_type", "connection_type"),
        },
        resolution_action=PROPOSE_SOLUTION,
        resolution_conditions=("all_info_collected", "diagnosed"),
        expected_resolution=(
            "Reset modem, check DSL line filters, escalate to ISP if issue persists"
        ),
    ),
    Scenario(
        id=1,
        name="Router Malfunction",
        customer_issue="My router won't connect to the internet, the light is red",
        initial_state={"reported_symptom": "router not connecting"},
        hidden_issue="Router firmware corrupted after power surge",
        required_fields=("account_id", "router_model", "error_indicator"),
        available_actions=(
            ASK_ACCOUNT_INFO, ASK_TECHNICAL_INFO, DIAGNOSE,
            PROPOSE_SOLUTION, ESCALATE, APOLOGIZE,
        ),
        ask_field_mapping={
            ASK_ACCOUNT_INFO: ("account_id",),
            ASK_TECHNICAL_INFO: ("router_model", "error_indicator"),
        },
        resolution_action=PROPOSE_SOLUTION,
        resolution_conditions=("all_info_collected", "diagnosed"),
        expected_resolution=(
            "Factory reset router, update firmware, replace router if needed"
        ),
    ),
    Scenario(
        id=2,
        name="Account Login Problem",
        customer_issue="I can't log into my account, it says invalid credentials",
        initial_state={"reported_symptom": "login failure"},
        hidden_issue="Account locked due to too many failed login attempts",
        required_fields=("account_id", "email", "login_error"),
        available_actions=(
            ASK_ACCOUNT_INFO, ASK_TECHNICAL_INFO, DIAGNOSE,
            PROPOSE_SOLUTION, ESCALATE, APOLOGIZE,
        ),
        ask_field_mapping={
            ASK_ACCOUNT_INFO: ("account_id", "email"),
            ASK_TECHNICAL_INFO: ("login_error",),
        },
        resolution_action=PROPOSE_SOLUTION,
        resolution_conditions=("all_info_collected", "diagnosed"),
        expected_resolution=(
            "Unlock account, verify identity, reset security token"
        ),
    ),
    Scenario(
        id=3,
        name="Password Reset Request",
        customer_issue="I forgot my password and need to reset it",
        initial_state={"reported_symptom": "forgot password"},
        hidden_issue="Customer hasn't updated recovery email in 2 years",
        required_fields=("account_id", "email", "security_question"),
        available_actions=(
            ASK_ACCOUNT_INFO, DIAGNOSE,
            PROPOSE_SOLUTION, ESCALATE, APOLOGIZE,
        ),
        ask_field_mapping={
            ASK_ACCOUNT_INFO: ("account_id", "email", "security_question"),
        },
        resolution_action=PROPOSE_SOLUTION,
        resolution_conditions=("all_info_collected", "diagnosed"),
        expected_resolution=(
            "Verify identity via security question, send password reset link"
        ),
    ),
    Scenario(
        id=4,
        name="Billing Issue",
        customer_issue="I was charged twice for my last bill",
        initial_state={"reported_symptom": "duplicate charge"},
        hidden_issue="Duplicate charge due to payment processor retry",
        required_fields=("account_id", "last_charge", "payment_method"),
        available_actions=(
            ASK_ACCOUNT_INFO, ASK_BILLING_INFO, DIAGNOSE,
            PROPOSE_SOLUTION, ESCALATE, APOLOGIZE,
        ),
        ask_field_mapping={
            ASK_ACCOUNT_INFO: ("account_id",),
            ASK_BILLING_INFO: ("last_charge", "payment_method"),
        },
        resolution_action=PROPOSE_SOLUTION,
        resolution_conditions=("all_info_collected", "diagnosed"),
        expected_resolution=(
            "Identify duplicate charge, issue refund for the duplicate"
        ),
    ),
)

NUM_SCENARIOS: int = len(SCENARIOS)


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------


@dataclass
class ConversationState:
    """Mutable state tracking a single conversation episode.

    Contains both observable state (included in observations) and hidden state
    (not accessible by the agent).

    Observable (in observation):
        - problem_type (via scenario.id)
        - turn_count
        - info_collected
        - frustration_level
        - user_message_type

    Hidden (NOT in observation):
        - scenario (full scenario object with hidden_issue)
        - diagnosed
        - solution_proposed
        - solution_rejected
        - last_action
        - consecutive_action_count
    """

    scenario: Scenario
    turn_count: int
    info_collected: dict[str, bool]
    frustration_level: float
    last_action: int | None
    consecutive_action_count: int
    diagnosed: bool
    solution_proposed: bool
    solution_rejected: bool
    user_message_type: int

    @property
    def hidden_issue(self) -> str:
        """The actual root cause, hidden from the agent."""
        return self.scenario.hidden_issue

    @property
    def problem_type(self) -> int:
        """The scenario ID, exposed in the observation."""
        return self.scenario.id


def init_state(scenario: Scenario) -> ConversationState:
    """Create a fresh ConversationState for a given scenario.

    Args:
        scenario: The scenario to initialize the conversation with.

    Returns:
        A new ConversationState with all fields at their initial values.
    """
    return ConversationState(
        scenario=scenario,
        turn_count=0,
        info_collected={field: False for field in scenario.required_fields},
        frustration_level=0.0,
        last_action=None,
        consecutive_action_count=0,
        diagnosed=False,
        solution_proposed=False,
        solution_rejected=False,
        user_message_type=MSG_INITIAL_PROBLEM,
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


class ScenarioRegistry:
    """Holds all scenario definitions and samples random scenarios via RNG.

    Attributes:
        _rng: NumPy random generator for reproducible sampling.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        """Initialize the registry with a random generator.

        Args:
            rng: NumPy Generator for reproducible scenario sampling.
        """
        self._rng = rng

    def sample(self) -> Scenario:
        """Sample a random scenario from the registry.

        Returns:
            A randomly selected Scenario instance.
        """
        idx = int(self._rng.integers(0, NUM_SCENARIOS))
        return SCENARIOS[idx]


# ---------------------------------------------------------------------------
# User response
# ---------------------------------------------------------------------------


@dataclass
class UserResponse:
    """The simulated user's response to a single agent action.

    Attributes:
        message_type: The user's message type for the observation.
        info_gathered: Whether new info was collected this turn.
        info_field: Which field was gathered (None if none).
        frustration_delta: Change to frustration level (applied and clamped by env).
        solution_accepted: Whether the proposed solution was accepted.
        solution_rejected: Whether the proposed solution was rejected.
        terminated: Whether this action terminates the episode.
    """

    message_type: int
    info_gathered: bool = False
    info_field: str | None = None
    frustration_delta: float = 0.0
    solution_accepted: bool = False
    solution_rejected: bool = False
    terminated: bool = False


# ---------------------------------------------------------------------------
# User state machine
# ---------------------------------------------------------------------------


class UserStateMachine:
    """Simulates customer responses to agent actions.

    The state machine processes agent actions against the current conversation
    state and produces a UserResponse. It updates the conversation state in-place
    (info_collected, diagnosed, solution flags) and returns the response for
    the environment to apply frustration changes and build observations.
    """

    @staticmethod
    def process_action(
        state: ConversationState, action: int
    ) -> UserResponse:
        """Process an agent action and return the user's response.

        Args:
            state: The current conversation state (mutated in-place).
            action: The agent's action integer.

        Returns:
            UserResponse describing the user's reaction.
        """
        if action in (ASK_ACCOUNT_INFO, ASK_BILLING_INFO, ASK_TECHNICAL_INFO):
            return UserStateMachine._handle_ask(state, action)
        elif action == DIAGNOSE:
            return UserStateMachine._handle_diagnose(state)
        elif action == PROPOSE_SOLUTION:
            return UserStateMachine._handle_propose_solution(state)
        elif action == ESCALATE:
            return UserStateMachine._handle_escalate(state)
        elif action == APOLOGIZE:
            return UserStateMachine._handle_apologize(state)
        else:
            return UserResponse(
                message_type=MSG_NO_RELEVANT_INFO,
                frustration_delta=FRUSTRATION_INCREASE_IRRELEVANT,
            )

    @staticmethod
    def _handle_ask(
        state: ConversationState, action: int
    ) -> UserResponse:
        """Handle ASK_* info gathering actions.

        Looks up the action in the scenario's ask_field_mapping. If the action
        has uncollected mapped fields, marks the first one as collected. If all
        mapped fields are collected or the action has no mapping, returns
        a no-relevant-info response with frustration increase.
        """
        mapping = state.scenario.ask_field_mapping.get(action, ())

        if not mapping:
            return UserResponse(
                message_type=MSG_NO_RELEVANT_INFO,
                frustration_delta=FRUSTRATION_INCREASE_IRRELEVANT,
            )

        for field_name in mapping:
            if not state.info_collected.get(field_name, False):
                state.info_collected[field_name] = True
                return UserResponse(
                    message_type=MSG_INFO_PROVIDED,
                    info_gathered=True,
                    info_field=field_name,
                )

        return UserResponse(
            message_type=MSG_NO_RELEVANT_INFO,
            frustration_delta=FRUSTRATION_INCREASE_REPEATED,
        )

    @staticmethod
    def _handle_diagnose(state: ConversationState) -> UserResponse:
        """Handle the DIAGNOSE action.

        If all required info is collected, the user acknowledges the diagnosis
        and frustration decreases slightly. If info is incomplete, the user
        becomes frustrated and frustration increases.
        """
        state.diagnosed = True

        if UserStateMachine._all_info_collected(state):
            return UserResponse(
                message_type=MSG_DIAGNOSIS_ACK,
                frustration_delta=-FRUSTRATION_DECREASE_DIAGNOSIS,
            )
        return UserResponse(
            message_type=MSG_FRUSTRATED,
            frustration_delta=FRUSTRATION_INCREASE_PREMATURE_DIAGNOSIS,
        )

    @staticmethod
    def _handle_propose_solution(state: ConversationState) -> UserResponse:
        """Handle the PROPOSE_SOLUTION action.

        Solution is accepted only if all required info is collected AND the
        agent has diagnosed the problem. Otherwise the solution is rejected
        and frustration increases.
        """
        state.solution_proposed = True

        if (
            UserStateMachine._all_info_collected(state)
            and state.diagnosed
        ):
            return UserResponse(
                message_type=MSG_SOLUTION_ACCEPTED,
                solution_accepted=True,
                terminated=True,
            )
        state.solution_rejected = True
        return UserResponse(
            message_type=MSG_SOLUTION_REJECTED,
            solution_rejected=True,
            frustration_delta=FRUSTRATION_INCREASE_WRONG_SOLUTION,
        )

    @staticmethod
    def _handle_escalate(state: ConversationState) -> UserResponse:
        """Handle the ESCALATE action. Always terminates the episode."""
        return UserResponse(
            message_type=MSG_ESCALATION_CONFIRM,
            terminated=True,
        )

    @staticmethod
    def _handle_apologize(state: ConversationState) -> UserResponse:
        """Handle the APOLOGIZE action.

        If frustration is above zero, it decreases by the apology delta.
        If frustration is already zero, no change occurs.
        """
        if state.frustration_level > 0.0:
            return UserResponse(
                message_type=MSG_INFO_PROVIDED,
                frustration_delta=-FRUSTRATION_DECREASE_APOLOGIZE,
            )
        return UserResponse(
            message_type=MSG_INFO_PROVIDED,
        )

    @staticmethod
    def _all_info_collected(state: ConversationState) -> bool:
        """Check if all required info fields have been collected."""
        return all(state.info_collected.values())
