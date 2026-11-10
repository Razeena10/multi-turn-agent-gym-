"""Centralized configuration constants for the multi-turn-agent-gym project.

All reward values, action definitions, message types, problem type parameters,
and default settings are defined here. No logic or side effects — just constants.

This module is the single source of truth for tunable parameters, satisfying
NFR-01.4 (constants in one location, no scattered magic numbers).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Action constants (FR-04 AC-04.02)
# ---------------------------------------------------------------------------

ASK_ACCOUNT_INFO: int = 0
ASK_BILLING_INFO: int = 1
ASK_TECHNICAL_INFO: int = 2
DIAGNOSE: int = 3
PROPOSE_SOLUTION: int = 4
ESCALATE: int = 5
APOLOGIZE: int = 6

NUM_ACTIONS: int = 7

ACTION_NAMES: dict[int, str] = {
    ASK_ACCOUNT_INFO: "ASK_ACCOUNT_INFO",
    ASK_BILLING_INFO: "ASK_BILLING_INFO",
    ASK_TECHNICAL_INFO: "ASK_TECHNICAL_INFO",
    DIAGNOSE: "DIAGNOSE",
    PROPOSE_SOLUTION: "PROPOSE_SOLUTION",
    ESCALATE: "ESCALATE",
    APOLOGIZE: "APOLOGIZE",
}

# ---------------------------------------------------------------------------
# Message type constants (Design Section 5.3)
# ---------------------------------------------------------------------------

MSG_INITIAL_PROBLEM: int = 0
MSG_INFO_PROVIDED: int = 1
MSG_NO_RELEVANT_INFO: int = 2
MSG_SOLUTION_ACCEPTED: int = 3
MSG_SOLUTION_REJECTED: int = 4
MSG_FRUSTRATED: int = 5
MSG_ESCALATION_CONFIRM: int = 6
MSG_DIAGNOSIS_ACK: int = 7

NUM_MESSAGE_TYPES: int = 8

MESSAGE_TYPE_NAMES: dict[int, str] = {
    MSG_INITIAL_PROBLEM: "MSG_INITIAL_PROBLEM",
    MSG_INFO_PROVIDED: "MSG_INFO_PROVIDED",
    MSG_NO_RELEVANT_INFO: "MSG_NO_RELEVANT_INFO",
    MSG_SOLUTION_ACCEPTED: "MSG_SOLUTION_ACCEPTED",
    MSG_SOLUTION_REJECTED: "MSG_SOLUTION_REJECTED",
    MSG_FRUSTRATED: "MSG_FRUSTRATED",
    MSG_ESCALATION_CONFIRM: "MSG_ESCALATION_CONFIRM",
    MSG_DIAGNOSIS_ACK: "MSG_DIAGNOSIS_ACK",
}

# ---------------------------------------------------------------------------
# Reward constants (FR-05 AC-05.01 through AC-05.08)
# ---------------------------------------------------------------------------

REWARD_RESOLVE: float = 10.0
REWARD_INFO_GATHERED: float = 1.0
REWARD_TURN_PENALTY: float = -0.5
REWARD_WRONG_SOLUTION: float = -3.0
REWARD_REPEATED_ACTION: float = -2.0
REWARD_ESCALATE: float = 2.0
REWARD_TRUNCATE: float = -5.0
REWARD_APOLOGIZE_HIGH_FRUSTRATION: float = 0.5
REWARD_APOLOGIZE_LOW_FRUSTRATION: float = 0.0

# ---------------------------------------------------------------------------
# Frustration thresholds and deltas (Design Section 9.5)
# ---------------------------------------------------------------------------

FRUSTRATION_APOLOGY_THRESHOLD: float = 0.5
FRUSTRATION_HIGH_THRESHOLD: float = 0.7

FRUSTRATION_INCREASE_IRRELEVANT: float = 0.15
FRUSTRATION_INCREASE_REPEATED: float = 0.1
FRUSTRATION_INCREASE_WRONG_SOLUTION: float = 0.2
FRUSTRATION_INCREASE_PREMATURE_DIAGNOSIS: float = 0.1
FRUSTRATION_DECREASE_DIAGNOSIS: float = 0.05
FRUSTRATION_DECREASE_APOLOGIZE: float = 0.3

# ---------------------------------------------------------------------------
# Default parameters (FR-07 AC-07.01, FR-09 AC-09.01)
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS: int = 15
DEFAULT_SEED: int = 42
DEFAULT_NUM_EPISODES: int = 100

# ---------------------------------------------------------------------------
# Observation space parameters (FR-03, Design Section 5.4)
# ---------------------------------------------------------------------------

# Maximum number of required info fields across all problem types.
# Used for the fixed-length MultiBinary info_collected vector.
# Problem types have at most 3 required fields; 4 provides padding.
MAX_INFO_FIELDS: int = 4

# ---------------------------------------------------------------------------
# Field-to-action reverse lookup (Design Section 10.3)
# Maps each info field name to the action that gathers it.
# ---------------------------------------------------------------------------

FIELD_TO_ACTION: dict[str, int] = {
    "account_id": ASK_ACCOUNT_INFO,
    "email": ASK_ACCOUNT_INFO,
    "security_question": ASK_ACCOUNT_INFO,
    "last_charge": ASK_BILLING_INFO,
    "payment_method": ASK_BILLING_INFO,
    "device_type": ASK_TECHNICAL_INFO,
    "connection_type": ASK_TECHNICAL_INFO,
    "router_model": ASK_TECHNICAL_INFO,
    "error_indicator": ASK_TECHNICAL_INFO,
    "login_error": ASK_TECHNICAL_INFO,
}

# ---------------------------------------------------------------------------
# Termination reason strings (FR-06 AC-06.05)
# ---------------------------------------------------------------------------

TERMINATION_RESOLVED: str = "resolved"
TERMINATION_ESCALATED: str = "escalated"
TERMINATION_TRUNCATED: str = "truncated"
TERMINATION_ERROR: str = "error"
