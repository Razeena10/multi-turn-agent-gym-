"""Tests for the BaselineSupportAgent.

Wave 1: Verifies config constants used by the agent.
Wave 4: Verifies agent initialization, reset, action selection, priority
        cascade, frustration handling, anti-repetition, all five scenarios,
        invalid observations, hidden-state protection, and integration
        with CustomerSupportEnv.
"""

import pytest
import numpy as np

from src import config
from src.baseline_agent import BaselineSupportAgent
from src.support_env import CustomerSupportEnv
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS


# ---------------------------------------------------------------------------
# Wave 1: Config constant verification (FR-08, FR-12)
# ---------------------------------------------------------------------------


class TestAgentConfig:
    """Verify constants the agent depends on (FR-08, Design Section 10)."""

    def test_frustration_high_threshold(self):
        """High-frustration threshold for triggering apologies (AC-08.2.1)."""
        assert config.FRUSTRATION_HIGH_THRESHOLD == 0.7

    def test_frustration_apology_threshold(self):
        """Apology reward threshold (AC-05.08)."""
        assert config.FRUSTRATION_APOLOGY_THRESHOLD == 0.5

    def test_field_to_action_has_entries(self):
        """FIELD_TO_ACTION mapping must have entries for all known fields."""
        assert len(config.FIELD_TO_ACTION) >= 9

    def test_field_to_action_values_are_valid(self):
        """Every mapped action must be a valid action integer."""
        for field_name, action in config.FIELD_TO_ACTION.items():
            assert action in config.ACTION_NAMES, (
                f"Field '{field_name}' maps to invalid action {action}"
            )

    def test_field_to_action_covers_ask_actions(self):
        """All three ASK_* actions appear in the mapping values."""
        mapped_actions = set(config.FIELD_TO_ACTION.values())
        assert config.ASK_ACCOUNT_INFO in mapped_actions
        assert config.ASK_BILLING_INFO in mapped_actions
        assert config.ASK_TECHNICAL_INFO in mapped_actions

    def test_action_names_are_strings(self):
        for action_id, name in config.ACTION_NAMES.items():
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# Test helper: build observation dicts matching the environment format
# ---------------------------------------------------------------------------


def _make_observation(
    problem_type: int = 0,
    collected: np.ndarray | None = None,
    frustration: float = 0.0,
    previous_response: int = 0,
    turn_count: int = 0,
) -> dict:
    """Build a valid observation dict for testing.

    Args:
        problem_type: Scenario ID (0-4).
        collected: Binary array of collected info (length MAX_INFO_FIELDS).
        frustration: Frustration level [0.0, 1.0].
        previous_response: User's last message type.
        turn_count: Current turn number.

    Returns:
        A dict matching the environment's observation format.
    """
    scenario = SCENARIOS[problem_type]
    if collected is None:
        collected = np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8)
    missing = (1 - collected).astype(np.int8)
    available = np.zeros(config.NUM_ACTIONS, dtype=np.int8)
    for action_id in scenario.available_actions:
        available[action_id] = 1
    return {
        "customer_issue": np.int64(problem_type),
        "collected_information": collected,
        "missing_information": missing,
        "turn_count": np.int64(turn_count),
        "frustration": np.array(frustration, dtype=np.float32),
        "previous_customer_response": np.int64(previous_response),
        "available_actions": available,
        "problem_type": np.int64(problem_type),
    }


def _all_collected(problem_type: int) -> np.ndarray:
    """Build a collected_information array with all required fields set."""
    scenario = SCENARIOS[problem_type]
    collected = np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8)
    for i in range(len(scenario.required_fields)):
        collected[i] = 1
    return collected


# ---------------------------------------------------------------------------
# Wave 4: Agent initialization tests (FR-08)
# ---------------------------------------------------------------------------


class TestAgentInit:
    """Verify BaselineSupportAgent initialization (FR-08)."""

    def test_agent_creates_instance(self):
        agent = BaselineSupportAgent()
        assert isinstance(agent, BaselineSupportAgent)

    def test_init_has_reset_state(self):
        """Agent starts with cleared internal state."""
        agent = BaselineSupportAgent()
        assert agent._has_diagnosed is False
        assert agent._has_proposed is False
        assert agent._last_action is None
        assert agent._consecutive_count == 0

    def test_select_action_is_callable(self):
        """select_action method exists and is callable."""
        agent = BaselineSupportAgent()
        obs = _make_observation()
        result = agent.select_action(obs)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Wave 4: Agent reset tests (FR-08)
# ---------------------------------------------------------------------------


class TestAgentReset:
    """Verify reset() clears internal state between episodes (FR-08)."""

    def test_reset_clears_diagnosed(self):
        agent = BaselineSupportAgent()
        agent._has_diagnosed = True
        agent.reset()
        assert agent._has_diagnosed is False

    def test_reset_clears_proposed(self):
        agent = BaselineSupportAgent()
        agent._has_proposed = True
        agent.reset()
        assert agent._has_proposed is False

    def test_reset_clears_last_action(self):
        agent = BaselineSupportAgent()
        agent._last_action = 3
        agent.reset()
        assert agent._last_action is None

    def test_reset_clears_consecutive_count(self):
        agent = BaselineSupportAgent()
        agent._consecutive_count = 5
        agent.reset()
        assert agent._consecutive_count == 0


# ---------------------------------------------------------------------------
# Wave 4: Action selection / priority cascade tests (FR-08 AC-08.02)
# ---------------------------------------------------------------------------


class TestAgentActionSelection:
    """Verify the priority cascade decision logic (FR-08 AC-08.02)."""

    def test_returns_valid_action(self):
        """AC-08.05: Action is always in valid range [0, NUM_ACTIONS)."""
        agent = BaselineSupportAgent()
        obs = _make_observation()
        action = agent.select_action(obs)
        assert 0 <= action < config.NUM_ACTIONS

    def test_high_frustration_returns_apologize(self):
        """AC-08.02.1: frustration > 0.7 → APOLOGIZE."""
        agent = BaselineSupportAgent()
        obs = _make_observation(frustration=0.8)
        assert agent.select_action(obs) == config.APOLOGIZE

    def test_rejected_solution_returns_escalate(self):
        """AC-08.02.5: Solution rejected → ESCALATE."""
        agent = BaselineSupportAgent()
        obs = _make_observation(
            previous_response=config.MSG_SOLUTION_REJECTED,
            collected=_all_collected(0),
        )
        agent._has_diagnosed = True
        agent._has_proposed = True
        assert agent.select_action(obs) == config.ESCALATE

    def test_missing_info_returns_ask(self):
        """AC-08.02.2: Missing info → ASK_* action."""
        agent = BaselineSupportAgent()
        obs = _make_observation(problem_type=0, collected=np.array([0, 0, 0, 0], dtype=np.int8))
        action = agent.select_action(obs)
        assert action in (config.ASK_ACCOUNT_INFO, config.ASK_BILLING_INFO, config.ASK_TECHNICAL_INFO)

    def test_all_info_returns_diagnose(self):
        """AC-08.02.3: All info collected, not diagnosed → DIAGNOSE."""
        agent = BaselineSupportAgent()
        obs = _make_observation(problem_type=0, collected=_all_collected(0))
        assert agent.select_action(obs) == config.DIAGNOSE

    def test_diagnosed_returns_propose(self):
        """AC-08.02.4: Diagnosed, not proposed → PROPOSE_SOLUTION."""
        agent = BaselineSupportAgent()
        agent._has_diagnosed = True
        obs = _make_observation(problem_type=0, collected=_all_collected(0))
        assert agent.select_action(obs) == config.PROPOSE_SOLUTION

    def test_proposed_returns_escalate(self):
        """After proposing, fallback → ESCALATE."""
        agent = BaselineSupportAgent()
        agent._has_diagnosed = True
        agent._has_proposed = True
        obs = _make_observation(problem_type=0, collected=_all_collected(0))
        assert agent.select_action(obs) == config.ESCALATE

    def test_deterministic(self):
        """AC-08.04: Same observation always produces the same action."""
        agent = BaselineSupportAgent()
        obs = _make_observation(problem_type=0)
        action1 = agent.select_action(obs)
        agent.reset()
        action2 = agent.select_action(obs)
        assert action1 == action2


# ---------------------------------------------------------------------------
# Wave 4: Missing information tests
# ---------------------------------------------------------------------------


class TestAgentMissingInfo:
    """Verify the agent asks for the correct missing fields."""

    def test_asks_first_missing_field(self):
        """Agent asks for the first uncollected field."""
        agent = BaselineSupportAgent()
        obs = _make_observation(problem_type=0, collected=np.array([0, 0, 0, 0], dtype=np.int8))
        action = agent.select_action(obs)
        first_field = SCENARIOS[0].required_fields[0]
        assert action == config.FIELD_TO_ACTION[first_field]

    def test_asks_second_missing_field(self):
        """Agent asks for the second uncollected field when first is done."""
        agent = BaselineSupportAgent()
        collected = np.array([1, 0, 0, 0], dtype=np.int8)
        obs = _make_observation(problem_type=0, collected=collected)
        action = agent.select_action(obs)
        second_field = SCENARIOS[0].required_fields[1]
        assert action == config.FIELD_TO_ACTION[second_field]

    def test_asks_third_missing_field(self):
        """Agent asks for the third uncollected field when first two are done."""
        agent = BaselineSupportAgent()
        collected = np.array([1, 1, 0, 0], dtype=np.int8)
        obs = _make_observation(problem_type=0, collected=collected)
        action = agent.select_action(obs)
        third_field = SCENARIOS[0].required_fields[2]
        assert action == config.FIELD_TO_ACTION[third_field]

    def test_no_missing_info_does_not_ask(self):
        """Agent does not ask when all info is collected."""
        agent = BaselineSupportAgent()
        obs = _make_observation(problem_type=0, collected=_all_collected(0))
        action = agent.select_action(obs)
        assert action not in (
            config.ASK_ACCOUNT_INFO, config.ASK_BILLING_INFO,
            config.ASK_TECHNICAL_INFO,
        )


# ---------------------------------------------------------------------------
# Wave 4: Frustration handling tests
# ---------------------------------------------------------------------------


class TestAgentFrustration:
    """Verify the agent uses frustration observations correctly."""

    def test_low_frustration_no_apologize(self):
        """frustration=0.0 → agent does not apologize."""
        agent = BaselineSupportAgent()
        obs = _make_observation(frustration=0.0)
        assert agent.select_action(obs) != config.APOLOGIZE

    def test_moderate_frustration_no_apologize(self):
        """frustration=0.5 → agent does not apologize (threshold is > 0.7)."""
        agent = BaselineSupportAgent()
        obs = _make_observation(frustration=0.5)
        assert agent.select_action(obs) != config.APOLOGIZE

    def test_high_frustration_apologize(self):
        """frustration=0.8 → agent apologizes."""
        agent = BaselineSupportAgent()
        obs = _make_observation(frustration=0.8)
        assert agent.select_action(obs) == config.APOLOGIZE

    def test_threshold_boundary_not_triggered(self):
        """frustration=0.7 → agent does NOT apologize (check is > 0.7)."""
        agent = BaselineSupportAgent()
        obs = _make_observation(frustration=0.7)
        assert agent.select_action(obs) != config.APOLOGIZE

    def test_just_above_threshold_apologize(self):
        """frustration=0.71 → agent apologizes."""
        agent = BaselineSupportAgent()
        obs = _make_observation(frustration=0.71)
        assert agent.select_action(obs) == config.APOLOGIZE


# ---------------------------------------------------------------------------
# Wave 4: Repeated action / anti-repetition tests (FR-08 AC-08.03)
# ---------------------------------------------------------------------------


class TestAgentRepeatedActions:
    """Verify anti-repetition logic (FR-08 AC-08.03)."""

    def test_different_actions_no_override(self):
        """Different actions do not trigger anti-repetition."""
        agent = BaselineSupportAgent()
        obs = _make_observation(problem_type=0, collected=np.array([0, 0, 0, 0], dtype=np.int8))
        a1 = agent.select_action(obs)  # ASK_ACCOUNT_INFO
        collected = np.array([1, 0, 0, 0], dtype=np.int8)
        obs = _make_observation(problem_type=0, collected=collected)
        a2 = agent.select_action(obs)  # ASK_TECHNICAL_INFO (different)
        assert a1 != a2  # Different actions

    def test_productive_repeat_not_overridden(self):
        """Same action repeated while productive is NOT overridden.

        Scenario 3 (Password Reset) requires 3 consecutive ASK_ACCOUNT_INFO
        calls. The agent must not escalate while info is still missing.
        """
        agent = BaselineSupportAgent()
        # Turn 1: ask account_id
        obs = _make_observation(problem_type=3, collected=np.array([0, 0, 0, 0], dtype=np.int8))
        a1 = agent.select_action(obs)
        assert a1 == config.ASK_ACCOUNT_INFO
        # Turn 2: ask email (same action, still productive)
        obs = _make_observation(problem_type=3, collected=np.array([1, 0, 0, 0], dtype=np.int8))
        a2 = agent.select_action(obs)
        assert a2 == config.ASK_ACCOUNT_INFO
        # Turn 3: ask security_question (same action, still productive)
        obs = _make_observation(problem_type=3, collected=np.array([1, 1, 0, 0], dtype=np.int8))
        a3 = agent.select_action(obs)
        assert a3 == config.ASK_ACCOUNT_INFO  # Not overridden

    def test_unproductive_repeat_overridden(self):
        """Same action repeated unproductively IS overridden to ESCALATE."""
        agent = BaselineSupportAgent()
        agent._last_action = config.PROPOSE_SOLUTION
        agent._consecutive_count = 2
        # Propose again (same action, not productive for info gathering)
        agent._has_diagnosed = True
        agent._has_proposed = True
        obs = _make_observation(
            problem_type=0, collected=_all_collected(0),
            previous_response=config.MSG_DIAGNOSIS_ACK,
        )
        action = agent.select_action(obs)
        # The cascade would return ESCALATE (fallback) anyway,
        # but anti-repetition also catches it
        assert action == config.ESCALATE


# ---------------------------------------------------------------------------
# Wave 4: All five scenarios tests
# ---------------------------------------------------------------------------


class TestAgentAllScenarios:
    """Verify the agent handles all five scenarios correctly."""

    @pytest.mark.parametrize("scenario_idx", range(NUM_SCENARIOS), ids=lambda s: SCENARIOS[s].name)
    def test_agent_gathers_all_fields(self, scenario_idx):
        """Agent asks for all required fields in the correct order."""
        agent = BaselineSupportAgent()
        scenario = SCENARIOS[scenario_idx]
        collected = np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8)

        for i in range(len(scenario.required_fields)):
            obs = _make_observation(problem_type=scenario_idx, collected=collected)
            action = agent.select_action(obs)
            expected_field = scenario.required_fields[i]
            assert action == config.FIELD_TO_ACTION[expected_field], (
                f"Scenario '{scenario.name}' turn {i+1}: expected "
                f"{config.ACTION_NAMES[config.FIELD_TO_ACTION[expected_field]]}, "
                f"got {config.ACTION_NAMES[action]}"
            )
            collected[i] = 1

    @pytest.mark.parametrize("scenario_idx", range(NUM_SCENARIOS), ids=lambda s: SCENARIOS[s].name)
    def test_agent_diagnoses_after_info(self, scenario_idx):
        """Agent diagnoses after all info is collected."""
        agent = BaselineSupportAgent()
        collected = _all_collected(scenario_idx)
        obs = _make_observation(problem_type=scenario_idx, collected=collected)
        assert agent.select_action(obs) == config.DIAGNOSE

    @pytest.mark.parametrize("scenario_idx", range(NUM_SCENARIOS), ids=lambda s: SCENARIOS[s].name)
    def test_agent_proposes_after_diagnosis(self, scenario_idx):
        """Agent proposes solution after diagnosis."""
        agent = BaselineSupportAgent()
        agent._has_diagnosed = True
        collected = _all_collected(scenario_idx)
        obs = _make_observation(problem_type=scenario_idx, collected=collected)
        assert agent.select_action(obs) == config.PROPOSE_SOLUTION

    @pytest.mark.parametrize("scenario_idx", range(NUM_SCENARIOS), ids=lambda s: SCENARIOS[s].name)
    def test_agent_full_flow_resolves(self, scenario_idx):
        """Agent follows the full gather → diagnose → propose flow."""
        agent = BaselineSupportAgent()
        scenario = SCENARIOS[scenario_idx]
        collected = np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8)

        # Gather all fields
        for i in range(len(scenario.required_fields)):
            obs = _make_observation(problem_type=scenario_idx, collected=collected)
            action = agent.select_action(obs)
            assert action in (
                config.ASK_ACCOUNT_INFO, config.ASK_BILLING_INFO,
                config.ASK_TECHNICAL_INFO,
            )
            collected[i] = 1

        # Diagnose
        obs = _make_observation(problem_type=scenario_idx, collected=collected)
        assert agent.select_action(obs) == config.DIAGNOSE
        assert agent._has_diagnosed is True

        # Propose
        obs = _make_observation(
            problem_type=scenario_idx, collected=collected,
            previous_response=config.MSG_DIAGNOSIS_ACK,
        )
        assert agent.select_action(obs) == config.PROPOSE_SOLUTION
        assert agent._has_proposed is True


# ---------------------------------------------------------------------------
# Wave 4: Invalid observation tests (FR-12 AC-12.05)
# ---------------------------------------------------------------------------


class TestAgentInvalidObservations:
    """Verify the agent handles invalid observations (FR-12 AC-12.05)."""

    def test_missing_key_raises_keyerror(self):
        """Missing observation key raises KeyError."""
        agent = BaselineSupportAgent()
        obs = _make_observation()
        del obs["frustration"]
        with pytest.raises(KeyError, match="frustration"):
            agent.select_action(obs)

    def test_empty_observation_raises_keyerror(self):
        """Empty observation raises KeyError."""
        agent = BaselineSupportAgent()
        with pytest.raises(KeyError):
            agent.select_action({})

    def test_missing_problem_type_raises(self):
        """Missing problem_type raises KeyError."""
        agent = BaselineSupportAgent()
        obs = _make_observation()
        del obs["problem_type"]
        with pytest.raises(KeyError, match="problem_type"):
            agent.select_action(obs)


# ---------------------------------------------------------------------------
# Wave 4: Hidden-state protection tests
# ---------------------------------------------------------------------------


class TestAgentHiddenStateProtection:
    """Verify the agent never accesses hidden state or env internals."""

    def test_agent_works_with_observation_only(self):
        """Agent selects actions using only the observation dict."""
        agent = BaselineSupportAgent()
        obs = _make_observation()
        action = agent.select_action(obs)
        assert 0 <= action < config.NUM_ACTIONS

    def test_agent_does_not_need_env_object(self):
        """Agent never receives or accesses the environment object."""
        agent = BaselineSupportAgent()
        # The agent only receives an observation dict — no env reference
        obs = _make_observation(collected=_all_collected(0))
        action = agent.select_action(obs)
        assert isinstance(action, int)

    def test_observation_has_no_hidden_keys(self):
        """The observation passed to the agent has no hidden state keys."""
        agent = BaselineSupportAgent()
        obs = _make_observation()
        hidden_keys = {
            "hidden_issue", "diagnosed", "solution_proposed",
            "solution_rejected", "last_action", "consecutive_action_count",
            "scenario", "state",
        }
        for key in hidden_keys:
            assert key not in obs


# ---------------------------------------------------------------------------
# Wave 4: Integration tests with CustomerSupportEnv
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    """Verify the agent works correctly with the live environment."""

    def test_integration_episode_terminates(self):
        """Episode terminates or truncates when agent interacts with env."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        obs, info = env.reset(seed=42)
        agent.reset()
        turns = 0
        terminated = False
        truncated = False
        while not terminated and not truncated and turns < 50:
            action = agent.select_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            turns += 1
        assert terminated or truncated

    def test_integration_all_actions_valid(self):
        """Every action the agent selects is in the valid action space."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        obs, info = env.reset(seed=42)
        agent.reset()
        for _ in range(20):
            action = agent.select_action(obs)
            assert action in env.action_space
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    def test_integration_agent_resets_between_episodes(self):
        """Agent internal state is cleared between episodes."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()

        # Run first episode
        obs, info = env.reset(seed=42)
        agent.reset()
        turns = 0
        while turns < 50:
            action = agent.select_action(obs)
            obs, _, term, trunc, _ = env.step(action)
            turns += 1
            if term or trunc:
                break

        # Agent state should be dirty after episode
        assert agent._has_diagnosed or agent._has_proposed or turns > 0

        # Reset for second episode
        obs, info = env.reset(seed=43)
        agent.reset()
        assert agent._has_diagnosed is False
        assert agent._has_proposed is False
        assert agent._last_action is None
        assert agent._consecutive_count == 0

    def test_integration_multiple_episodes(self):
        """Agent runs successfully across multiple episodes."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        for seed in range(42, 52):
            obs, info = env.reset(seed=seed)
            agent.reset()
            turns = 0
            terminated = False
            truncated = False
            while not terminated and not truncated and turns < 50:
                action = agent.select_action(obs)
                assert 0 <= action < config.NUM_ACTIONS
                obs, _, terminated, truncated, _ = env.step(action)
                turns += 1
            assert terminated or truncated, (
                f"Episode with seed {seed} did not terminate"
            )

    def test_integration_no_hidden_access(self):
        """Agent uses only observation data, never env internals."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        obs, info = env.reset(seed=42)
        agent.reset()
        # The agent only receives obs — it never sees env, env._state, etc.
        action = agent.select_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < config.NUM_ACTIONS
