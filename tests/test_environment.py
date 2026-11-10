"""Tests for the customer support environment and scenarios.

Wave 1: Verifies config constants related to the environment.
Wave 2: Verifies scenario definitions, conversation state, scenario registry,
        and the user state machine.
"""

import pytest
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from src import config
from src.support_env import CustomerSupportEnv
from src.support_scenarios import (
    Scenario,
    SCENARIOS,
    NUM_SCENARIOS,
    ConversationState,
    init_state,
    ScenarioRegistry,
    UserResponse,
    UserStateMachine,
)


# ---------------------------------------------------------------------------
# Wave 1: Config constant verification (FR-04, FR-03, FR-05)
# ---------------------------------------------------------------------------


class TestActionConstants:
    """Verify action constants defined in config (FR-04 AC-04.02)."""

    def test_num_actions(self):
        assert config.NUM_ACTIONS == 7

    def test_action_values_sequential(self):
        """Action integers should be 0 through 6."""
        expected = {0, 1, 2, 3, 4, 5, 6}
        actual = {
            config.ASK_ACCOUNT_INFO, config.ASK_BILLING_INFO,
            config.ASK_TECHNICAL_INFO, config.DIAGNOSE,
            config.PROPOSE_SOLUTION, config.ESCALATE,
            config.APOLOGIZE,
        }
        assert actual == expected

    def test_action_names_mapping(self):
        """Each action integer maps to a readable name (FR-04 AC-04.03)."""
        assert config.ACTION_NAMES[0] == "ASK_ACCOUNT_INFO"
        assert config.ACTION_NAMES[1] == "ASK_BILLING_INFO"
        assert config.ACTION_NAMES[2] == "ASK_TECHNICAL_INFO"
        assert config.ACTION_NAMES[3] == "DIAGNOSE"
        assert config.ACTION_NAMES[4] == "PROPOSE_SOLUTION"
        assert config.ACTION_NAMES[5] == "ESCALATE"
        assert config.ACTION_NAMES[6] == "APOLOGIZE"

    def test_action_names_count_matches(self):
        assert len(config.ACTION_NAMES) == config.NUM_ACTIONS


class TestMessageTypes:
    """Verify message type constants (Design Section 5.3)."""

    def test_num_message_types(self):
        assert config.NUM_MESSAGE_TYPES == 8

    def test_message_type_values_sequential(self):
        expected = {0, 1, 2, 3, 4, 5, 6, 7}
        actual = {
            config.MSG_INITIAL_PROBLEM, config.MSG_INFO_PROVIDED,
            config.MSG_NO_RELEVANT_INFO, config.MSG_SOLUTION_ACCEPTED,
            config.MSG_SOLUTION_REJECTED, config.MSG_FRUSTRATED,
            config.MSG_ESCALATION_CONFIRM, config.MSG_DIAGNOSIS_ACK,
        }
        assert actual == expected

    def test_message_type_names_mapping(self):
        assert config.MESSAGE_TYPE_NAMES[0] == "MSG_INITIAL_PROBLEM"
        assert config.MESSAGE_TYPE_NAMES[7] == "MSG_DIAGNOSIS_ACK"

    def test_message_type_names_count_matches(self):
        assert len(config.MESSAGE_TYPE_NAMES) == config.NUM_MESSAGE_TYPES


class TestObservationConfig:
    """Verify observation space parameters (FR-03, Design Section 5.4)."""

    def test_max_info_fields_is_positive(self):
        assert config.MAX_INFO_FIELDS > 0

    def test_default_max_turns(self):
        assert config.DEFAULT_MAX_TURNS == 15


# ---------------------------------------------------------------------------
# Wave 2: Scenario definition tests (FR-02)
# ---------------------------------------------------------------------------


class TestScenarioDefinitions:
    """Verify all scenario definitions are valid and complete (FR-02)."""

    def test_num_scenarios(self):
        """AC-02.01: At least 5 distinct customer problem types."""
        assert NUM_SCENARIOS >= 5
        assert len(SCENARIOS) == NUM_SCENARIOS

    def test_scenario_ids_unique(self):
        ids = [s.id for s in SCENARIOS]
        assert len(ids) == len(set(ids)), "Scenario IDs must be unique"

    def test_scenario_ids_sequential(self):
        """IDs should be 0 through NUM_SCENARIOS-1."""
        expected_ids = set(range(NUM_SCENARIOS))
        actual_ids = {s.id for s in SCENARIOS}
        assert actual_ids == expected_ids

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_customer_issue(self, scenario):
        """Each scenario has a non-empty customer issue description."""
        assert isinstance(scenario.customer_issue, str)
        assert len(scenario.customer_issue) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_hidden_issue(self, scenario):
        """Each scenario has a non-empty hidden issue state."""
        assert isinstance(scenario.hidden_issue, str)
        assert len(scenario.hidden_issue) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_initial_state(self, scenario):
        """Each scenario has a non-empty initial state dict."""
        assert isinstance(scenario.initial_state, dict)
        assert len(scenario.initial_state) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_required_fields(self, scenario):
        """Each scenario has non-empty required fields."""
        assert isinstance(scenario.required_fields, tuple)
        assert len(scenario.required_fields) > 0
        for field_name in scenario.required_fields:
            assert isinstance(field_name, str)
            assert len(field_name) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_available_actions(self, scenario):
        """Each scenario defines available troubleshooting actions."""
        assert isinstance(scenario.available_actions, tuple)
        assert len(scenario.available_actions) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_available_actions_are_valid(self, scenario):
        """All available action IDs must be valid action integers."""
        for action_id in scenario.available_actions:
            assert action_id in config.ACTION_NAMES, (
                f"Scenario '{scenario.name}' has invalid action {action_id}"
            )

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_resolution_conditions(self, scenario):
        """Each scenario has non-empty resolution conditions."""
        assert isinstance(scenario.resolution_conditions, tuple)
        assert len(scenario.resolution_conditions) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_has_expected_resolution(self, scenario):
        """Each scenario has a non-empty expected resolution description."""
        assert isinstance(scenario.expected_resolution, str)
        assert len(scenario.expected_resolution) > 0

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_resolution_action_is_propose_solution(self, scenario):
        """The resolution action is always PROPOSE_SOLUTION."""
        assert scenario.resolution_action == config.PROPOSE_SOLUTION

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_ask_field_mapping_covers_required_fields(self, scenario):
        """Union of all mapped fields must cover all required fields."""
        mapped_fields: set[str] = set()
        for fields in scenario.ask_field_mapping.values():
            mapped_fields.update(fields)
        required_set = set(scenario.required_fields)
        assert required_set == mapped_fields, (
            f"Scenario '{scenario.name}': required fields {required_set} "
            f"do not match mapped fields {mapped_fields}"
        )

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_ask_field_mapping_fields_in_required(self, scenario):
        """All mapped fields must be in required_fields."""
        required_set = set(scenario.required_fields)
        for action_id, fields in scenario.ask_field_mapping.items():
            for field_name in fields:
                assert field_name in required_set, (
                    f"Scenario '{scenario.name}': field '{field_name}' "
                    f"is not in required_fields"
                )

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_all_required_fields_in_field_to_action(self, scenario):
        """All required fields must have a mapping in config.FIELD_TO_ACTION."""
        for field_name in scenario.required_fields:
            assert field_name in config.FIELD_TO_ACTION, (
                f"Scenario '{scenario.name}': field '{field_name}' "
                f"is not in config.FIELD_TO_ACTION"
            )

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_ask_field_mapping_actions_are_valid(self, scenario):
        """All keys in ask_field_mapping must be valid ASK_* action IDs."""
        valid_ask_actions = {
            config.ASK_ACCOUNT_INFO,
            config.ASK_BILLING_INFO,
            config.ASK_TECHNICAL_INFO,
        }
        for action_id in scenario.ask_field_mapping:
            assert action_id in valid_ask_actions, (
                f"Scenario '{scenario.name}': mapping key {action_id} "
                f"is not a valid ASK_* action"
            )

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_scenario_names_are_unique(self, scenario):
        """All scenario names must be unique."""
        names = [s.name for s in SCENARIOS]
        assert names.count(scenario.name) == 1


# ---------------------------------------------------------------------------
# Wave 2: Conversation state tests
# ---------------------------------------------------------------------------


class TestConversationState:
    """Verify ConversationState initialization and hidden state access."""

    def test_init_state_sets_initial_values(self):
        """init_state creates a state with all initial values."""
        scenario = SCENARIOS[0]
        state = init_state(scenario)
        assert state.scenario is scenario
        assert state.turn_count == 0
        assert state.frustration_level == 0.0
        assert state.last_action is None
        assert state.consecutive_action_count == 0
        assert state.diagnosed is False
        assert state.solution_proposed is False
        assert state.solution_rejected is False
        assert state.user_message_type == config.MSG_INITIAL_PROBLEM

    def test_init_state_info_collected_all_false(self):
        """All info_collected entries start as False."""
        scenario = SCENARIOS[0]
        state = init_state(scenario)
        assert len(state.info_collected) == len(scenario.required_fields)
        for field_name in scenario.required_fields:
            assert state.info_collected[field_name] is False

    def test_hidden_issue_accessible(self):
        """The hidden issue is accessible on the state but not in observation."""
        scenario = SCENARIOS[0]
        state = init_state(scenario)
        assert state.hidden_issue == scenario.hidden_issue
        assert len(state.hidden_issue) > 0

    def test_problem_type_matches_scenario_id(self):
        """The problem_type property matches the scenario ID."""
        scenario = SCENARIOS[2]
        state = init_state(scenario)
        assert state.problem_type == scenario.id

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
    def test_info_collected_keys_match_required_fields(self, scenario):
        """info_collected dict keys match required_fields exactly."""
        state = init_state(scenario)
        assert set(state.info_collected.keys()) == set(scenario.required_fields)


# ---------------------------------------------------------------------------
# Wave 2: Scenario registry tests (FR-02 AC-02.06, NFR-02)
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    """Verify the scenario registry samples scenarios correctly."""

    def test_sample_returns_scenario(self):
        """sample() returns a Scenario instance."""
        rng = np.random.default_rng(42)
        registry = ScenarioRegistry(rng)
        result = registry.sample()
        assert isinstance(result, Scenario)

    def test_sample_reproducible_with_same_seed(self):
        """Same seed produces same sequence of samples (NFR-02.01)."""
        registry1 = ScenarioRegistry(np.random.default_rng(42))
        registry2 = ScenarioRegistry(np.random.default_rng(42))
        samples1 = [registry1.sample().id for _ in range(20)]
        samples2 = [registry2.sample().id for _ in range(20)]
        assert samples1 == samples2

    def test_sample_different_with_different_seed(self):
        """Different seeds produce different sequences (typically)."""
        registry1 = ScenarioRegistry(np.random.default_rng(42))
        registry2 = ScenarioRegistry(np.random.default_rng(99))
        samples1 = [registry1.sample().id for _ in range(20)]
        samples2 = [registry2.sample().id for _ in range(20)]
        assert samples1 != samples2

    def test_sample_covers_all_scenarios(self):
        """Sampling many times covers all scenario IDs."""
        rng = np.random.default_rng(42)
        registry = ScenarioRegistry(rng)
        sampled_ids = {registry.sample().id for _ in range(500)}
        assert sampled_ids == set(range(NUM_SCENARIOS))

    def test_sample_returns_valid_id(self):
        """Sampled scenario ID is in valid range."""
        rng = np.random.default_rng(42)
        registry = ScenarioRegistry(rng)
        for _ in range(100):
            scenario = registry.sample()
            assert 0 <= scenario.id < NUM_SCENARIOS


# ---------------------------------------------------------------------------
# Wave 2: User state machine tests (FR-02, FR-05)
# ---------------------------------------------------------------------------


class TestUserStateMachineAskHandlers:
    """Test the ASK_* action handlers in UserStateMachine."""

    def test_ask_gathers_first_uncollected_field(self):
        """ASK action marks the first uncollected mapped field as collected."""
        state = init_state(SCENARIOS[0])
        response = UserStateMachine.process_action(state, config.ASK_ACCOUNT_INFO)
        assert response.info_gathered is True
        assert response.info_field == "account_id"
        assert state.info_collected["account_id"] is True
        assert response.message_type == config.MSG_INFO_PROVIDED

    def test_ask_gathers_second_field_on_repeat(self):
        """Second ASK with same action gathers the next uncollected field."""
        state = init_state(SCENARIOS[0])
        UserStateMachine.process_action(state, config.ASK_TECHNICAL_INFO)
        response = UserStateMachine.process_action(state, config.ASK_TECHNICAL_INFO)
        assert response.info_gathered is True
        assert response.info_field == "connection_type"
        assert state.info_collected["connection_type"] is True

    def test_ask_all_fields_collected_returns_no_relevant_info(self):
        """ASK action with all mapped fields collected returns no-relevant-info."""
        state = init_state(SCENARIOS[0])
        UserStateMachine.process_action(state, config.ASK_ACCOUNT_INFO)
        response = UserStateMachine.process_action(state, config.ASK_ACCOUNT_INFO)
        assert response.info_gathered is False
        assert response.message_type == config.MSG_NO_RELEVANT_INFO
        assert response.frustration_delta == pytest.approx(
            config.FRUSTRATION_INCREASE_REPEATED
        )

    def test_ask_no_mapping_returns_no_relevant_info(self):
        """ASK action with no mapping for this scenario is irrelevant."""
        state = init_state(SCENARIOS[3])
        response = UserStateMachine.process_action(state, config.ASK_TECHNICAL_INFO)
        assert response.info_gathered is False
        assert response.message_type == config.MSG_NO_RELEVANT_INFO
        assert response.frustration_delta == pytest.approx(
            config.FRUSTRATION_INCREASE_IRRELEVANT
        )

    def test_ask_no_mapping_for_billing_on_tech_scenario(self):
        """ASK_BILLING_INFO on a tech-only scenario is irrelevant."""
        state = init_state(SCENARIOS[0])
        response = UserStateMachine.process_action(state, config.ASK_BILLING_INFO)
        assert response.info_gathered is False
        assert response.message_type == config.MSG_NO_RELEVANT_INFO
        assert response.frustration_delta == pytest.approx(
            config.FRUSTRATION_INCREASE_IRRELEVANT
        )


class TestUserStateMachineDiagnose:
    """Test the DIAGNOSE action handler."""

    def test_diagnose_with_all_info_collected(self):
        """Diagnose after all info collected returns diagnosis ack."""
        state = init_state(SCENARIOS[0])
        for field_name in state.scenario.required_fields:
            state.info_collected[field_name] = True
        response = UserStateMachine.process_action(state, config.DIAGNOSE)
        assert response.message_type == config.MSG_DIAGNOSIS_ACK
        assert state.diagnosed is True
        assert response.frustration_delta == pytest.approx(
            -config.FRUSTRATION_DECREASE_DIAGNOSIS
        )

    def test_diagnose_with_incomplete_info(self):
        """Diagnose before all info collected returns frustrated message."""
        state = init_state(SCENARIOS[0])
        response = UserStateMachine.process_action(state, config.DIAGNOSE)
        assert response.message_type == config.MSG_FRUSTRATED
        assert state.diagnosed is True
        assert response.frustration_delta == pytest.approx(
            config.FRUSTRATION_INCREASE_PREMATURE_DIAGNOSIS
        )


class TestUserStateMachineProposeSolution:
    """Test the PROPOSE_SOLUTION action handler."""

    def test_propose_solution_accepted(self):
        """Solution accepted when all info collected and diagnosed."""
        state = init_state(SCENARIOS[0])
        for field_name in state.scenario.required_fields:
            state.info_collected[field_name] = True
        UserStateMachine.process_action(state, config.DIAGNOSE)
        response = UserStateMachine.process_action(state, config.PROPOSE_SOLUTION)
        assert response.solution_accepted is True
        assert response.terminated is True
        assert response.message_type == config.MSG_SOLUTION_ACCEPTED
        assert state.solution_proposed is True
        assert state.solution_rejected is False

    def test_propose_solution_rejected_no_diagnosis(self):
        """Solution rejected when info collected but not diagnosed."""
        state = init_state(SCENARIOS[0])
        for field_name in state.scenario.required_fields:
            state.info_collected[field_name] = True
        response = UserStateMachine.process_action(state, config.PROPOSE_SOLUTION)
        assert response.solution_accepted is False
        assert response.solution_rejected is True
        assert response.terminated is False
        assert response.message_type == config.MSG_SOLUTION_REJECTED
        assert response.frustration_delta == pytest.approx(
            config.FRUSTRATION_INCREASE_WRONG_SOLUTION
        )
        assert state.solution_rejected is True

    def test_propose_solution_rejected_incomplete_info(self):
        """Solution rejected when info is incomplete."""
        state = init_state(SCENARIOS[0])
        UserStateMachine.process_action(state, config.DIAGNOSE)
        response = UserStateMachine.process_action(state, config.PROPOSE_SOLUTION)
        assert response.solution_accepted is False
        assert response.solution_rejected is True
        assert response.message_type == config.MSG_SOLUTION_REJECTED


class TestUserStateMachineEscalate:
    """Test the ESCALATE action handler."""

    def test_escalate_terminates(self):
        """Escalate always terminates the episode."""
        state = init_state(SCENARIOS[0])
        response = UserStateMachine.process_action(state, config.ESCALATE)
        assert response.terminated is True
        assert response.message_type == config.MSG_ESCALATION_CONFIRM

    def test_escalate_terminates_immediately(self):
        """Escalate works on first turn (AC-09.05 edge case)."""
        state = init_state(SCENARIOS[0])
        response = UserStateMachine.process_action(state, config.ESCALATE)
        assert response.terminated is True


class TestUserStateMachineApologize:
    """Test the APOLOGIZE action handler."""

    def test_apologize_with_frustration(self):
        """Apologize when frustration > 0 decreases frustration."""
        state = init_state(SCENARIOS[0])
        state.frustration_level = 0.5
        response = UserStateMachine.process_action(state, config.APOLOGIZE)
        assert response.message_type == config.MSG_INFO_PROVIDED
        assert response.frustration_delta == pytest.approx(
            -config.FRUSTRATION_DECREASE_APOLOGIZE
        )

    def test_apologize_without_frustration(self):
        """Apologize when frustration is 0 has no effect."""
        state = init_state(SCENARIOS[0])
        state.frustration_level = 0.0
        response = UserStateMachine.process_action(state, config.APOLOGIZE)
        assert response.message_type == config.MSG_INFO_PROVIDED
        assert response.frustration_delta == pytest.approx(0.0)


class TestUserStateMachineFullFlow:
    """Test a complete conversation flow through the state machine."""

    def test_full_resolution_flow(self):
        """A complete gather-diagnose-propose flow resolves successfully."""
        state = init_state(SCENARIOS[0])

        # Gather all required fields
        for field_name in state.scenario.required_fields:
            action = config.FIELD_TO_ACTION[field_name]
            response = UserStateMachine.process_action(state, action)
            assert response.info_gathered is True

        # Diagnose
        response = UserStateMachine.process_action(state, config.DIAGNOSE)
        assert response.message_type == config.MSG_DIAGNOSIS_ACK
        assert state.diagnosed is True

        # Propose solution
        response = UserStateMachine.process_action(state, config.PROPOSE_SOLUTION)
        assert response.solution_accepted is True
        assert response.terminated is True

    def test_full_resolution_flow_all_scenarios(self):
        """Every scenario can be resolved through the correct flow."""
        for scenario in SCENARIOS:
            state = init_state(scenario)

            # Gather all required fields
            for field_name in scenario.required_fields:
                action = config.FIELD_TO_ACTION[field_name]
                response = UserStateMachine.process_action(state, action)
                assert response.info_gathered is True, (
                    f"Scenario '{scenario.name}': failed to gather '{field_name}'"
                )

            # Diagnose
            response = UserStateMachine.process_action(state, config.DIAGNOSE)
            assert state.diagnosed is True

            # Propose solution
            response = UserStateMachine.process_action(state, config.PROPOSE_SOLUTION)
            assert response.solution_accepted is True, (
                f"Scenario '{scenario.name}': solution was not accepted"
            )
            assert response.terminated is True


# ---------------------------------------------------------------------------
# Wave 3: Environment initialization tests (FR-01, FR-07, FR-12)
# ---------------------------------------------------------------------------


class TestEnvironmentInit:
    """Verify CustomerSupportEnv initialization (FR-01, FR-07, FR-12)."""

    def test_env_is_gymnasium_env(self):
        """AC-01.01: Environment is a subclass of gymnasium.Env."""
        env = CustomerSupportEnv()
        assert isinstance(env, gymnasium.Env)

    def test_action_space_is_discrete(self):
        """AC-04.01: Action space is Discrete(NUM_ACTIONS)."""
        env = CustomerSupportEnv()
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == config.NUM_ACTIONS

    def test_observation_space_is_dict(self):
        """AC-03.01: Observation space is a Dict."""
        env = CustomerSupportEnv()
        assert isinstance(env.observation_space, spaces.Dict)

    def test_default_max_turns(self):
        """AC-07.01: Default max_turns is 15."""
        env = CustomerSupportEnv()
        assert env.max_turns == config.DEFAULT_MAX_TURNS

    def test_custom_max_turns(self):
        """AC-07.01: Custom max_turns is accepted."""
        env = CustomerSupportEnv(max_turns=5)
        assert env.max_turns == 5

    def test_invalid_max_turns_zero(self):
        """AC-07.03: max_turns=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_turns"):
            CustomerSupportEnv(max_turns=0)

    def test_invalid_max_turns_negative(self):
        """AC-07.03: Negative max_turns raises ValueError."""
        with pytest.raises(ValueError, match="max_turns"):
            CustomerSupportEnv(max_turns=-1)

    def test_invalid_max_turns_float(self):
        """AC-12.04: Non-integer max_turns raises ValueError."""
        with pytest.raises(ValueError, match="max_turns"):
            CustomerSupportEnv(max_turns=5.5)


# ---------------------------------------------------------------------------
# Wave 3: Observation and action space tests (FR-03, FR-04)
# ---------------------------------------------------------------------------


class TestEnvironmentSpaces:
    """Verify observation and action space structure (FR-03, FR-04)."""

    def test_observation_space_has_all_fields(self):
        """AC-03.02-03.06: All 8 agent-visible fields present."""
        env = CustomerSupportEnv()
        expected_keys = {
            "customer_issue", "collected_information", "missing_information",
            "turn_count", "frustration", "previous_customer_response",
            "available_actions", "problem_type",
        }
        assert set(env.observation_space.spaces.keys()) == expected_keys

    def test_collected_information_is_multibinary(self):
        """collected_information is MultiBinary(MAX_INFO_FIELDS)."""
        env = CustomerSupportEnv()
        space = env.observation_space["collected_information"]
        assert isinstance(space, spaces.MultiBinary)
        assert space.n == config.MAX_INFO_FIELDS

    def test_missing_information_is_multibinary(self):
        """missing_information is MultiBinary(MAX_INFO_FIELDS)."""
        env = CustomerSupportEnv()
        space = env.observation_space["missing_information"]
        assert isinstance(space, spaces.MultiBinary)
        assert space.n == config.MAX_INFO_FIELDS

    def test_available_actions_is_multibinary(self):
        """available_actions is MultiBinary(NUM_ACTIONS)."""
        env = CustomerSupportEnv()
        space = env.observation_space["available_actions"]
        assert isinstance(space, spaces.MultiBinary)
        assert space.n == config.NUM_ACTIONS

    def test_frustration_is_box(self):
        """frustration is Box(0.0, 1.0)."""
        env = CustomerSupportEnv()
        space = env.observation_space["frustration"]
        assert isinstance(space, spaces.Box)


# ---------------------------------------------------------------------------
# Wave 3: Reset tests (FR-01, NFR-02)
# ---------------------------------------------------------------------------


class TestEnvironmentReset:
    """Verify reset() behavior (FR-01, NFR-02)."""

    def test_reset_returns_2tuple(self):
        """AC-01.02: reset() returns a 2-tuple (observation, info)."""
        env = CustomerSupportEnv()
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_in_space(self):
        """AC-03.07: Observation from reset is in observation_space."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_reset_info_has_scenario_id(self):
        """Info contains scenario_id."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert "scenario_id" in info
        assert isinstance(info["scenario_id"], int)
        assert 0 <= info["scenario_id"] < NUM_SCENARIOS

    def test_reset_info_has_turn_count(self):
        """Info contains turn_count == 0."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert info["turn_count"] == 0

    def test_reset_reproducible(self):
        """NFR-02: Same seed produces same scenario."""
        env1 = CustomerSupportEnv()
        env2 = CustomerSupportEnv()
        _, info1 = env1.reset(seed=42)
        _, info2 = env2.reset(seed=42)
        assert info1["scenario_id"] == info2["scenario_id"]

    def test_reset_mid_episode(self):
        """AC-12.03: reset() during active episode is allowed."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        env.step(config.ASK_ACCOUNT_INFO)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert info["turn_count"] == 0

    def test_reset_initial_frustration_zero(self):
        """Frustration starts at 0.0 after reset."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert obs["frustration"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Wave 3: Step tests (FR-01)
# ---------------------------------------------------------------------------


class TestEnvironmentStep:
    """Verify step() return structure (FR-01)."""

    def test_step_returns_5tuple(self):
        """AC-01.03: step() returns a 5-tuple."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        result = env.step(config.ASK_ACCOUNT_INFO)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_observation_in_space(self):
        """AC-03.07: Observation from step is in observation_space."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        obs, reward, term, trunc, info = env.step(config.ASK_ACCOUNT_INFO)
        assert env.observation_space.contains(obs)

    def test_step_increments_turn(self):
        """Turn count increments by 1 after each step."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(config.ASK_ACCOUNT_INFO)
        assert info["turn_count"] == 1

    def test_step_reward_is_float(self):
        """Reward is a finite float."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert isinstance(reward, float)
        assert reward == reward  # NaN check

    def test_step_terminated_is_bool(self):
        """terminated is a bool."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, term, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert isinstance(term, bool)

    def test_step_truncated_is_bool(self):
        """truncated is a bool."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, trunc, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert isinstance(trunc, bool)


# ---------------------------------------------------------------------------
# Wave 3: Every configured action test (FR-04)
# ---------------------------------------------------------------------------


class TestEnvironmentAllActions:
    """Verify every configured action produces valid output (FR-04)."""

    @pytest.mark.parametrize(
        "action", range(config.NUM_ACTIONS),
        ids=lambda a: config.ACTION_NAMES[a],
    )
    def test_action_produces_valid_output(self, action):
        """Each action returns a valid 5-tuple with observation in space."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        obs, reward, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Wave 3: Reward tests (FR-05)
# ---------------------------------------------------------------------------


class TestEnvironmentRewards:
    """Verify reward calculation for each reward type (FR-05)."""

    def test_reward_turn_penalty(self):
        """AC-05.03: Each step has -0.5 base penalty."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(config.APOLOGIZE)
        assert reward == pytest.approx(config.REWARD_TURN_PENALTY)

    def test_reward_info_gathered(self):
        """AC-05.02: Gathering info yields +1.0."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        scenario = SCENARIOS[info["scenario_id"]]
        first_field = scenario.required_fields[0]
        action = config.FIELD_TO_ACTION[first_field]
        _, reward, _, _, _ = env.step(action)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_INFO_GATHERED
        )

    def test_reward_resolution(self):
        """AC-05.01: Correct resolution yields +10.0."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        scenario = SCENARIOS[info["scenario_id"]]
        for field_name in scenario.required_fields:
            action = config.FIELD_TO_ACTION[field_name]
            env.step(action)
        env.step(config.DIAGNOSE)
        _, reward, term, trunc, _ = env.step(config.PROPOSE_SOLUTION)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_RESOLVE
        )
        assert term is True

    def test_reward_wrong_solution(self):
        """AC-05.04: Wrong solution yields -3.0."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, term, _, _ = env.step(config.PROPOSE_SOLUTION)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_WRONG_SOLUTION
        )
        assert term is False

    def test_reward_repeated_action(self):
        """AC-05.05: Unproductive repeated action yields -2.0 penalty.

        PROPOSE_SOLUTION repeated without progress (no info gathered, solution
        still rejected) triggers the penalty. This is the canonical unproductive
        repeat — the action cannot gather new information.
        """
        env = CustomerSupportEnv()
        env.reset(seed=42)
        env.step(config.PROPOSE_SOLUTION)
        _, reward, _, _, _ = env.step(config.PROPOSE_SOLUTION)
        # Second propose (unproductive): -0.5 (turn) + -3.0 (wrong) + -2.0 (repeated)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY
            + config.REWARD_WRONG_SOLUTION
            + config.REWARD_REPEATED_ACTION
        )

    def test_productive_repeated_ask_no_penalty(self):
        """Progress-aware: productive repeated ASK is NOT penalised.

        Scenario 2 (Account Login) gathers account_id then email, both via
        ASK_ACCOUNT_INFO. The second call gathers a new field so no penalty.
        """
        # Find seed that gives scenario 2 (Account Login)
        env = CustomerSupportEnv()
        for seed in range(42, 200):
            _, info = env.reset(seed=seed)
            if info["scenario_id"] == 2:
                break
        # Turn 1: ASK_ACCOUNT_INFO → gathers account_id
        _, r1, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert r1 == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_INFO_GATHERED
        )
        # Turn 2: ASK_ACCOUNT_INFO again → gathers email (different field, no penalty)
        _, r2, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert r2 == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_INFO_GATHERED
        ), "Productive repeated ASK should NOT incur the repeated-action penalty"

    def test_unproductive_repeated_ask_penalty(self):
        """Progress-aware: ASK repeated after all its fields collected IS penalised."""
        # Scenario 0 — ASK_ACCOUNT_INFO gathers only account_id
        env = CustomerSupportEnv()
        for seed in range(42, 200):
            _, info = env.reset(seed=seed)
            if info["scenario_id"] == 0:
                break
        # Turn 1: gather account_id
        env.step(config.ASK_ACCOUNT_INFO)
        # Turn 2: repeat — account_id already collected, returns MSG_NO_RELEVANT_INFO
        _, reward, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        # No info gathered → penalty fires
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_REPEATED_ACTION
        )

    def test_all_five_scenarios_no_unintended_penalty(self):
        """All five scenarios resolve with +10.50 optimal reward."""
        expected = {
            0: pytest.approx(10.5),  # 3 fields, 2 different actions
            1: pytest.approx(10.5),  # 3 fields, 2 different actions
            2: pytest.approx(10.5),  # 3 fields, 2 different actions
            3: pytest.approx(10.5),  # 3 fields, all via ASK_ACCOUNT_INFO (productive)
            4: pytest.approx(10.5),  # 3 fields, 2 different actions
        }
        for scenario_idx in range(5):
            env = CustomerSupportEnv()
            from src.baseline_agent import BaselineSupportAgent
            agent = BaselineSupportAgent()
            for seed in range(42, 500):
                obs, info = env.reset(seed=seed)
                if info["scenario_id"] == scenario_idx:
                    break
            agent.reset()
            total = 0.0
            for _ in range(20):
                action = agent.select_action(obs)
                obs, r, term, trunc, _ = env.step(action)
                total += r
                if term or trunc:
                    break
            assert total == expected[scenario_idx], (
                f"Scenario {scenario_idx} ({SCENARIOS[scenario_idx].name}): "
                f"expected +10.5, got {total:.4f}"
            )

    def test_reward_escalate(self):
        """AC-05.06: Escalation yields +2.0."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, term, _, _ = env.step(config.ESCALATE)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_ESCALATE
        )

    def test_reward_truncate(self):
        """AC-05.07: Turn limit yields -5.0 terminal penalty."""
        env = CustomerSupportEnv(max_turns=2)
        env.reset(seed=42)
        env.step(config.ASK_ACCOUNT_INFO)
        _, reward, term, trunc, info = env.step(config.DIAGNOSE)
        assert trunc is True
        assert term is False
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY + config.REWARD_TRUNCATE
        )

    def test_reward_apologize_high_frustration(self):
        """AC-05.08: Apologize at high frustration yields +0.5."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        env._state.frustration_level = 0.6
        _, reward, _, _, _ = env.step(config.APOLOGIZE)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY
            + config.REWARD_APOLOGIZE_HIGH_FRUSTRATION
        )

    def test_reward_apologize_low_frustration(self):
        """AC-05.08: Apologize at low frustration yields 0.0."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(config.APOLOGIZE)
        assert reward == pytest.approx(
            config.REWARD_TURN_PENALTY
            + config.REWARD_APOLOGIZE_LOW_FRUSTRATION
        )


# ---------------------------------------------------------------------------
# Wave 3: Termination and truncation tests (FR-06, FR-07)
# ---------------------------------------------------------------------------


class TestEnvironmentTermination:
    """Verify episode termination rules (FR-06, FR-07)."""

    def test_resolved_termination(self):
        """AC-06.01: Correct solution sets terminated=True."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        scenario = SCENARIOS[info["scenario_id"]]
        for field_name in scenario.required_fields:
            env.step(config.FIELD_TO_ACTION[field_name])
        env.step(config.DIAGNOSE)
        _, _, term, trunc, info = env.step(config.PROPOSE_SOLUTION)
        assert term is True
        assert trunc is False
        assert info.get("termination_reason") == "resolved"
        assert info["success"] is True

    def test_escalated_termination(self):
        """AC-06.02: Escalation sets terminated=True."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, term, trunc, info = env.step(config.ESCALATE)
        assert term is True
        assert trunc is False
        assert info.get("termination_reason") == "escalated"
        assert info["success"] is False

    def test_truncated_termination(self):
        """AC-06.03/AC-07.02: Turn limit sets truncated=True."""
        env = CustomerSupportEnv(max_turns=2)
        env.reset(seed=42)
        env.step(config.ASK_ACCOUNT_INFO)
        _, _, term, trunc, info = env.step(config.DIAGNOSE)
        assert trunc is True
        assert term is False
        assert info.get("termination_reason") == "truncated"
        assert info["success"] is False

    def test_termination_reason_in_info(self):
        """AC-06.05: termination_reason present on terminal steps."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, term, _, info = env.step(config.ESCALATE)
        assert "termination_reason" in info

    def test_never_both_terminated_and_truncated(self):
        """terminated and truncated are never both True."""
        env = CustomerSupportEnv(max_turns=5)
        env.reset(seed=42)
        for _ in range(5):
            _, _, term, trunc, _ = env.step(config.ASK_ACCOUNT_INFO)
            assert not (term and trunc), "Both terminated and truncated are True"

    def test_episode_reward_in_terminal_info(self):
        """episode_reward is in info on terminal steps."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(config.ESCALATE)
        assert "episode_reward" in info


# ---------------------------------------------------------------------------
# Wave 3: Error handling tests (FR-12)
# ---------------------------------------------------------------------------


class TestEnvironmentErrors:
    """Verify error handling and validation (FR-12)."""

    def test_invalid_action_out_of_range(self):
        """AC-12.01: Out-of-range action raises ValueError."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)

    def test_invalid_action_negative(self):
        """AC-12.01: Negative action raises ValueError."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(-1)

    def test_step_after_termination_raises(self):
        """AC-12.02: step() after termination raises RuntimeError."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        env.step(config.ESCALATE)
        with pytest.raises(RuntimeError, match="Episode has ended"):
            env.step(config.ASK_ACCOUNT_INFO)

    def test_no_state_mutation_on_error(self):
        """AC-12.08: Invalid action does not mutate state."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        turn_before = env._state.turn_count
        with pytest.raises(ValueError):
            env.step(99)
        assert env._state.turn_count == turn_before

    def test_non_integer_action_raises(self):
        """AC-12.01: Non-integer action raises ValueError."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        with pytest.raises(ValueError, match="must be an integer"):
            env.step("not_an_int")


# ---------------------------------------------------------------------------
# Wave 3: Hidden-state protection tests
# ---------------------------------------------------------------------------


class TestHiddenStateProtection:
    """Verify hidden state is never exposed in observations or info."""

    HIDDEN_KEYS: set[str] = {
        "hidden_issue", "diagnosed", "solution_proposed",
        "solution_rejected", "last_action", "consecutive_action_count",
        "scenario", "state", "conversation_state",
    }

    VISIBLE_OBS_KEYS: set[str] = {
        "customer_issue", "collected_information", "missing_information",
        "turn_count", "frustration", "previous_customer_response",
        "available_actions", "problem_type",
    }

    def test_reset_observation_only_has_visible_keys(self):
        """Observation from reset only has the 8 visible keys."""
        env = CustomerSupportEnv()
        obs, _ = env.reset(seed=42)
        assert set(obs.keys()) == self.VISIBLE_OBS_KEYS

    def test_step_observation_only_has_visible_keys(self):
        """Observation from step only has the 8 visible keys."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        assert set(obs.keys()) == self.VISIBLE_OBS_KEYS

    def test_reset_observation_no_hidden_keys(self):
        """No hidden keys in reset observation."""
        env = CustomerSupportEnv()
        obs, _ = env.reset(seed=42)
        for key in self.HIDDEN_KEYS:
            assert key not in obs, f"Hidden key '{key}' found in observation"

    def test_step_observation_no_hidden_keys(self):
        """No hidden keys in step observation."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(config.ASK_ACCOUNT_INFO)
        for key in self.HIDDEN_KEYS:
            assert key not in obs, f"Hidden key '{key}' found in observation"

    def test_terminal_observation_no_hidden_keys(self):
        """No hidden keys in terminal step observation."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(config.ESCALATE)
        for key in self.HIDDEN_KEYS:
            assert key not in obs, f"Hidden key '{key}' in terminal obs"

    def test_info_no_hidden_issue(self):
        """hidden_issue never appears in info."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert "hidden_issue" not in info
        _, _, _, _, info = env.step(config.ASK_ACCOUNT_INFO)
        assert "hidden_issue" not in info

    def test_info_no_diagnosed(self):
        """diagnosed never appears in info."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(config.DIAGNOSE)
        assert "diagnosed" not in info

    def test_info_no_solution_flags(self):
        """solution_proposed and solution_rejected never in info."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(config.PROPOSE_SOLUTION)
        assert "solution_proposed" not in info
        assert "solution_rejected" not in info

    def test_info_no_scenario_object(self):
        """Scenario object never passed in info."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert "scenario" not in info
        _, _, _, _, info2 = env.step(config.ASK_ACCOUNT_INFO)
        assert "scenario" not in info2

    def test_terminal_info_no_hidden_keys(self):
        """No hidden keys in terminal step info."""
        env = CustomerSupportEnv()
        env.reset(seed=42)
        _, _, _, _, info = env.step(config.ESCALATE)
        for key in self.HIDDEN_KEYS:
            assert key not in info, f"Hidden key '{key}' in terminal info"


# ---------------------------------------------------------------------------
# Wave 3: Observation validity throughout an episode
# ---------------------------------------------------------------------------


class TestObservationValidity:
    """Verify observations are always valid throughout an episode."""

    def test_observation_valid_full_episode(self):
        """Observation is in space at every step of a full episode."""
        env = CustomerSupportEnv()
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        scenario = SCENARIOS[info["scenario_id"]]
        for field_name in scenario.required_fields:
            obs, _, _, _, _ = env.step(config.FIELD_TO_ACTION[field_name])
            assert env.observation_space.contains(obs)
        obs, _, _, _, _ = env.step(config.DIAGNOSE)
        assert env.observation_space.contains(obs)
        obs, _, term, trunc, _ = env.step(config.PROPOSE_SOLUTION)
        assert env.observation_space.contains(obs)
        assert term is True

    def test_observation_turn_count_never_exceeds_max(self):
        """AC-07.04: turn_count in observation never exceeds max_turns."""
        env = CustomerSupportEnv(max_turns=3)
        env.reset(seed=42)
        for _ in range(3):
            obs, _, term, trunc, _ = env.step(config.ASK_ACCOUNT_INFO)
            assert int(obs["turn_count"]) <= 3
            if term or trunc:
                break


# ---------------------------------------------------------------------------
# Wave 3: Reproducibility tests (NFR-02)
# ---------------------------------------------------------------------------


class TestEnvironmentReproducibility:
    """Verify reproducibility with seeds (NFR-02)."""

    def test_same_seed_same_episode(self):
        """Same seed produces identical rewards and outcomes."""
        def run_episode(seed: int) -> tuple:
            env = CustomerSupportEnv()
            obs, info = env.reset(seed=seed)
            scenario = SCENARIOS[info["scenario_id"]]
            total_reward = 0.0
            turns = 0
            for field_name in scenario.required_fields:
                _, r, _, _, _ = env.step(config.FIELD_TO_ACTION[field_name])
                total_reward += r
                turns += 1
            _, r, _, _, _ = env.step(config.DIAGNOSE)
            total_reward += r
            turns += 1
            _, r, term, trunc, _ = env.step(config.PROPOSE_SOLUTION)
            total_reward += r
            turns += 1
            return total_reward, turns, term, trunc

        result1 = run_episode(42)
        result2 = run_episode(42)
        assert result1 == result2

    def test_truncated_before_turn_limit(self):
        """AC-07.05: truncated is False for all steps before turn limit."""
        env = CustomerSupportEnv(max_turns=5)
        env.reset(seed=42)
        for i in range(4):
            _, _, _, trunc, _ = env.step(config.ASK_ACCOUNT_INFO)
            assert trunc is False, f"truncated=True on turn {i + 1}"


# ---------------------------------------------------------------------------
# Wave 3: Gymnasium env_checker validation
# ---------------------------------------------------------------------------


class TestEnvChecker:
    """Verify the environment passes Gymnasium's env_checker."""

    def test_check_env_passes(self):
        """gymnasium.utils.env_checker.check_env passes without errors."""
        env = CustomerSupportEnv()
        check_env(env, skip_render_check=True)
