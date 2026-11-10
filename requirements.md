# Requirements Specification — multi-turn-agent-gym

## 1. Overview

This document defines the functional and non-functional requirements for **multi-turn-agent-gym**, a custom Gymnasium environment that simulates a customer-support agent resolving user problems over multiple conversation turns. The project includes a rule-based baseline agent, an evaluation harness, and an automated test suite.

### 1.1 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| RL Environment Framework | Gymnasium |
| Numerical Operations | NumPy |
| Testing Framework | Pytest |

### 1.2 Definitions

| Term | Definition |
|---|---|
| **Episode** | A single complete conversation from initial user message to resolution, escalation, or truncation. |
| **Turn** | One agent action followed by one simulated user response within an episode. |
| **Observation** | The structured data the agent receives at each turn, encoding conversation state. |
| **Action** | An integer selecting one of a fixed set of agent behaviors. |
| **Resolution** | The user's problem is solved correctly; episode terminates with success. |
| **Escalation** | The agent hands off to a human; episode terminates as acceptable but non-ideal. |
| **Truncation** | The episode ends because the maximum turn limit is reached without resolution. |

---

## 2. Functional Requirements

### FR-01: Custom Gymnasium Environment

**Description:**
The project must implement a custom Gymnasium environment class (`CustomerSupportEnv`) that inherits from `gymnasium.Env` and conforms to the Gymnasium API. The environment must implement `reset()`, `step()`, `render()`, and `close()` methods.

**Acceptance Criteria:**

- AC-01.1: The environment class is a subclass of `gymnasium.Env`.
- AC-01.2: `reset()` returns a 2-tuple `(observation, info)`.
- AC-01.3: `step(action)` returns a 5-tuple `(observation, reward, terminated, truncated, info)`.
- AC-01.4: The environment registers with Gymnasium under the namespace `CustomerSupport-v0`.
- AC-01.5: The environment accepts an optional `max_turns` constructor parameter with a default value.
- AC-01.6: The environment accepts an optional `seed` parameter for reproducibility.

**Expected Behavior:**

When `reset()` is called, the environment samples a new customer problem, initializes the conversation state to turn 0, generates the initial user message, and returns the initial observation along with an info dict. When `step(action)` is called, the environment processes the agent's action, generates a simulated user response, updates internal state, computes reward, and checks termination/truncation conditions. Calling `render()` outputs a human-readable representation of the current conversation to the console.

---

### FR-02: Customer Support Scenarios

**Description:**
The environment must simulate multiple customer support problem types, each with a defined set of required information fields, a correct resolution action, and a realistic conversation flow. The simulated user must respond to agent actions in a state-dependent manner.

**Acceptance Criteria:**

- AC-02.1: At least 5 distinct customer problem types are defined (e.g., billing dispute, technical glitch, account lockout, order not received, refund request).
- AC-02.2: Each problem type specifies: a display name, a list of required information fields, and the action that constitutes a correct resolution.
- AC-02.3: The simulated user provides the requested information when the agent asks for it, but only if that information is relevant to the problem.
- AC-02.4: The simulated user confirms resolution when the correct solution is proposed and all required information has been gathered.
- AC-02.5: The simulated user rejects an incorrect solution proposal, increasing a frustration counter.
- AC-02.6: Problem selection at reset is random but controllable via the seed parameter.

**Expected Behavior:**

On `reset()`, a problem type is randomly selected from the defined set. The user's initial message reflects the problem type. During the episode, the user responds differently depending on whether the agent is gathering information, diagnosing, proposing a solution, or asking irrelevant questions. The user's frustration level increases when the agent repeats actions, proposes incorrect solutions, or asks for irrelevant information.

---

### FR-03: Observation Space

**Description:**
The environment must define a structured observation space that encodes the full conversation state at each turn, enabling the agent to make informed decisions.

**Acceptance Criteria:**

- AC-03.1: The observation space is a `gymnasium.spaces.Dict` containing named sub-spaces.
- AC-03.2: The observation includes a `user_message_type` field (integer) identifying the category of the user's latest message.
- AC-03.3: The observation includes a `turn_count` field (integer) indicating the current turn number, bounded by `[0, max_turns]`.
- AC-03.4: The observation includes a `problem_type` field (integer) identifying the sampled problem type, bounded by `[0, num_problem_types - 1]`.
- AC-03.5: The observation includes an `info_collected` field (binary vector) indicating which required information fields have been gathered.
- AC-03.6: The observation includes a `frustration_level` field (float) in range `[0.0, 1.0]` representing the user's current frustration.
- AC-03.7: Every observation returned by `reset()` and `step()` is contained within the defined observation space (`observation_space.contains(obs)` returns `True`).

**Expected Behavior:**

At each turn, the observation encodes: what the user just said (message type), how far into the conversation we are (turn count), what the problem is (problem type), what we know (info collected vector), and how frustrated the user is (frustration level). This gives the agent sufficient information to select an appropriate action without requiring raw text processing.

---

### FR-04: Discrete Action Space

**Description:**
The environment must define a discrete action space representing the set of actions available to the support agent at each turn.

**Acceptance Criteria:**

- AC-04.1: The action space is a `gymnasium.spaces.Discrete(n)` instance where `n` equals the number of defined agent actions.
- AC-04.2: The following actions are defined at minimum:
  - `0` — `ASK_ACCOUNT_INFO`: Request account/identity information.
  - `1` — `ASK_BILLING_INFO`: Request billing or payment details.
  - `2` — `ASK_TECHNICAL_INFO`: Request technical details (device, error message, etc.).
  - `3` — `DIAGNOSE`: Attempt to diagnose the problem based on collected info.
  - `4` — `PROPOSE_SOLUTION`: Propose a solution to the user.
  - `5` — `ESCALATE`: Escalate to a human agent.
  - `6` — `APOLOGIZE`: Apologize to reduce user frustration.
- AC-04.3: Each action integer has a human-readable name accessible via a mapping.
- AC-04.4: The action space is accessible via `env.action_space`.
- AC-04.5: Passing an action outside the valid range to `step()` raises a `ValueError` or `gymnasium` validation error.

**Expected Behavior:**

The agent selects an integer action at each turn. The environment interprets that integer as a specific support action, executes the corresponding behavior, and returns the resulting observation and reward. Invalid actions are rejected before any state mutation occurs.

---

### FR-05: Reward Calculation

**Description:**
The environment must compute a scalar reward at each step that incentivizes efficient, correct problem resolution while penalizing wasted turns and incorrect actions.

**Acceptance Criteria:**

- AC-05.1: Correct problem resolution (correct solution with all required info gathered) yields a reward of `+10.0`.
- AC-05.2: Gathering a previously-uncollected required information field yields a reward of `+1.0`.
- AC-05.3: Each turn taken incurs an efficiency penalty of `-0.5`.
- AC-05.4: Proposing an incorrect solution yields a reward of `-3.0`.
- AC-05.5: Repeating the same action consecutively **without making meaningful progress** (no new information gathered) yields a reward of `-2.0`. Productive repeated actions — where the same ASK_* action gathers a different required field each time — are exempt from this penalty. APOLOGIZE is always exempt.
- AC-05.6: Escalation yields a reward of `+2.0` (acceptable fallback, not ideal).
- AC-05.7: Exceeding the turn limit without resolution yields a terminal reward of `-5.0`.
- AC-05.8: Apologizing when frustration is above a threshold (e.g., 0.5) yields a reward of `+0.5`; apologizing when frustration is low yields `0.0`.
- AC-05.9: Reward values are always finite floating-point numbers.
- AC-05.10: The reward for a terminal step (resolution, escalation, or truncation) includes only the terminal event reward plus the standard turn penalty, not stacked bonuses.

**Expected Behavior:**

At each `step()`, the environment evaluates the agent's action against the current conversation state and computes a reward according to the table above. Rewards accumulate across the episode. An optimal agent gathers all required information efficiently, proposes the correct solution, and terminates with the maximum possible cumulative reward. The reward function discourages looping, incorrect solutions, and unnecessary escalation.

---

### FR-06: Episode Termination

**Description:**
The environment must clearly define the conditions under which an episode terminates (the `terminated` flag is set to `True`), indicating the conversation has reached a natural conclusion.

**Acceptance Criteria:**

- AC-06.1: `terminated` is set to `True` when the agent proposes a correct solution and all required information has been gathered (successful resolution).
- AC-06.2: `terminated` is set to `True` when the agent escalates to a human agent.
- AC-06.3: `terminated` is set to `True` when the maximum turn limit is reached without resolution (in this case `truncated` may also be `True`).
- AC-06.4: `terminated` is `False` for all non-terminal steps.
- AC-06.5: When `terminated` is `True`, the info dict includes a `termination_reason` key with a descriptive string (e.g., `"resolved"`, `"escalated"`, `"truncated"`).
- AC-06.6: The environment does not allow further `step()` calls after termination without calling `reset()`.

**Expected Behavior:**

An episode continues turn-by-turn until one of three terminal conditions is met: successful resolution, escalation, or truncation. Once any terminal condition is reached, the step returns `terminated=True` (and/or `truncated=True`), and the episode is over. The info dict explains why the episode ended.

---

### FR-07: Maximum Turn Limit

**Description:**
The environment must enforce a configurable maximum number of turns per episode to prevent infinite conversations and encourage efficient resolution.

**Acceptance Criteria:**

- AC-07.1: The `max_turns` parameter is accepted in the environment constructor with a default value of `15`.
- AC-07.2: When `turn_count` reaches `max_turns`, the next `step()` call sets `truncated=True` and applies the truncation penalty of `-5.0`.
- AC-07.3: `max_turns` must be a positive integer; a value of `0` or negative raises a `ValueError` at construction time.
- AC-07.4: The `turn_count` field in the observation never exceeds `max_turns`.
- AC-07.5: The `truncated` flag is `False` for all steps before the turn limit is reached.

**Expected Behavior:**

Each episode is bounded by `max_turns`. If the agent has not resolved or escalated the problem by the time the turn limit is reached, the episode ends with truncation. The turn limit is configurable per environment instance, allowing experimentation with shorter or longer conversations.

---

### FR-08: Rule-Based Baseline Agent

**Description:**
The project must include a rule-based baseline agent that follows a deterministic, scripted policy to resolve customer problems. This agent serves as a performance benchmark.

**Acceptance Criteria:**

- AC-08.1: The agent class implements an `act(observation)` method that accepts an observation dict and returns a valid action integer.
- AC-08.2: The agent policy follows this priority order:
  1. If frustration level is above 0.7, apologize to reduce frustration.
  2. If not all required information fields are collected, request the next uncollected field using the appropriate `ASK_*` action.
  3. If all information is collected, diagnose the problem (`DIAGNOSE`).
  4. After diagnosing, propose a solution (`PROPOSE_SOLUTION`).
  5. If the proposed solution is rejected (frustration increases after `PROPOSE_SOLUTION`), escalate (`ESCALATE`).
- AC-08.3: The agent does not repeat the same action more than twice consecutively unless the action is `APOLOGIZE`.
- AC-08.4: The agent is deterministic given the same observation (no randomness in action selection).
- AC-08.5: Every action returned by the agent is within the valid action space.

**Expected Behavior:**

The rule-based agent systematically gathers information, diagnoses, and proposes a solution. It apologizes when the user is frustrated, avoids repeating itself, and escalates when its proposed solution fails. Because it is deterministic, running it against the same sequence of problems (fixed seed) always produces the same actions and rewards.

---

### FR-09: Episode Evaluation

**Description:**
The project must include an evaluation harness that runs a specified number of episodes using the rule-based agent, collecting per-episode data and computing aggregate statistics.

**Acceptance Criteria:**

- AC-09.1: The evaluation function accepts parameters: `num_episodes` (default `100`), `max_turns` (default `15`), `seed` (default `42`), and `agent` (default: rule-based agent instance).
- AC-09.2: For each episode, the harness records: total reward, number of turns, termination reason, and a boolean success flag (True if `termination_reason == "resolved"`).
- AC-09.3: The harness resets the environment with the provided seed (or a derived per-episode seed) before each episode.
- AC-09.4: The harness runs the full episode loop: `reset()` → agent `act()` loop → `step()` → until `terminated` or `truncated`.
- AC-09.5: The harness does not crash if an episode terminates immediately (e.g., 1-turn escalation).
- AC-09.6: The harness handles environment exceptions gracefully and logs them without halting the entire evaluation.

**Expected Behavior:**

The evaluation harness orchestrates 100 episodes (by default) of the rule-based agent against the environment. It collects the outcome of each episode and passes the raw data to the statistics computation step. The harness is callable from both the command line and as a Python function.

---

### FR-10: Performance Metrics

**Description:**
The evaluation harness must compute aggregate performance metrics from the collected per-episode data.

**Acceptance Criteria:**

- AC-10.1: The following metrics are computed and returned:
  - `mean_reward` — Mean total reward across all episodes.
  - `std_reward` — Standard deviation of total reward.
  - `min_reward` — Minimum total reward observed.
  - `max_reward` — Maximum total reward observed.
  - `success_rate` — Fraction of episodes resolved successfully (termination_reason == "resolved"), as a float in `[0.0, 1.0]`.
  - `mean_turns` — Mean number of turns per episode.
  - `std_turns` — Standard deviation of turns per episode.
  - `escalation_rate` — Fraction of episodes that ended in escalation.
  - `truncation_rate` — Fraction of episodes that ended in truncation.
  - `num_episodes` — Total number of episodes run.
- AC-10.2: All numeric metrics are finite floating-point values (or integers where appropriate).
- AC-10.3: `success_rate + escalation_rate + truncation_rate` equals `1.0` (every episode ends in exactly one terminal state), accounting for floating-point tolerance.
- AC-10.4: Metrics are computed using NumPy for numerical operations.

**Expected Behavior:**

After all episodes complete, the harness aggregates the per-episode records and computes the metrics listed above. The metrics provide a complete picture of agent performance: reward quality, resolution efficiency, and failure modes.

---

### FR-11: JSON Evaluation Report

**Description:**
The evaluation harness must output a JSON report file containing the aggregate metrics and per-episode breakdown.

**Acceptance Criteria:**

- AC-11.1: The report is written to `results/evaluation_results.json` relative to the project root.
- AC-11.2: The JSON structure includes a top-level `metrics` object containing all metrics from FR-10.
- AC-11.3: The JSON structure includes a top-level `episodes` array, where each element is an object with keys: `episode_id`, `total_reward`, `num_turns`, `termination_reason`, `success`.
- AC-11.4: The JSON structure includes a top-level `config` object with: `num_episodes`, `max_turns`, `seed`, and `agent_type`.
- AC-11.5: The JSON file is valid and parseable by `json.load()`.
- AC-11.6: The `results/` directory is created if it does not exist.
- AC-11.7: The metrics are also printed to stdout in a human-readable format.

**Expected Behavior:**

After evaluation completes, the harness writes a structured JSON file to disk and prints a summary table to the console. The JSON file can be loaded by downstream analysis tools or inspected manually. The file includes both aggregate statistics and per-episode detail for debugging and further analysis.

---

### FR-12: Error Handling and Validation

**Description:**
The environment, agent, and evaluation harness must validate inputs and handle errors gracefully to prevent silent failures and undefined behavior.

**Acceptance Criteria:**

- AC-12.1: `step()` validates that the action is within the action space; an out-of-range action raises a `ValueError` with a descriptive message.
- AC-12.2: `step()` called after termination (without `reset()`) raises a `RuntimeError` with a message indicating the episode has ended.
- AC-12.3: `reset()` called mid-episode is allowed and reinitializes the environment (no error).
- AC-12.4: The environment constructor validates that `max_turns` is a positive integer.
- AC-12.5: The agent's `act()` method validates that the observation contains all expected keys; missing keys raise a `KeyError` with a descriptive message.
- AC-12.6: The evaluation harness validates that `num_episodes` is a positive integer.
- AC-12.7: The evaluation harness validates that `seed` is an integer or `None`.
- AC-12.8: The environment does not mutate its internal state if an invalid action is passed to `step()` (validation occurs before state update).
- AC-12.9: All error messages are descriptive and actionable, referencing the parameter or state that caused the error.

**Expected Behavior:**

Invalid inputs are caught early with clear error messages. The environment never enters an inconsistent state due to invalid actions. The evaluation harness does not silently skip or corrupt data on error. All validation errors are raised as standard Python exceptions (`ValueError`, `KeyError`, `RuntimeError`) with messages that help the developer diagnose the issue.

---

## 3. Non-Functional Requirements

### NFR-01: Maintainability

**Description:**
The codebase must be structured for readability, modularity, and ease of modification.

**Acceptance Criteria:**

- AC-NFR-01.1: Code is organized into clear modules: `env/`, `agents/`, `evaluation/`, `tests/`.
- AC-NFR-01.2: All public classes and functions have docstrings following the Google or NumPy docstring convention.
- AC-NFR-01.3: No function exceeds 50 lines of logic (excluding docstrings and blank lines).
- AC-NFR-01.4: Constants (reward values, action names, problem types) are defined in a single configuration location and not scattered as magic numbers.
- AC-NFR-01.5: The environment class does not exceed 300 lines total.

### NFR-02: Reproducibility

**Description:**
The environment and evaluation must produce identical results given the same seed.

**Acceptance Criteria:**

- AC-NFR-02.1: All randomness in the environment (problem sampling, user behavior) uses a single `numpy.random.Generator` instance seeded via `reset(seed=...)`.
- AC-NFR-02.2: Running the evaluation harness twice with the same seed produces identical metrics and per-episode data.
- AC-NFR-02.3: The rule-based agent is fully deterministic (no randomness).
- AC-NFR-02.4: The evaluation report JSON includes the seed used, enabling result reproduction.

### NFR-03: Testability

**Description:**
The project must be fully testable with Pytest, covering all major components and edge cases.

**Acceptance Criteria:**

- AC-NFR-03.1: The test suite runs successfully via `pytest tests/` with exit code 0.
- AC-NFR-03.2: Tests cover environment API compliance, reward calculation, termination logic, agent behavior, and evaluation metrics.
- AC-NFR-03.3: Tests use fixed seeds for reproducibility.
- AC-NFR-03.4: Tests include edge cases: immediate escalation, turn-limit truncation, repeated actions, and invalid actions.
- AC-NFR-03.5: Tests do not depend on external network access or files outside the project directory.
- AC-NFR-03.6: Test execution completes in under 30 seconds for the full suite.
- AC-NFR-03.7: No test relies on floating-point exact equality; all float comparisons use a tolerance (`pytest.approx` or `np.isclose`).

### NFR-04: Dependency Management

**Description:**
Project dependencies must be explicitly declared and pinned.

**Acceptance Criteria:**

- AC-NFR-04.1: A `requirements.txt` file lists all runtime dependencies with pinned versions.
- AC-NFR-04.2: A `requirements-dev.txt` file (or equivalent) lists test dependencies with pinned versions.
- AC-NFR-04.3: The project has no dependencies beyond `gymnasium`, `numpy`, and `pytest` (and their transitive dependencies).

### NFR-05: Code Quality

**Description:**
The code must follow Python best practices and be free of common issues.

**Acceptance Criteria:**

- AC-NFR-05.1: No bare `except:` clauses; all exception handlers catch specific exception types.
- AC-NFR-05.2: No `import *` statements anywhere in the codebase.
- AC-NFR-05.3: All modules use explicit imports.
- AC-NFR-05.4: No mutable default arguments in function or method signatures.
- AC-NFR-05.5: Type hints are used on all public function and method signatures.

---

## 4. Requirement Traceability

| Requirement ID | Component | Tested By |
|---|---|---|
| FR-01 | `env/customer_support_env.py` | `test_env.py` |
| FR-02 | `env/conversation.py` | `test_env.py` |
| FR-03 | `env/spaces.py`, `env/customer_support_env.py` | `test_env.py` |
| FR-04 | `env/spaces.py`, `env/customer_support_env.py` | `test_env.py` |
| FR-05 | `env/customer_support_env.py` | `test_env.py` |
| FR-06 | `env/customer_support_env.py` | `test_env.py` |
| FR-07 | `env/customer_support_env.py` | `test_env.py` |
| FR-08 | `agents/rule_based_agent.py` | `test_agent.py` |
| FR-09 | `evaluation/run_evaluation.py` | `test_evaluation.py` |
| FR-10 | `evaluation/run_evaluation.py` | `test_evaluation.py` |
| FR-11 | `evaluation/run_evaluation.py` | `test_evaluation.py` |
| FR-12 | All modules | `test_env.py`, `test_agent.py`, `test_evaluation.py` |
| NFR-01 | All modules | Manual review |
| NFR-02 | `env/customer_support_env.py`, `evaluation/run_evaluation.py` | `test_env.py`, `test_evaluation.py` |
| NFR-03 | `tests/` | `pytest tests/` exit code |
| NFR-04 | `requirements.txt`, `requirements-dev.txt` | Manual review |
| NFR-05 | All modules | Manual review |
