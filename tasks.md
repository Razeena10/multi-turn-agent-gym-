# Implementation Tasks — multi-turn-agent-gym

## Overview

This document breaks the project into 7 development waves, each containing small, sequential tasks. Every task references the requirement IDs it satisfies (from `requirements.md`), includes implementation details (from `design.md`), expected outputs, and verification commands. Tasks within a wave are sequential unless explicitly marked as independent. Waves are sequential — each wave depends on the prior wave's outputs.

**Working directory:** `C:\Week3\multi-turn-agent-gym`
**Python:** 3.14 | **Shell:** PowerShell on Windows

---

## Wave 1: Project Foundation

**Goal:** Set up the project skeleton, install dependencies, and define all centralized constants.

**Depends on:** Nothing (starting point)

---

### Task 1.1: Install Gymnasium

**Requirements:** NFR-04 (AC-NFR-04.1, AC-NFR-04.3)

**Implementation details:**
- Run `pip install gymnasium` to install the Gymnasium package.
- Verify it imports correctly with a simple Python one-liner.
- Record the installed version for pinning in Task 1.3.

**Expected output:**
- Gymnasium installed in the Python environment.
- Version printed to console (e.g., `1.0.0` or similar).

**Verification:**
```powershell
pip install gymnasium
python -c "import gymnasium; print(gymnasium.__version__)"
```

---

### Task 1.2: Create directory structure and package init files

**Requirements:** NFR-01 (AC-NFR-01.1)

**Implementation details:**
- Create the following directories: `env/`, `agents/`, `evaluation/`, `tests/`
- Create `__init__.py` files in each package directory:
  - `env/__init__.py` — will export `CustomerSupportEnv` and register the environment (populated in Wave 3).
  - `agents/__init__.py` — will export `RuleBasedAgent` (populated in Wave 4).
  - `evaluation/__init__.py` — empty init.
  - `tests/__init__.py` — empty init.
- For now, create these as minimal files. `env/__init__.py` can contain a placeholder comment. The actual exports will be added in later waves when the classes exist.

**Expected output:**
```
env/__init__.py
agents/__init__.py
evaluation/__init__.py
tests/__init__.py
```

**Verification:**
```powershell
python -c "import env; import agents; import evaluation; import tests; print('All packages importable')"
```

---

### Task 1.3: Create requirements.txt and requirements-dev.txt

**Requirements:** NFR-04 (AC-NFR-04.1, AC-NFR-04.2, AC-NFR-04.3)

**Implementation details:**
- Create `requirements.txt` with pinned runtime dependencies:
  - `gymnasium==<installed_version>` (from Task 1.1)
  - `numpy==2.5.4` (already installed, pin the version)
- Create `requirements-dev.txt` with pinned test dependencies:
  - `pytest==9.1.1` (already installed, pin the version)
  - Include `-r requirements.txt` as the first line to pull in runtime deps.

**Expected output:**
- `requirements.txt` with 2 pinned lines.
- `requirements-dev.txt` with 3 lines (reference + pytest).

**Verification:**
```powershell
pip install -r requirements-dev.txt --dry-run
```

---

### Task 1.4: Create env/config.py — action and message type constants

**Requirements:** FR-04 (AC-04.2, AC-04.3), NFR-01 (AC-NFR-01.4)

**Implementation details:**
- Create `env/config.py` with all action constants as module-level integers:
  ```python
  ASK_ACCOUNT_INFO = 0
  ASK_BILLING_INFO = 1
  ASK_TECHNICAL_INFO = 2
  DIAGNOSE = 3
  PROPOSE_SOLUTION = 4
  ESCALATE = 5
  APOLOGIZE = 6
  ```
- Define `NUM_ACTIONS = 7`.
- Define `ACTION_NAMES` dict mapping each integer to its human-readable name string:
  ```python
  ACTION_NAMES = {
      0: "ASK_ACCOUNT_INFO",
      1: "ASK_BILLING_INFO",
      2: "ASK_TECHNICAL_INFO",
      3: "DIAGNOSE",
      4: "PROPOSE_SOLUTION",
      5: "ESCALATE",
      6: "APOLOGIZE",
  }
  ```
- Define message type constants (8 types per design Section 5.3):
  ```python
  MSG_INITIAL_PROBLEM = 0
  MSG_INFO_PROVIDED = 1
  MSG_NO_RELEVANT_INFO = 2
  MSG_SOLUTION_ACCEPTED = 3
  MSG_SOLUTION_REJECTED = 4
  MSG_FRUSTRATED = 5
  MSG_ESCALATION_CONFIRM = 6
  MSG_DIAGNOSIS_ACK = 7
  ```
- Define `NUM_MESSAGE_TYPES = 8`.
- Define `MESSAGE_TYPE_NAMES` dict mapping each integer to name string.

**Expected output:**
- `env/config.py` with all action and message type constants.

**Verification:**
```powershell
python -c "from env.config import NUM_ACTIONS, NUM_MESSAGE_TYPES, ACTION_NAMES; print(NUM_ACTIONS, NUM_MESSAGE_TYPES); print(ACTION_NAMES[0])"
```

---

### Task 1.5: Create env/config.py — reward constants

**Requirements:** FR-05 (AC-05.1 through AC-05.8), NFR-01 (AC-NFR-01.4)

**Implementation details:**
- Add reward constants to `env/config.py` (same file as Task 1.4):
  ```python
  REWARD_RESOLVE = 10.0
  REWARD_INFO_GATHERED = 1.0
  REWARD_TURN_PENALTY = -0.5
  REWARD_WRONG_SOLUTION = -3.0
  REWARD_REPEATED_ACTION = -2.0
  REWARD_ESCALATE = 2.0
  REWARD_TRUNCATE = -5.0
  REWARD_APOLOGIZE_HIGH_FRUSTRATION = 0.5
  REWARD_APOLOGIZE_LOW_FRUSTRATION = 0.0
  ```
- Define frustration thresholds:
  ```python
  FRUSTRATION_APOLOGY_THRESHOLD = 0.5
  FRUSTRATION_HIGH_THRESHOLD = 0.7
  FRUSTRATION_INCREASE_IRRELEVANT = 0.15
  FRUSTRATION_INCREASE_REPEATED = 0.1
  FRUSTRATION_INCREASE_WRONG_SOLUTION = 0.2
  FRUSTRATION_INCREASE_PREMATURE_DIAGNOSIS = 0.1
  FRUSTRATION_DECREASE_DIAGNOSIS = 0.05
  FRUSTRATION_DECREASE_APOLOGIZE = 0.3
  ```

**Expected output:**
- `env/config.py` now includes all reward and frustration constants.

**Verification:**
```powershell
python -c "from env.config import REWARD_RESOLVE, REWARD_TURN_PENALTY; print(REWARD_RESOLVE, REWARD_TURN_PENALTY)"
```

---

### Task 1.6: Create env/config.py — default parameters and field-to-action mapping

**Requirements:** NFR-01 (AC-NFR-01.4), FR-04 (AC-04.2)

**Implementation details:**
- Add default parameters to `env/config.py`:
  ```python
  DEFAULT_MAX_TURNS = 15
  DEFAULT_SEED = 42
  DEFAULT_NUM_EPISODES = 100
  ```
- Define `FIELD_TO_ACTION` reverse-lookup dict (per design Section 10.3) mapping each info field name to the action that gathers it:
  ```python
  FIELD_TO_ACTION = {
      "account_id": ASK_ACCOUNT_INFO,
      "email": ASK_ACCOUNT_INFO,
      "last_charge": ASK_BILLING_INFO,
      "payment_method": ASK_BILLING_INFO,
      "order_number": ASK_BILLING_INFO,
      "transaction_id": ASK_BILLING_INFO,
      "refund_reason": ASK_BILLING_INFO,
      "device_type": ASK_TECHNICAL_INFO,
      "error_message": ASK_TECHNICAL_INFO,
  }
  ```
- Define `MAX_INFO_FIELDS = 4` (the maximum number of required fields across all problem types — refund request has 3, technical glitch has 3, etc.; set to 4 for padding).

**Expected output:**
- `env/config.py` now includes defaults and the FIELD_TO_ACTION mapping.

**Verification:**
```powershell
python -c "from env.config import FIELD_TO_ACTION, DEFAULT_MAX_TURNS; print(FIELD_TO_ACTION['account_id']); print(DEFAULT_MAX_TURNS)"
```

---

## Wave 2: Customer Support Scenarios

**Goal:** Define problem types, conversation state, and the simulated user state machine.

**Depends on:** Wave 1 (config.py with constants)

---

### Task 2.1: Define ProblemType dataclass and 5 problem types in config.py

**Requirements:** FR-02 (AC-02.1, AC-02.2), NFR-01 (AC-NFR-01.4)

**Implementation details:**
- Add a frozen `@dataclass` `ProblemType` to `env/config.py` with fields (per design Section 9.1):
  - `id: int`
  - `name: str`
  - `required_fields: tuple[str, ...]`
  - `resolution_action: int` (always `PROPOSE_SOLUTION = 4`)
  - `ask_field_mapping: dict[int, list[str]]` — maps action ID to list of field names that action gathers (in order; one per turn).
- Define 5 problem type instances in a tuple `PROBLEM_TYPES`:
  1. **Billing Dispute** (id=0): required_fields=("account_id", "last_charge", "payment_method"), mapping={ASK_ACCOUNT_INFO: ["account_id"], ASK_BILLING_INFO: ["last_charge", "payment_method"]}
  2. **Technical Glitch** (id=1): required_fields=("account_id", "device_type", "error_message"), mapping={ASK_ACCOUNT_INFO: ["account_id"], ASK_TECHNICAL_INFO: ["device_type", "error_message"]}
  3. **Account Lockout** (id=2): required_fields=("account_id", "email"), mapping={ASK_ACCOUNT_INFO: ["account_id", "email"]}
  4. **Order Not Received** (id=3): required_fields=("account_id", "order_number"), mapping={ASK_ACCOUNT_INFO: ["account_id"], ASK_BILLING_INFO: ["order_number"]}
  5. **Refund Request** (id=4): required_fields=("account_id", "transaction_id", "refund_reason"), mapping={ASK_ACCOUNT_INFO: ["account_id"], ASK_BILLING_INFO: ["transaction_id", "refund_reason"]}
- Define `NUM_PROBLEM_TYPES = len(PROBLEM_TYPES)`.
- Use `from __future__ import annotations` at the top of config.py for `dict[int, list[str]]` type hint compatibility, or use `Dict[int, List[str]]` from `typing`.

**Expected output:**
- `env/config.py` now includes `ProblemType` dataclass and `PROBLEM_TYPES` tuple with 5 entries.

**Verification:**
```powershell
python -c "from env.config import PROBLEM_TYPES, NUM_PROBLEM_TYPES; print(NUM_PROBLEM_TYPES); print([p.name for p in PROBLEM_TYPES])"
```

---

### Task 2.2: Create env/conversation.py — ConversationState dataclass

**Requirements:** FR-02 (AC-02.2), design Section 9.4

**Implementation details:**
- Create `env/conversation.py`.
- Import `ProblemType` from `env.config`.
- Define a mutable `@dataclass` `ConversationState` with fields (per design Section 9.4):
  ```python
  problem: ProblemType
  turn_count: int
  info_collected: dict[str, bool]
  frustration_level: float
  last_action: int | None
  consecutive_action_count: int
  diagnosed: bool
  solution_proposed: bool
  solution_rejected: bool
  user_message_type: int
  ```
- Add a `@classmethod` or factory function `init_state(problem: ProblemType) -> ConversationState` that creates a fresh state with:
  - turn_count = 0
  - info_collected = {field: False for field in problem.required_fields}
  - frustration_level = 0.0
  - last_action = None
  - consecutive_action_count = 0
  - diagnosed = False
  - solution_proposed = False
  - solution_rejected = False
  - user_message_type = MSG_INITIAL_PROBLEM (0)

**Expected output:**
- `env/conversation.py` with `ConversationState` dataclass and `init_state` factory.

**Verification:**
```powershell
python -c "from env.conversation import ConversationState, init_state; from env.config import PROBLEM_TYPES; s = init_state(PROBLEM_TYPES[0]); print(s.problem.name, s.turn_count, s.frustration_level)"
```

---

### Task 2.3: Create env/conversation.py — ProblemRegistry

**Requirements:** FR-02 (AC-02.6), NFR-02 (AC-NFR-02.1)

**Implementation details:**
- Add `ProblemRegistry` class to `env/conversation.py`:
  - `__init__(self, rng: np.random.Generator)` — stores the RNG.
  - `sample(self) -> ProblemType` — returns a random `ProblemType` from `PROBLEM_TYPES` using `self._rng.integers(0, NUM_PROBLEM_TYPES)`.
- Import `PROBLEM_TYPES`, `NUM_PROBLEM_TYPES` from `env.config`.
- Import `numpy as np`.
- The registry is a thin wrapper that decouples problem sampling from the environment class. The environment passes its RNG to the registry.

**Expected output:**
- `ProblemRegistry` class in `env/conversation.py`.

**Verification:**
```powershell
python -c "from env.conversation import ProblemRegistry; import numpy as np; r = ProblemRegistry(np.random.default_rng(42)); p = r.sample(); print(p.name)"
```

---

### Task 2.4: Create env/conversation.py — UserStateMachine (ASK handlers)

**Requirements:** FR-02 (AC-02.3, AC-02.5), design Section 9.5

**Implementation details:**
- Add `UserStateMachine` class to `env/conversation.py`.
- Define a response dataclass `UserResponse` with fields:
  ```python
  message_type: int          # The user's message type for the observation
  info_gathered: bool        # Whether new info was collected this turn
  info_field: str | None     # Which field was gathered (if any)
  frustration_delta: float   # Change to frustration level (applied + clamped by env)
  solution_accepted: bool    # Whether solution was accepted
  solution_rejected: bool    # Whether solution was rejected
  terminated: bool           # Whether this action terminates the episode
  ```
- Implement `process_action(state: ConversationState, action: int) -> UserResponse` as the main dispatch method.
- Implement `handle_ask_account_info(state)`, `handle_ask_billing_info(state)`, `handle_ask_technical_info(state)` — each follows the same pattern:
  1. Look up the action in `state.problem.ask_field_mapping`.
  2. If mapping exists and has uncollected fields, mark the first uncollected field as collected, return `MSG_INFO_PROVIDED`, `info_gathered=True`.
  3. If mapping exists but all fields collected, return `MSG_NO_RELEVANT_INFO`, frustration_delta=+0.1.
  4. If no mapping (empty list or key absent), return `MSG_NO_RELEVANT_INFO`, frustration_delta=+0.15.
- The three ASK handlers can share a private helper `_handle_ask(state, action_id)` to avoid duplication, with the public methods delegating to it.

**Expected output:**
- `UserStateMachine` with `process_action` dispatch and three ASK handlers in `env/conversation.py`.

**Verification:**
```powershell
python -c "from env.conversation import UserStateMachine, ConversationState, init_state; from env.config import PROBLEM_TYPES, ASK_ACCOUNT_INFO; s = init_state(PROBLEM_TYPES[0]); r = UserStateMachine.process_action(s, ASK_ACCOUNT_INFO); print(r.message_type, r.info_gathered, r.info_field)"
```

---

### Task 2.5: Create env/conversation.py — UserStateMachine (remaining handlers)

**Requirements:** FR-02 (AC-02.4, AC-02.5), FR-05 (AC-05.1 through AC-05.8), design Section 9.5

**Implementation details:**
- Implement remaining handlers in `UserStateMachine`:
  - `handle_diagnose(state)`:
    - If all info collected: return `MSG_DIAGNOSIS_ACK`, set `state.diagnosed = True`, frustration_delta=-0.05.
    - If info incomplete: return `MSG_FRUSTRATED`, set `state.diagnosed = True`, frustration_delta=+0.1.
  - `handle_propose_solution(state)`:
    - If all info collected AND `diagnosed == True`: return `MSG_SOLUTION_ACCEPTED`, `solution_accepted=True`, `terminated=True`.
    - If all info collected but `diagnosed == False`: return `MSG_SOLUTION_REJECTED`, `solution_rejected=True`, frustration_delta=+0.2.
    - If info incomplete: return `MSG_SOLUTION_REJECTED`, `solution_rejected=True`, frustration_delta=+0.2.
  - `handle_escalate(state)`:
    - Always: return `MSG_ESCALATION_CONFIRM`, `terminated=True`.
  - `handle_apologize(state)`:
    - If frustration > 0.0: frustration_delta=-0.3, return `MSG_INFO_PROVIDED` (ack).
    - If frustration == 0.0: no change, return `MSG_INFO_PROVIDED` (ack).
- Add a helper `_all_info_collected(state) -> bool` that checks all values in `state.info_collected` are True.
- Wire `process_action` to dispatch to the correct handler based on the action integer.

**Expected output:**
- `UserStateMachine` fully implemented with all 7 action handlers.

**Verification:**
```powershell
python -c "from env.conversation import UserStateMachine, ConversationState, init_state; from env.config import PROBLEM_TYPES, ESCALATE; s = init_state(PROBLEM_TYPES[0]); r = UserStateMachine.process_action(s, ESCALATE); print(r.message_type, r.terminated)"
```

---

### Task 2.6: Verify conversation module end-to-end

**Requirements:** FR-02 (all acceptance criteria), NFR-02 (AC-NFR-02.1)

**Implementation details:**
- Write a quick manual verification script (inline Python) that:
  1. Creates a `ProblemRegistry` with a fixed seed.
  2. Samples a problem.
  3. Initializes a `ConversationState`.
  4. Runs a full conversation: ask all required info fields → diagnose → propose solution.
  5. Checks that the final response has `solution_accepted=True`, `terminated=True`.
  6. Repeats with a second seed to confirm reproducibility.
- This is not a permanent test — just a sanity check before building the environment on top.

**Expected output:**
- Console output showing the conversation flow and final resolution status.

**Verification:**
```powershell
python -c "from env.conversation import ProblemRegistry, ConversationState, init_state, UserStateMachine; from env.config import *; import numpy as np; r = ProblemRegistry(np.random.default_rng(42)); p = r.sample(); s = init_state(p); print('Problem:', p.name); [print('Gathered:', UserStateMachine.process_action(s, FIELD_TO_ACTION[f]).info_field) for f in p.required_fields if UserStateMachine.process_action(s, FIELD_TO_ACTION[f]).info_gathered]; resp = UserStateMachine.process_action(s, DIAGNOSE); print('Diagnosed:', resp.message_type); resp = UserStateMachine.process_action(s, PROPOSE_SOLUTION); print('Resolved:', resp.solution_accepted, resp.terminated)"
```

---

## Wave 3: Custom Gymnasium Environment

**Goal:** Build the `CustomerSupportEnv` class that wraps the conversation logic in a Gymnasium-compatible API.

**Depends on:** Wave 2 (conversation.py, config.py fully defined)

---

### Task 3.1: Create env/spaces.py — action and observation space builders

**Requirements:** FR-03 (AC-03.1, AC-03.2, AC-03.3, AC-03.4, AC-03.5, AC-03.6), FR-04 (AC-04.1, AC-04.4), design Section 5.1

**Implementation details:**
- Create `env/spaces.py`.
- Import `gymnasium.spaces as spaces` and `numpy as np`.
- Import constants from `env/config`: `NUM_ACTIONS`, `NUM_MESSAGE_TYPES`, `NUM_PROBLEM_TYPES`, `MAX_INFO_FIELDS`, `DEFAULT_MAX_TURNS`.
- Implement `build_action_space() -> spaces.Discrete`:
  - Returns `spaces.Discrete(NUM_ACTIONS)`.
- Implement `build_observation_space(max_turns: int = DEFAULT_MAX_TURNS) -> spaces.Dict`:
  - Returns:
    ```python
    spaces.Dict({
        "user_message_type": spaces.Discrete(NUM_MESSAGE_TYPES, start=0),
        "turn_count": spaces.Discrete(max_turns + 1, start=0),
        "problem_type": spaces.Discrete(NUM_PROBLEM_TYPES, start=0),
        "info_collected": spaces.MultiBinary(MAX_INFO_FIELDS),
        "frustration_level": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
    })
    ```
- Implement `build_observation(state: ConversationState) -> dict`:
  - Converts `ConversationState` fields to NumPy types:
    - `user_message_type` → `np.int64(state.user_message_type)`
    - `turn_count` → `np.int64(state.turn_count)`
    - `problem_type` → `np.int64(state.problem.id)`
    - `info_collected` → `np.ndarray` of shape `(MAX_INFO_FIELDS,)`, dtype `np.int8`. Build by iterating over `MAX_INFO_FIELDS` positions; for each position `i`, check if there's a field at that index in `state.problem.required_fields` and whether it's collected. Positions beyond `len(required_fields)` are always 0.
    - `frustration_level` → `np.float32(state.frustration_level)`
  - Returns the dict.

**Expected output:**
- `env/spaces.py` with `build_action_space()`, `build_observation_space()`, and `build_observation()`.

**Verification:**
```powershell
python -c "from env.spaces import build_action_space, build_observation_space; a = build_action_space(); o = build_observation_space(15); print(type(a), type(o)); print(list(o.spaces.keys()))"
```

---

### Task 3.2: Create env/customer_support_env.py — __init__ and reset

**Requirements:** FR-01 (AC-01.1, AC-01.2, AC-01.5, AC-01.6), FR-07 (AC-07.1, AC-07.3), FR-12 (AC-12.4), design Section 4.2

**Implementation details:**
- Create `env/customer_support_env.py`.
- Import `gymnasium`, `numpy as np`, and from local modules: `config`, `conversation`, `spaces`.
- Define `class CustomerSupportEnv(gymnasium.Env)` with:
  - `__init__(self, max_turns: int = DEFAULT_MAX_TURNS)`:
    - Validate `max_turns`: if not `isinstance(max_turns, int)` or `max_turns <= 0`, raise `ValueError` with descriptive message (AC-12.4, AC-07.3).
    - Store `self.max_turns = max_turns`.
    - Build `self.action_space = build_action_space()`.
    - Build `self.observation_space = build_observation_space(max_turns)`.
    - Initialize `self._rng = None`.
    - Set `self._episode_active = False`.
    - Set `self._state = None`.
  - `reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]`:
    - If `seed is not None`: validate it's an integer, then `self._rng = np.random.default_rng(seed)`.
    - If `seed is None` and `self._rng is None`: `self._rng = np.random.default_rng()`.
    - Create `ProblemRegistry(self._rng)` and sample a problem.
    - Initialize `self._state = init_state(problem)`.
    - Set `self._episode_active = True`.
    - Build observation via `build_observation(self._state)`.
    - Build info dict: `{"problem_type": problem.name, "turn_count": 0}`.
    - Return `(observation, info)`.
  - Add `metadata = {"render_modes": ["human"]}` class attribute.

**Expected output:**
- `env/customer_support_env.py` with `__init__` and `reset` methods.

**Verification:**
```powershell
python -c "from env.customer_support_env import CustomerSupportEnv; e = CustomerSupportEnv(); obs, info = e.reset(seed=42); print(type(obs), list(obs.keys()), info)"
```

---

### Task 3.3: Create env/customer_support_env.py — step (validation + state update)

**Requirements:** FR-01 (AC-01.3), FR-12 (AC-12.1, AC-12.2, AC-12.8), design Section 4.2

**Implementation details:**
- Implement `step(self, action: int) -> tuple[dict, float, bool, bool, dict]`:
  - **Validation phase (before any state mutation — AC-12.8):**
    1. If `not self._episode_active`: raise `RuntimeError("Episode has ended. Call reset() to start a new episode.")` (AC-12.2).
    2. If `not isinstance(action, (int, np.integer))`: raise `ValueError(f"Action must be an integer, got {type(action).__name__}")`.
    3. If `action not in self.action_space`: raise `ValueError(f"Invalid action {action}. Must be in [0, {self.action_space.n}).")` (AC-12.1).
  - **State update phase:**
    4. Increment `self._state.turn_count`.
    5. Call `response = UserStateMachine.process_action(self._state, action)`.
    6. Apply frustration delta: `self._state.frustration_level = max(0.0, min(1.0, self._state.frustration_level + response.frustration_delta))`.
    7. If `response.info_gathered` and `response.info_field`: set `self._state.info_collected[response.info_field] = True`.
    8. Update `self._state.user_message_type = response.message_type`.
    9. Update `self._state.solution_proposed = True` if action == `PROPOSE_SOLUTION`.
    10. Update `self._state.solution_rejected = True` if `response.solution_rejected`.
    11. Update last_action tracking: if `action == self._state.last_action`: increment `consecutive_action_count`; else: set `last_action = action`, `consecutive_action_count = 1`.

**Expected output:**
- `step()` method with validation and state update (reward and termination in next tasks).

**Verification:**
```powershell
python -c "from env.customer_support_env import CustomerSupportEnv; from env.config import ASK_ACCOUNT_INFO; e = CustomerSupportEnv(); e.reset(seed=42); obs, r, t, tr, info = e.step(ASK_ACCOUNT_INFO); print(r, t, tr, info)"
```
(Reward will be 0.0 at this point — computed in Task 3.4. The test just verifies no crash and correct tuple structure.)

---

### Task 3.4: Create env/customer_support_env.py — step (reward computation)

**Requirements:** FR-05 (AC-05.1 through AC-05.10), design Section 7.2

**Implementation details:**
- Add a private method `_compute_reward(self, action: int, response: UserResponse) -> float`:
  - Start: `reward = REWARD_TURN_PENALTY` (always -0.5, AC-05.3).
  - Repeated action check (progress-aware): if `action == self._state.last_action and action != APOLOGIZE and self._state.consecutive_action_count >= 2 and not response.info_gathered`: add `REWARD_REPEATED_ACTION` (-2.0, AC-05.5). The `not response.info_gathered` condition exempts productive repeated ASK actions that collect different required fields each time (e.g., Scenario 3 Password Reset).
  - Info gathered: if `response.info_gathered`: add `REWARD_INFO_GATHERED` (+1.0, AC-05.2).
  - Propose solution: if `action == PROPOSE_SOLUTION`:
    - If `response.solution_accepted`: add `REWARD_RESOLVE` (+10.0, AC-05.1). Do NOT add `REWARD_WRONG_SOLUTION`.
    - Else: add `REWARD_WRONG_SOLUTION` (-3.0, AC-05.4).
  - Escalate: if `action == ESCALATE`: add `REWARD_ESCALATE` (+2.0, AC-05.6).
  - Apologize: if `action == APOLOGIZE`:
    - If `self._state.frustration_level > FRUSTRATION_APOLOGY_THRESHOLD`: add `REWARD_APOLOGIZE_HIGH_FRUSTRATION` (+0.5, AC-05.8).
    - Else: add `REWARD_APOLOGIZE_LOW_FRUSTRATION` (0.0, AC-05.8).
  - Return `reward` as a float.
- Call `reward = self._compute_reward(action, response)` in `step()` after state update.
- Note: the truncation reward (`REWARD_TRUNCATE = -5.0`) is applied in the termination check (Task 3.5), not here.

**Expected output:**
- `step()` now returns correct reward values.

**Verification:**
```powershell
python -c "from env.customer_support_env import CustomerSupportEnv; from env.config import ASK_ACCOUNT_INFO, DIAGNOSE; e = CustomerSupportEnv(); e.reset(seed=42); obs, r, t, tr, info = e.step(ASK_ACCOUNT_INFO); print('Reward after info:', r); assert r == pytest.approx(0.5)" 
```
(Expect +1.0 info bonus - 0.5 turn penalty = +0.5. Note: use `pytest.approx` in actual tests; for manual check, `print(r)` should show `0.5`.)

---

### Task 3.5: Create env/customer_support_env.py — step (termination and truncation)

**Requirements:** FR-06 (AC-06.1 through AC-06.6), FR-07 (AC-07.2, AC-07.4, AC-07.5), design Section 8.3

**Implementation details:**
- Add termination check after reward computation in `step()`:
  - Initialize `terminated = False`, `truncated = False`, `termination_reason = None`.
  - **Check 1 (resolution):** if `action == PROPOSE_SOLUTION and response.solution_accepted`:
    - `terminated = True`, `termination_reason = "resolved"` (AC-06.1).
  - **Check 2 (escalation):** elif `action == ESCALATE`:
    - `terminated = True`, `termination_reason = "escalated"` (AC-06.2).
  - **Check 3 (truncation):** elif `self._state.turn_count >= self.max_turns`:
    - `truncated = True`, `termination_reason = "truncated"` (AC-07.2).
    - Add `REWARD_TRUNCATE` to reward: `reward += REWARD_TRUNCATE` (AC-05.7).
  - These are mutually exclusive — elif chain ensures only one fires.
  - If terminated or truncated: set `self._episode_active = False` (AC-06.6).
  - Build info dict:
    - Always include `"turn_count": self._state.turn_count` and `"frustration_level": self._state.frustration_level`.
    - If terminal: include `"termination_reason": termination_reason` (AC-06.5).
  - Build observation via `build_observation(self._state)`.
  - Return `(observation, reward, terminated, truncated, info)`.

**Expected output:**
- `step()` now handles termination, truncation, and returns the full 5-tuple.

**Verification:**
```powershell
python -c "from env.customer_support_env import CustomerSupportEnv; from env.config import ESCALATE; e = CustomerSupportEnv(); e.reset(seed=42); obs, r, t, tr, info = e.step(ESCALATE); print(t, tr, info.get('termination_reason')); assert t == True and tr == False"
```

---

### Task 3.6: Create env/customer_support_env.py — render and close

**Requirements:** FR-01 (AC-01.1 — render, close), design Section 4.2

**Implementation details:**
- Implement `render(self) -> None`:
  - If `self._state is None`: print "No active episode." and return.
  - Otherwise print a formatted summary:
    ```
    === Conversation State ===
    Problem:          {problem.name}
    Turn:             {turn_count}
    Info Collected:   {collected fields / total fields}
    Frustration:      {frustration_level:.2f}
    Last User Msg:    {MESSAGE_TYPE_NAMES[user_message_type]}
    Last Action:      {ACTION_NAMES.get(last_action, 'None')}
    ```
  - Use plain `print()` statements (render mode "human").
- Implement `close(self) -> None`:
  - Set `self._episode_active = False`.
  - Set `self._state = None`.
  - No external resources to release.

**Expected output:**
- `render()` and `close()` methods implemented.

**Verification:**
```powershell
python -c "from env.customer_support_env import CustomerSupportEnv; e = CustomerSupportEnv(); e.reset(seed=42); e.render(); e.close(); e.render()"
```

---

### Task 3.7: Register environment with Gymnasium

**Requirements:** FR-01 (AC-01.4), design Section 1.3

**Implementation details:**
- In `env/__init__.py`:
  - Import `gymnasium` and `CustomerSupportEnv` from `env.customer_support_env`.
  - Register the environment:
    ```python
    from gymnasium.envs.registration import register
    register(id="CustomerSupport-v0", entry_point="env.customer_support_env:CustomerSupportEnv")
    ```
  - Export `CustomerSupportEnv` in `__all__`.
- Verify registration by creating the env via `gymnasium.make("CustomerSupport-v0")`.

**Expected output:**
- `env/__init__.py` exports and registers the environment.

**Verification:**
```powershell
python -c "import env; import gymnasium; e = gymnasium.make('CustomerSupport-v0'); print(type(e.unwrapped))"
```

---

### Task 3.8: Verify full environment lifecycle manually

**Requirements:** FR-01, FR-02, FR-03, FR-04, FR-05, FR-06, FR-07 (integration check)

**Implementation details:**
- Run a manual full-episode script:
  1. Create env with `max_turns=15`.
  2. `reset(seed=42)`.
  3. Get the problem type from info.
  4. Manually step through: ask all required fields (using `FIELD_TO_ACTION`), diagnose, propose solution.
  5. Print each step's reward, terminated, truncated, termination_reason.
  6. Verify final step has `terminated=True`, `termination_reason="resolved"`.
  7. Verify `observation_space.contains(obs)` is True at every step.
  8. Verify calling `step()` after termination raises `RuntimeError`.

**Expected output:**
- Console output showing the full episode with correct resolution.

**Verification:**
```powershell
python -c "
from env.customer_support_env import CustomerSupportEnv
from env.config import FIELD_TO_ACTION, DIAGNOSE, PROPOSE_SOLUTION, ESCALATE
e = CustomerSupportEnv()
obs, info = e.reset(seed=42)
print('Problem:', info['problem_type'])
total_reward = 0.0
for field in e._state.problem.required_fields:
    action = FIELD_TO_ACTION[field]
    obs, r, t, tr, i = e.step(action)
    total_reward += r
    print(f'Ask {field}: reward={r}, info_collected={obs[\"info_collected\"]}')
obs, r, t, tr, i = e.step(DIAGNOSE)
total_reward += r
print(f'Diagnose: reward={r}')
obs, r, t, tr, i = e.step(PROPOSE_SOLUTION)
total_reward += r
print(f'Propose: reward={r}, terminated={t}, truncated={tr}, reason={i.get(\"termination_reason\")}')
print(f'Total reward: {total_reward}')
print(f'Obs in space: {e.observation_space.contains(obs)}')
"
```

---

## Wave 4: Baseline Rule-Based Agent

**Goal:** Implement the deterministic rule-based agent that serves as a performance benchmark.

**Depends on:** Wave 3 (environment fully functional)

---

### Task 4.1: Create agents/rule_based_agent.py — class skeleton and validation

**Requirements:** FR-08 (AC-08.1, AC-08.5), FR-12 (AC-12.5), design Section 10

**Implementation details:**
- Create `agents/rule_based_agent.py`.
- Import constants from `env/config`: action constants, `FRUSTRATION_HIGH_THRESHOLD`, `PROBLEM_TYPES`, `FIELD_TO_ACTION`, `NUM_ACTIONS`.
- Define `class RuleBasedAgent`:
  - `__init__(self)`: initialize internal state:
    - `self.last_action: int | None = None`
    - `self.consecutive_count: int = 0`
    - `self.has_diagnosed: bool = False`
    - `self.has_proposed: bool = False`
  - `reset(self) -> None`: clear all internal state back to initial values.
  - `act(self, observation: dict) -> int`:
    - **Validation (AC-12.5):** check observation has all required keys: `"user_message_type"`, `"turn_count"`, `"problem_type"`, `"info_collected"`, `"frustration_level"`. If any missing, raise `KeyError` with descriptive message: `f"Observation missing required key: '{key}'"`.
    - The actual policy logic is implemented in Task 4.2.

**Expected output:**
- `agents/rule_based_agent.py` with class skeleton, validation, and `reset()`.

**Verification:**
```powershell
python -c "from agents.rule_based_agent import RuleBasedAgent; a = RuleBasedAgent(); print(a.act({'user_message_type': 0, 'turn_count': 0, 'problem_type': 0, 'info_collected': [0,0,0,0], 'frustration_level': 0.0}))"
```
(Will return a default action — policy logic added in Task 4.2.)

---

### Task 4.2: Implement agent priority cascade policy

**Requirements:** FR-08 (AC-08.2.1 through AC-08.2.5), design Section 10.1

**Implementation details:**
- Implement the priority cascade in `act()` (after validation):
  1. **Frustration check (AC-08.2.1):** if `float(observation["frustration_level"]) > FRUSTRATION_HIGH_THRESHOLD`: return `APOLOGIZE`.
  2. **Info gathering (AC-08.2.2):**
     - Get the problem: `problem = PROBLEM_TYPES[int(observation["problem_type"])]`.
     - Convert `info_collected` observation (NumPy array) to a dict: `{field: bool(observation["info_collected"][i]) for i, field in enumerate(problem.required_fields)}`.
     - Find uncollected fields: `[f for f in problem.required_fields if not collected[f]]`.
     - If uncollected fields exist: get the first one, look up its action via `FIELD_TO_ACTION[field]`, return that action.
  3. **Diagnose (AC-08.2.3):** if all info collected and `not self.has_diagnosed`: return `DIAGNOSE`. Set `self.has_diagnosed = True` after the action is selected (or detect from observation — see note below).
  4. **Propose solution (AC-08.2.4):** if `self.has_diagnosed and not self.has_proposed`: return `PROPOSE_SOLUTION`. Set `self.has_proposed = True`.
  5. **Escalate after rejection (AC-08.2.5):** if `self.has_proposed` and the solution was rejected (detect via `observation["user_message_type"] == MSG_SOLUTION_REJECTED`): return `ESCALATE`.
  6. **Fallback:** return `ESCALATE`.

**Design note on tracking `has_diagnosed` and `has_proposed`:** The agent tracks these internally because they are not in the observation. After returning `DIAGNOSE`, set `self.has_diagnosed = True`. After returning `PROPOSE_SOLUTION`, set `self.has_proposed = True`. On `reset()`, these are cleared.

**Expected output:**
- `act()` implements the full priority cascade.

**Verification:**
```powershell
python -c "
from agents.rule_based_agent import RuleBasedAgent
from env.config import ASK_ACCOUNT_INFO, DIAGNOSE, PROPOSE_SOLUTION, APOLOGIZE
a = RuleBasedAgent()
# Test info gathering
obs = {'user_message_type': 0, 'turn_count': 0, 'problem_type': 0, 'info_collected': [0,0,0,0], 'frustration_level': 0.0}
action = a.act(obs)
print('First action (should be ASK_*):', action)
# Test frustration -> apologize
obs2 = {'user_message_type': 0, 'turn_count': 1, 'problem_type': 0, 'info_collected': [0,0,0,0], 'frustration_level': 0.8}
print('Frustrated action (should be APOLOGIZE=6):', a.act(obs2))
"
```

---

### Task 4.3: Implement agent anti-repetition logic

**Requirements:** FR-08 (AC-08.3), design Section 10.2

**Implementation details:**
- Before returning the action in `act()`, add anti-repetition check:
  - If `action == self.last_action and action != APOLOGIZE and self.consecutive_count >= 2`: override `action = ESCALATE`.
- After determining the final action, update tracking:
  - If `action == self.last_action`: `self.consecutive_count += 1`.
  - Else: `self.last_action = action`, `self.consecutive_count = 1`.
- This check runs after the priority cascade determines the action but before it is returned. The tracking update happens after the check.
- `APOLOGIZE` is exempt — it can be repeated indefinitely (AC-08.3).

**Expected output:**
- Agent avoids repeating non-apologize actions more than twice.

**Verification:**
```powershell
python -c "
from agents.rule_based_agent import RuleBasedAgent
from env.config import ASK_ACCOUNT_INFO, ESCALATE
a = RuleBasedAgent()
obs = {'user_message_type': 0, 'turn_count': 0, 'problem_type': 0, 'info_collected': [0,0,0,0], 'frustration_level': 0.0}
a1 = a.act(obs); a2 = a.act(obs); a3 = a.act(obs)
print('Three asks:', a1, a2, a3)
print('Third should escalate (5):', a3 == ESCALATE)
"
```

---

### Task 4.4: Wire up agents/__init__.py and verify agent against environment

**Requirements:** FR-08 (AC-08.4), NFR-02 (AC-NFR-02.3), design Section 10.4

**Implementation details:**
- In `agents/__init__.py`:
  - Import and export `RuleBasedAgent`.
  - Set `__all__ = ["RuleBasedAgent"]`.
- Run a full episode with the agent against the environment:
  1. Create `CustomerSupportEnv` and `RuleBasedAgent`.
  2. `env.reset(seed=42)`, `agent.reset()`.
  3. Loop: `action = agent.act(obs)`, `obs, r, t, tr, info = env.step(action)`, accumulate reward.
  4. Continue until `t or tr`.
  5. Print total reward, turns, termination reason.
  6. Run twice with same seed — verify identical results (determinism, AC-08.4).

**Expected output:**
- `agents/__init__.py` exports `RuleBasedAgent`.
- Console output showing a full episode and determinism verification.

**Verification:**
```powershell
python -c "
from env.customer_support_env import CustomerSupportEnv
from agents.rule_based_agent import RuleBasedAgent

def run_episode(seed):
    e = CustomerSupportEnv()
    a = RuleBasedAgent()
    obs, info = e.reset(seed=seed)
    a.reset()
    total = 0.0
    turns = 0
    while True:
        action = a.act(obs)
        obs, r, t, tr, info = e.step(action)
        total += r
        turns += 1
        if t or tr:
            break
    return total, turns, info.get('termination_reason')

r1 = run_episode(42)
r2 = run_episode(42)
print('Run 1:', r1)
print('Run 2:', r2)
print('Deterministic:', r1 == r2)
"
```

---

## Wave 5: Evaluation and Metrics

**Goal:** Build the evaluation harness that runs 100 episodes, computes metrics, and outputs a JSON report.

**Depends on:** Wave 4 (agent and environment both working)

---

### Task 5.1: Create evaluation/run_evaluation.py — input validation and run_single_episode

**Requirements:** FR-09 (AC-09.1, AC-09.2, AC-09.3, AC-09.4, AC-09.5), FR-12 (AC-12.6, AC-12.7), design Section 11

**Implementation details:**
- Create `evaluation/run_evaluation.py`.
- Import from `env.customer_support_env`: `CustomerSupportEnv`.
- Import from `agents.rule_based_agent`: `RuleBasedAgent`.
- Import from `env.config`: `DEFAULT_NUM_EPISODES`, `DEFAULT_MAX_TURNS`, `DEFAULT_SEED`.
- Implement `run_single_episode(env: CustomerSupportEnv, agent: RuleBasedAgent, seed: int) -> dict`:
  1. `agent.reset()`.
  2. `obs, info = env.reset(seed=seed)`.
  3. `episode_reward = 0.0`.
  4. `terminated, truncated = False, False`.
  5. Loop while `not terminated and not truncated`:
     - `action = agent.act(obs)`.
     - `obs, reward, terminated, truncated, info = env.step(action)`.
     - `episode_reward += reward`.
  6. Return dict:
     ```python
     {
         "total_reward": episode_reward,
         "num_turns": env._state.turn_count,
         "termination_reason": info.get("termination_reason", "unknown"),
         "success": info.get("termination_reason") == "resolved",
     }
     ```
  - Handle the case where the episode terminates immediately (1-turn escalation) — the while loop body runs once and exits (AC-09.5).

**Expected output:**
- `evaluation/run_evaluation.py` with `run_single_episode()`.

**Verification:**
```powershell
python -c "
from evaluation.run_evaluation import run_single_episode
from env.customer_support_env import CustomerSupportEnv
from agents.rule_based_agent import RuleBasedAgent
e = CustomerSupportEnv()
a = RuleBasedAgent()
result = run_single_episode(e, a, 42)
print(result)
"
```

---

### Task 5.2: Create evaluation/run_evaluation.py — run_evaluation orchestrator

**Requirements:** FR-09 (AC-09.1, AC-09.6), design Section 11.1

**Implementation details:**
- Implement `run_evaluation(num_episodes: int = DEFAULT_NUM_EPISODES, max_turns: int = DEFAULT_MAX_TURNS, seed: int | None = DEFAULT_SEED, agent: RuleBasedAgent | None = None, output_path: str = "results/evaluation_results.json") -> dict`:
  - **Validate inputs (AC-12.6, AC-12.7):**
    - If `not isinstance(num_episodes, int) or num_episodes <= 0`: raise `ValueError("num_episodes must be a positive integer, got {num_episodes}")`.
    - If `seed is not None and not isinstance(seed, int)`: raise `ValueError("seed must be an integer or None, got {type(seed).__name__}")`.
  - Create environment: `env = CustomerSupportEnv(max_turns=max_turns)`.
  - Create agent if None: `agent = agent or RuleBasedAgent()`.
  - Initialize `episode_records = []`.
  - For `episode_id in range(num_episodes)`:
    - `episode_seed = seed + episode_id` (additive seeding, design Section 11.2).
    - **Error handling (AC-09.6):** wrap in try/except `Exception as e`:
      - On success: `record = run_single_episode(env, agent, episode_seed)`.
      - On exception: `record = {"total_reward": 0.0, "num_turns": 0, "termination_reason": "error", "success": False, "error": str(e)}`, log warning.
    - Add `"episode_id": episode_id` to record.
    - Append to `episode_records`.
  - Call `metrics = compute_metrics(episode_records)` (implemented in Task 5.3).
  - Build config dict: `{"num_episodes": num_episodes, "max_turns": max_turns, "seed": seed, "agent_type": type(agent).__name__}`.
  - Call `write_report(metrics, episode_records, config, output_path)` (Task 5.4).
  - Call `print_summary(metrics)` (Task 5.5).
  - Return `{"config": config, "metrics": metrics, "episodes": episode_records}`.

**Expected output:**
- `run_evaluation()` orchestrator function.

**Verification:**
```powershell
python -c "from evaluation.run_evaluation import run_evaluation; r = run_evaluation(num_episodes=5, seed=42); print(r['config']); print(len(r['episodes']))"
```

---

### Task 5.3: Create evaluation/run_evaluation.py — compute_metrics

**Requirements:** FR-10 (AC-10.1, AC-10.2, AC-10.3, AC-10.4), design Section 12

**Implementation details:**
- Import `numpy as np`.
- Implement `compute_metrics(episode_records: list[dict]) -> dict`:
  - Extract arrays using NumPy:
    - `rewards = np.array([r["total_reward"] for r in episode_records])`
    - `turns = np.array([r["num_turns"] for r in episode_records])`
    - `reasons = np.array([r["termination_reason"] for r in episode_records])`
  - Compute metrics (all cast to native Python types for JSON):
    ```python
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_rate": float(np.mean(reasons == "resolved")),
        "mean_turns": float(np.mean(turns)),
        "std_turns": float(np.std(turns)),
        "escalation_rate": float(np.mean(reasons == "escalated")),
        "truncation_rate": float(np.mean(reasons == "truncated")),
        "num_episodes": len(episode_records),
    }
    ```
  - Use `ddof=0` (population std) per design Section 12.3.
  - Return `metrics`.

**Expected output:**
- `compute_metrics()` function returning a dict with all 10 metric keys.

**Verification:**
```powershell
python -c "
from evaluation.run_evaluation import compute_metrics
records = [
    {'total_reward': 10.0, 'num_turns': 5, 'termination_reason': 'resolved', 'success': True},
    {'total_reward': 2.0, 'num_turns': 1, 'termination_reason': 'escalated', 'success': False},
]
m = compute_metrics(records)
print(m)
print('Rates sum:', m['success_rate'] + m['escalation_rate'] + m['truncation_rate'])
"
```

---

### Task 5.4: Create evaluation/run_evaluation.py — write_report (JSON)

**Requirements:** FR-11 (AC-11.1, AC-11.2, AC-11.3, AC-11.4, AC-11.5, AC-11.6), design Section 13

**Implementation details:**
- Import `json` and `os`.
- Implement `write_report(metrics: dict, episode_records: list[dict], config: dict, output_path: str) -> None`:
  - Build the report dict with key order `config`, `metrics`, `episodes` (design Section 13.2):
    ```python
    report = {
        "config": config,
        "metrics": metrics,
        "episodes": episode_records,
    }
    ```
  - Create the results directory if it doesn't exist (AC-11.6):
    - `os.makedirs(os.path.dirname(output_path), exist_ok=True)` — handle the case where `output_path` has no directory component (use `os.path.split` and default to current dir if empty).
  - Write JSON with `json.dump(report, f, indent=2)` (AC-11.5).
  - Ensure all values are JSON-serializable (floats and ints, not numpy types — already handled in `compute_metrics`).

**Expected output:**
- `write_report()` function that writes valid JSON to `results/evaluation_results.json`.

**Verification:**
```powershell
python -c "
from evaluation.run_evaluation import write_report
import json
write_report({'mean_reward': 5.0}, [{'episode_id': 0, 'total_reward': 5.0, 'num_turns': 3, 'termination_reason': 'resolved', 'success': True}], {'num_episodes': 1, 'max_turns': 15, 'seed': 42, 'agent_type': 'RuleBasedAgent'}, 'results/evaluation_results.json')
with open('results/evaluation_results.json') as f:
    data = json.load(f)
print(list(data.keys()))
print(data['config'])
"
```

---

### Task 5.5: Create evaluation/run_evaluation.py — print_summary and CLI

**Requirements:** FR-11 (AC-11.7), FR-09 (callable from CLI), design Section 13.3

**Implementation details:**
- Implement `print_summary(metrics: dict) -> None`:
  - Print a formatted table to stdout (design Section 13.3):
    ```
    === Evaluation Results ===
    Episodes:           {num_episodes}
    Mean Reward:        {mean_reward:.2f}
    Std Reward:         {std_reward:.2f}
    Min Reward:         {min_reward:.2f}
    Max Reward:         {max_reward:.2f}
    Success Rate:       {success_rate * 100:.1f}%
    Escalation Rate:    {escalation_rate * 100:.1f}%
    Truncation Rate:    {truncation_rate * 100:.1f}%
    Mean Turns:         {mean_turns:.1f}
    Std Turns:          {std_turns:.1f}
    ```
  - Print path to saved report at the end.
- Implement `main()`:
  - Use `argparse` to parse CLI args:
    - `--episodes` (int, default `DEFAULT_NUM_EPISODES`)
    - `--max-turns` (int, default `DEFAULT_MAX_TURNS`)
    - `--seed` (int, default `DEFAULT_SEED`)
    - `--output` (str, default `"results/evaluation_results.json"`)
  - Call `run_evaluation(num_episodes=args.episodes, max_turns=args.max_turns, seed=args.seed, output_path=args.output)`.
- Add `if __name__ == "__main__": main()` at the bottom.

**Expected output:**
- `print_summary()` and `main()` functions.
- CLI entry point working.

**Verification:**
```powershell
python -m evaluation.run_evaluation --episodes 10 --seed 42
```
Expect: console output with metrics table and `results/evaluation_results.json` created.

---

## Wave 6: Automated Testing

**Goal:** Write the full Pytest suite covering all components and edge cases.

**Depends on:** Wave 5 (all implementation complete and verified manually)

---

### Task 6.1: Create tests/conftest.py — shared fixtures

**Requirements:** NFR-03 (AC-NFR-03.3), design Section 15.2

**Implementation details:**
- Create `tests/conftest.py`.
- Import `pytest`, `CustomerSupportEnv`, `RuleBasedAgent`.
- Define fixtures:
  - `@pytest.fixture` `env()`: returns `CustomerSupportEnv(max_turns=15)` — fresh env per test.
  - `@pytest.fixture` `seeded_env(env)`: calls `env.reset(seed=42)`, returns `env` — deterministic state.
  - `@pytest.fixture` `agent()`: returns `RuleBasedAgent()` — fresh agent per test.
  - `@pytest.fixture` `small_env()`: returns `CustomerSupportEnv(max_turns=3)` — for truncation tests.
  - `@pytest.fixture(scope="session")` `fixed_seed()`: returns `42`.

**Expected output:**
- `tests/conftest.py` with all 5 fixtures.

**Verification:**
```powershell
python -m pytest tests/conftest.py --co
```
(Should list available fixtures with no errors.)

---

### Task 6.2: Create tests/test_env.py — API compliance tests

**Requirements:** FR-01 (AC-01.1, AC-01.2, AC-01.3), FR-03 (AC-03.1, AC-03.7), FR-04 (AC-04.1, AC-04.3)

**Implementation details:**
- Create `tests/test_env.py`.
- Import `gymnasium`, `gymnasium.spaces as spaces`, pytest, and environment/config.
- Write tests:
  - `test_env_subclass(env)`: `assert isinstance(env, gymnasium.Env)`.
  - `test_reset_returns_tuple(seeded_env)`: result = `env.reset(seed=42)`; `assert isinstance(result, tuple)`; `assert len(result) == 2`.
  - `test_step_returns_5tuple(seeded_env)`: `obs, r, t, tr, info = env.step(0)`; `assert isinstance((obs, r, t, tr, info), tuple)`; `assert len((obs, r, t, tr, info)) == 5`.
  - `test_action_space_discrete(env)`: `assert isinstance(env.action_space, spaces.Discrete)`.
  - `test_observation_space_dict(env)`: `assert isinstance(env.observation_space, spaces.Dict)`.
  - `test_observation_contains_all_fields(seeded_env)`: obs = `env.reset(seed=42)[0]`; check all 5 keys present: `"user_message_type"`, `"turn_count"`, `"problem_type"`, `"info_collected"`, `"frustration_level"`.
  - `test_observation_in_space(seeded_env)`: after reset and after each step, `assert env.observation_space.contains(obs)`.
  - `test_action_names_mapping()`: from `env.config import ACTION_NAMES`; assert `ACTION_NAMES[0] == "ASK_ACCOUNT_INFO"` etc. for all 7.
  - `test_render_output(seeded_env, capsys)`: `env.render()`; `captured = capsys.readouterr()`; `assert len(captured.out) > 0`.

**Expected output:**
- `tests/test_env.py` with ~9 API compliance tests.

**Verification:**
```powershell
python -m pytest tests/test_env.py::test_env_subclass tests/test_env.py::test_reset_returns_tuple tests/test_env.py::test_step_returns_5tuple tests/test_env.py::test_action_space_discrete tests/test_env.py::test_observation_space_dict tests/test_env.py::test_observation_contains_all_fields tests/test_env.py::test_observation_in_space tests/test_env.py::test_action_names_mapping tests/test_env.py::test_render_output -v
```

---

### Task 6.3: Create tests/test_env.py — reward tests

**Requirements:** FR-05 (AC-05.1 through AC-05.10)

**Implementation details:**
- Add reward tests to `tests/test_env.py`:
  - `test_reward_turn_penalty(seeded_env)`: step with any action; assert reward includes `-0.5` base. Use `pytest.approx`.
  - `test_reward_info_gathered(seeded_env)`: step with an ASK_* action that gathers a field; assert `+1.0` bonus is included (reward should be `+0.5` = `1.0 - 0.5`).
  - `test_reward_resolution(seeded_env)`: run a full episode (gather all info, diagnose, propose); assert final step reward is `+10.0 - 0.5 = +9.5`.
  - `test_reward_wrong_solution(seeded_env)`: propose solution before diagnosing or with incomplete info; assert `-3.0 - 0.5 = -3.5`.
  - `test_reward_repeated_action(seeded_env)`: step same ASK action twice; second identical step gets `-2.0` penalty (reward = `-0.5 - 2.0 = -2.5`).
  - `test_reward_escalate(seeded_env)`: step with ESCALATE; assert `+2.0 - 0.5 = +1.5`.
  - `test_reward_truncate(small_env)`: run until turn limit (max_turns=3); assert final reward includes `-5.0 - 0.5 = -5.5`.
  - `test_reward_apologize_high(small_env)`: create scenario where frustration > 0.5, step with APOLOGIZE; assert `+0.5 - 0.5 = 0.0`.
  - `test_reward_apologize_low(seeded_env)`: step APOLOGIZE at frustration 0.0; assert `0.0 - 0.5 = -0.5`.
  - Use `pytest.approx(expected, abs=0.01)` for all float comparisons (NFR-03.7).

**Expected output:**
- 9 reward tests added to `tests/test_env.py`.

**Verification:**
```powershell
python -m pytest tests/test_env.py -k "reward" -v
```

---

### Task 6.4: Create tests/test_env.py — termination and error handling tests

**Requirements:** FR-06 (AC-06.1 through AC-06.6), FR-07 (AC-07.2, AC-07.3, AC-07.4, AC-07.5), FR-12 (AC-12.1, AC-12.2, AC-12.3, AC-12.4, AC-12.8)

**Implementation details:**
- Add tests to `tests/test_env.py`:
  - `test_resolved_termination(seeded_env)`: run full episode to resolution; assert `terminated == True`, `truncated == False`, `info["termination_reason"] == "resolved"`.
  - `test_escalate_termination(seeded_env)`: step with ESCALATE; assert `terminated == True`, `truncated == False`, `info["termination_reason"] == "escalated"`.
  - `test_truncated_at_max_turns(small_env)`: run until turn limit (max_turns=3); assert `truncated == True`, `terminated == False`, `info["termination_reason"] == "truncated"`.
  - `test_termination_reason_in_info(seeded_env)`: after any terminal step, assert `"termination_reason" in info`.
  - `test_step_after_termination_raises(seeded_env)`: escalate (terminal), then call `step(0)`; assert `pytest.raises(RuntimeError)`.
  - `test_invalid_action_raises(seeded_env)`: `env.step(99)`; assert `pytest.raises(ValueError)`.
  - `test_invalid_action_negative(seeded_env)`: `env.step(-1)`; assert `pytest.raises(ValueError)`.
  - `test_invalid_max_turns_zero()`: `CustomerSupportEnv(max_turns=0)`; assert `pytest.raises(ValueError)`.
  - `test_invalid_max_turns_negative()`: `CustomerSupportEnv(max_turns=-5)`; assert `pytest.raises(ValueError)`.
  - `test_reset_mid_episode(seeded_env)`: step once, then `env.reset(seed=42)`; assert no error, returns valid tuple.
  - `test_no_state_mutation_on_error(seeded_env)`: record turn_count, call `env.step(99)` (invalid), catch ValueError, assert turn_count unchanged.

**Expected output:**
- 11 termination/error tests added to `tests/test_env.py`.

**Verification:**
```powershell
python -m pytest tests/test_env.py -k "termination or invalid or truncate or reset_mid or no_state or step_after" -v
```

---

### Task 6.5: Create tests/test_agent.py — agent policy tests

**Requirements:** FR-08 (AC-08.1 through AC-08.5), FR-12 (AC-12.5), design Section 15.3

**Implementation details:**
- Create `tests/test_agent.py`.
- Import `pytest`, `RuleBasedAgent`, and config constants.
- Write tests:
  - `test_act_returns_valid_action(agent)`: construct a valid obs dict, call `agent.act(obs)`; assert result is `int` and `0 <= result <= 6`.
  - `test_apologize_when_frustrated(agent)`: obs with `frustration_level=0.8`; assert `agent.act(obs) == APOLOGIZE` (6).
  - `test_gathers_info_first(agent)`: obs with `problem_type=0`, `info_collected=[0,0,0,0]`; assert result is an ASK_* action (0, 1, or 2).
  - `test_diagnoses_after_info(agent)`: obs with all info collected, `frustration_level=0.0`; need to track `has_diagnosed` — call act twice (first returns DIAGNOSE), then assert. Alternatively: manually set `agent.has_diagnosed = False`, construct obs with all info collected, assert result == `DIAGNOSE` (3).
  - `test_proposes_after_diagnosis(agent)`: set `agent.has_diagnosed = True`, `agent.has_proposed = False`, obs with all info collected; assert result == `PROPOSE_SOLUTION` (4).
  - `test_escalates_after_rejection(agent)`: set `agent.has_diagnosed = True`, `agent.has_proposed = True`, obs with `user_message_type=MSG_SOLUTION_REJECTED` (4); assert result == `ESCALATE` (5).
  - `test_no_excessive_repeats(agent)`: call `agent.act()` 3 times with same obs (info not collected); assert third call returns `ESCALATE` (5), not the same ASK action.
  - `test_deterministic(agent)`: same obs → same action, call twice, assert equal.
  - `test_missing_key_raises(agent)`: obs missing `"frustration_level"`; assert `pytest.raises(KeyError)`.
  - `test_agent_reset_clears_state(agent)`: set `agent.has_diagnosed = True`, `agent.last_action = 3`; call `agent.reset()`; assert `agent.has_diagnosed == False` and `agent.last_action is None`.

**Expected output:**
- `tests/test_agent.py` with 10 tests.

**Verification:**
```powershell
python -m pytest tests/test_agent.py -v
```

---

### Task 6.6: Create tests/test_evaluation.py — evaluation and metrics tests

**Requirements:** FR-09 (AC-09.1, AC-09.5), FR-10 (AC-10.1, AC-10.2, AC-10.3), FR-11 (AC-11.2, AC-11.3, AC-11.4, AC-11.5, AC-11.6), FR-12 (AC-12.6, AC-12.7), NFR-02 (AC-NFR-02.2), design Section 15.3

**Implementation details:**
- Create `tests/test_evaluation.py`.
- Import `pytest`, `json`, `os`, and `run_evaluation`, `compute_metrics` from `evaluation.run_evaluation`.
- Write tests:
  - `test_run_evaluation_returns_dict()`: result = `run_evaluation(num_episodes=5, seed=42, output_path="results/test_results.json")`; assert `isinstance(result, dict)`; assert `"metrics" in result` and `"episodes" in result` and `"config" in result`.
  - `test_num_episodes_correct()`: result = `run_evaluation(num_episodes=5, seed=42, output_path="results/test_results.json")`; assert `len(result["episodes"]) == 5`.
  - `test_metrics_all_present()`: result = `run_evaluation(num_episodes=5, seed=42, output_path="results/test_results.json")`; assert all 10 metric keys in `result["metrics"]`.
  - `test_success_rate_range()`: assert `0.0 <= result["metrics"]["success_rate"] <= 1.0`.
  - `test_rates_sum_to_one()`: m = `result["metrics"]`; assert `pytest.approx(m["success_rate"] + m["escalation_rate"] + m["truncation_rate"], abs=0.01) == 1.0`.
  - `test_reproducible()`: r1 = `run_evaluation(num_episodes=5, seed=42, output_path="results/test1.json")`; r2 = `run_evaluation(num_episodes=5, seed=42, output_path="results/test2.json")`; assert `r1["metrics"] == r2["metrics"]`.
  - `test_json_valid(tmp_path)`: output = `tmp_path / "results.json"`; `run_evaluation(num_episodes=5, seed=42, output_path=str(output))`; `data = json.load(open(output))`; assert `"config" in data and "metrics" in data and "episodes" in data`.
  - `test_json_has_config()`: from the loaded JSON, assert config has keys `"num_episodes"`, `"max_turns"`, `"seed"`, `"agent_type"`.
  - `test_json_has_episodes()`: from the loaded JSON, assert `len(data["episodes"]) == 5`; check first episode has keys `"episode_id"`, `"total_reward"`, `"num_turns"`, `"termination_reason"`, `"success"`.
  - `test_results_dir_created(tmp_path)`: output = `tmp_path / "subdir" / "results.json"`; `run_evaluation(num_episodes=3, seed=42, output_path=str(output))`; assert output.exists().
  - `test_invalid_num_episodes()`: `pytest.raises(ValueError)` when `run_evaluation(num_episodes=0)`.
  - `test_invalid_num_episodes_negative()`: `pytest.raises(ValueError)` when `run_evaluation(num_episodes=-1)`.
  - `test_invalid_seed()`: `pytest.raises(ValueError)` when `run_evaluation(num_episodes=5, seed="abc")`.
  - `test_small_num_episodes()`: `run_evaluation(num_episodes=1, seed=42, output_path="results/test_small.json")`; assert no crash, `len(result["episodes"]) == 1`.
  - Use `tmp_path` fixture for JSON file tests to avoid polluting `results/` during test runs.

**Expected output:**
- `tests/test_evaluation.py` with ~14 tests.

**Verification:**
```powershell
python -m pytest tests/test_evaluation.py -v
```

---

### Task 6.7: Run the full test suite

**Requirements:** NFR-03 (AC-NFR-03.1, AC-NFR-03.6, AC-NFR-03.7)

**Implementation details:**
- Run the complete test suite and verify:
  1. All tests pass (exit code 0).
  2. Total count matches expected (~44 tests across 3 files).
  3. Execution time is under 30 seconds.
  4. No warnings about bare excepts, mutable defaults, or import issues.
- If any tests fail, debug and fix the implementation (not the tests — tests are the spec).

**Expected output:**
- Full test suite passes with green output.

**Verification:**
```powershell
python -m pytest tests/ -v --tb=short
python -m pytest tests/ --durations=10
```

---

## Wave 7: Demo and Documentation

**Goal:** Run the full 100-episode evaluation, verify output, and write the README.

**Depends on:** Wave 6 (all tests passing)

---

### Task 7.1: Run full 100-episode evaluation

**Requirements:** FR-09 (AC-09.1), FR-11 (AC-11.1, AC-11.7)

**Implementation details:**
- Run the evaluation harness with default parameters (100 episodes, seed=42):
  ```powershell
  python -m evaluation.run_evaluation
  ```
- Verify:
  1. Console output shows the metrics summary table.
  2. `results/evaluation_results.json` is created.
  3. JSON file has 100 episode entries.
  4. Metrics are reasonable (success_rate > 0, no NaN values).

**Expected output:**
- Console metrics table.
- `results/evaluation_results.json` with 100 episodes.

**Verification:**
```powershell
python -m evaluation.run_evaluation
python -c "import json; data = json.load(open('results/evaluation_results.json')); print('Episodes:', len(data['episodes'])); print('Metrics:', data['metrics'])"
```

---

### Task 7.2: Verify reproducibility of full evaluation

**Requirements:** NFR-02 (AC-NFR-02.2, AC-NFR-02.4)

**Implementation details:**
- Run the evaluation twice with the same seed and compare the JSON outputs:
  1. Run `python -m evaluation.run_evaluation --output results/run1.json`
  2. Run `python -m evaluation.run_evaluation --output results/run2.json`
  3. Compare `run1.json` and `run2.json` — they should be byte-identical (or at minimum, metrics and episodes identical).

**Expected output:**
- Two JSON files that are identical.

**Verification:**
```powershell
python -m evaluation.run_evaluation --output results/run1.json
python -m evaluation.run_evaluation --output results/run2.json
python -c "
import json
r1 = json.load(open('results/run1.json'))
r2 = json.load(open('results/run2.json'))
assert r1['metrics'] == r2['metrics'], 'Metrics differ!'
assert r1['episodes'] == r2['episodes'], 'Episodes differ!'
print('Reproducibility verified: identical results')
"
```

---

### Task 7.3: Create README.md

**Requirements:** NFR-01 (AC-NFR-01.1), documentation best practices

**Implementation details:**
- Create `README.md` at the project root with the following sections:
  - **Project title and description** — one-paragraph summary of what the project does.
  - **Technology stack** — Python 3.14, Gymnasium, NumPy, Pytest.
  - **Project structure** — the directory layout (from design.md Section 1.3).
  - **Setup** — installation instructions:
    ```bash
    pip install -r requirements-dev.txt
    ```
  - **Running the evaluation** — command and expected output:
    ```bash
    python -m evaluation.run_evaluation --episodes 100 --seed 42
    ```
  - **Running tests** — command:
    ```bash
    python -m pytest tests/ -v
    ```
  - **Architecture overview** — brief summary referencing the design document.
  - **Key design decisions** — 3-4 bullet points (numeric observations, separated terminated/truncated, centralized config, deterministic baseline).
  - **Evaluation metrics explanation** — table of all metrics with descriptions.

**Expected output:**
- `README.md` at project root.

**Verification:**
```powershell
python -c "content = open('README.md').read(); assert 'multi-turn-agent-gym' in content; assert 'pytest' in content; assert 'gymnasium' in content; print('README valid')"
```

---

### Task 7.4: Clean up and final verification

**Requirements:** All functional and non-functional requirements

**Implementation details:**
- Remove any temporary test output files (e.g., `results/run1.json`, `results/run2.json`, `results/test_results.json`).
- Keep `results/evaluation_results.json` as the canonical output.
- Run a final comprehensive verification:
  1. Full test suite passes.
  2. Full evaluation runs and produces JSON.
  3. `requirements.txt` and `requirements-dev.txt` exist and are valid.
  4. All Python files have no syntax errors.
  5. No `import *` statements anywhere (NFR-05.2).
  6. No bare `except:` clauses (NFR-05.1).

**Expected output:**
- Clean project with all verification passing.

**Verification:**
```powershell
# Full test suite
python -m pytest tests/ -v --tb=short

# Full evaluation
python -m evaluation.run_evaluation

# Check for import * violations
python -c "import ast, os; [print(f'IMPORT * in {f}') for f in os.listdir('.') if f.endswith('.py') for node in ast.walk(ast.parse(open(f).read())) if isinstance(node, ast.ImportFrom) and node.names[0].name == '*']"

# Verify all files parse
python -c "import py_compile, os; [py_compile.compile(os.path.join(root, f), doraise=True) for root, _, files in os.walk('.') for f in files if f.endswith('.py')]; print('All Python files compile successfully')"
```

---

## Task Summary

| Wave | Tasks | Requirements Covered | Key Output |
|---|---|---|---|
| 1: Foundation | 1.1–1.6 | NFR-04, FR-04, NFR-01 | Dependencies installed, directory structure, `config.py` |
| 2: Scenarios | 2.1–2.6 | FR-02, NFR-02 | `conversation.py` with problem types, state, and user state machine |
| 3: Environment | 3.1–3.8 | FR-01, FR-03, FR-04, FR-05, FR-06, FR-07, FR-12 | `CustomerSupportEnv` fully functional |
| 4: Agent | 4.1–4.4 | FR-08, NFR-02 | `RuleBasedAgent` deterministic and verified |
| 5: Evaluation | 5.1–5.5 | FR-09, FR-10, FR-11, FR-12 | Evaluation harness with metrics and JSON output |
| 6: Testing | 6.1–6.7 | NFR-03, all FRs | Full Pytest suite passing (~44 tests) |
| 7: Demo & Docs | 7.1–7.4 | NFR-01, NFR-02, all | 100-episode evaluation, README, clean project |

**Total tasks:** 38
**Estimated test count:** ~44 tests across 3 test files
**Dependency rule:** Tasks within a wave are sequential. Waves are sequential. The only parallelizable tasks are within Wave 6 (test files are independent of each other), but they are listed sequentially for simplicity.
