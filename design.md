# Design Document — multi-turn-agent-gym

## 1. System Overview

### 1.1 Purpose

The system provides a custom Gymnasium environment that models a customer-support conversation as a multi-turn reinforcement learning problem. An agent interacts with a simulated user across a sequence of turns, selecting actions to gather information, diagnose, and resolve the user's problem. The environment provides structured observations, scalar rewards, and clear termination signals following Gymnasium conventions.

A rule-based baseline agent demonstrates a working policy. An evaluation harness runs the agent across 100 episodes and reports aggregate performance metrics. A Pytest suite validates every component.

### 1.2 Design Principles

| Principle | Rationale |
|---|---|
| **Gymnasium-native** | The environment follows Gymnasium's `reset()` 2-tuple and `step()` 5-tuple API exactly, ensuring compatibility with any Gymnasium-based RL tooling. |
| **Separated `terminated` and `truncated`** | Per Gymnasium's design philosophy, `terminated` means the episode reached a natural end (resolved or escalated), while `truncated` means it was cut short artificially (turn limit). These are never conflated, even though both end the episode. |
| **Numeric observation encoding** | Observations are pure numeric structures (integers, floats, binary vectors). This avoids NLP dependencies and keeps the environment lightweight and extensible for future learned agents. |
| **Centralized constants** | All reward values, action names, problem definitions, and message types live in a single configuration module (`env/config.py`). No magic numbers in logic code. |
| **Deterministic-by-default** | The rule-based agent is fully deterministic. The environment's only randomness is the RNG used for problem sampling, which is seeded via `reset(seed=...)`. |
| **Validation before mutation** | Every `step()` call validates the action before touching internal state. An invalid action leaves the environment unchanged. |

### 1.3 Module Layout

```
multi-turn-agent-gym/
├── requirements.txt
├── requirements-dev.txt
├── README.md
│
├── env/
│   ├── __init__.py            # Exports CustomerSupportEnv, registration
│   ├── config.py              # All constants: rewards, actions, problems, message types
│   ├── conversation.py        # Simulated user state machine, problem definitions
│   ├── spaces.py              # Observation & action space definitions
│   └── customer_support_env.py  # Gymnasium.Env subclass: reset, step, render, close
│
├── agents/
│   ├── __init__.py
│   └── rule_based_agent.py    # Deterministic baseline agent
│
├── evaluation/
│   ├── __init__.py
│   └── run_evaluation.py      # Evaluation harness + metrics + JSON report
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures (env, agent, seed)
│   ├── test_env.py            # Environment API, spaces, reward, termination
│   ├── test_agent.py          # Agent policy logic
│   └── test_evaluation.py     # Evaluation loop, metrics, JSON output
│
└── results/                   # Generated at runtime
    └── evaluation_results.json
```

**Design decision — `config.py` instead of spreading constants:** All reward values, action mappings, problem type definitions, and message type enums are centralized in `env/config.py`. This satisfies NFR-01.4 (no scattered magic numbers) and makes tuning the reward function or adding problem types a single-file change. The config module exports plain Python constants and dataclasses — no logic, no side effects.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        multi-turn-agent-gym                         │
│                                                                     │
│  ┌──────────────────┐     ┌──────────────────┐                     │
│  │  RuleBasedAgent   │     │  EvaluationHarness │                    │
│  │  (agents/)        │     │  (evaluation/)      │                    │
│  │                  │     │                    │                     │
│  │  act(obs)->int   │     │  run_episodes()    │                     │
│  └────────┬─────────┘     │  compute_metrics() │                     │
│           │               │  write_report()    │                     │
│           │               └────────┬───────────┘                     │
│           │                        │                                 │
│           │  action (int)          │ manages episodes                │
│           ▼                        ▼                                 │
│  ┌──────────────────────────────────────────┐                       │
│  │         CustomerSupportEnv                │                       │
│  │         (env/customer_support_env.py)     │                       │
│  │                                          │                       │
│  │  ┌─────────────┐  ┌──────────────────┐   │                       │
│  │  │  spaces.py  │  │  conversation.py  │   │                       │
│  │  │             │  │                  │   │                       │
│  │  │  Dict obs   │  │  ProblemRegistry  │   │                       │
│  │  │  Discrete   │  │  UserStateMachine │   │                       │
│  │  │  action     │  │                  │   │                       │
│  │  └─────────────┘  └──────────────────┘   │                       │
│  │                                          │                       │
│  │  ┌──────────────────────────────────┐   │                       │
│  │  │  config.py                        │   │                       │
│  │  │  REWARDS, ACTIONS, PROBLEMS,      │   │                       │
│  │  │  MESSAGE_TYPES, MAX_TURNS         │   │                       │
│  │  └──────────────────────────────────┘   │                       │
│  │                                          │                       │
│  │  reset() -> (obs, info)                  │                       │
│  │  step(action) -> (obs, reward,          │                       │
│  │                   terminated,            │                       │
│  │                   truncated, info)       │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  ┌──────────────────────────────────────────┐                       │
│  │  tests/                                  │                       │
│  │  test_env.py   test_agent.py             │                       │
│  │  test_evaluation.py   conftest.py        │                       │
│  └──────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram (Single Episode)

```
EvaluationHarness
    │
    │ 1. reset(seed=episode_seed)
    ▼
CustomerSupportEnv ───► samples problem type via RNG
    │                   initializes conversation state
    │
    │ 2. returns (observation, info)
    ▼
RuleBasedAgent
    │
    │ 3. act(observation) -> action (int)
    ▼
CustomerSupportEnv
    │
    │ 4. step(action)
    │    ├── validate action
    │    ├── interpret action via conversation.py
    │    ├── generate simulated user response
    │    ├── update conversation state
    │    ├── compute reward via config.py constants
    │    ├── check termination conditions
    │    └── build observation from current state
    │
    │ 5. returns (observation, reward, terminated, truncated, info)
    ▼
EvaluationHarness
    │
    │ 6. accumulate reward, check terminated/truncated
    │    if neither: go to step 2
    │    if either: record episode, go to next episode
    ▼
After all episodes: compute metrics, write JSON, print summary
```

---

## 3. Component Responsibilities

### 3.1 `env/config.py` — Configuration Constants

| Responsibility | Detail |
|---|---|
| Reward constants | All numeric reward values (`REWARD_RESOLVE = 10.0`, `REWARD_INFO = 1.0`, `REWARD_TURN_PENALTY = -0.5`, etc.) |
| Action definitions | Integer-to-name mapping, action count, and enum-like constants |
| Message types | Integer-to-name mapping for user message categories |
| Problem type definitions | Dataclass or dict describing each of the 5+ problem types |
| Default parameters | `DEFAULT_MAX_TURNS = 15`, `DEFAULT_SEED = 42`, `DEFAULT_NUM_EPISODES = 100` |
| Frustration thresholds | `FRUSTRATION_APOLOGY_THRESHOLD = 0.5`, `FRUSTRATION_HIGH_THRESHOLD = 0.7` |

**Design decision — dataclasses for problem types:** Each problem type is a frozen dataclass with fields: `id` (int), `name` (str), `required_fields` (list of field name strings), `resolution_action` (int, the action that solves it), and `info_request_actions` (list of ints, which ASK_* actions gather which fields). Frozen dataclasses provide immutability, type safety, and clean equality comparisons for tests. They are defined in `config.py` and imported wherever needed.

### 3.2 `env/spaces.py` — Space Definitions

| Responsibility | Detail |
|---|---|
| `build_action_space()` | Returns `gymnasium.spaces.Discrete(n_actions)` |
| `build_observation_space(max_turns, num_problems, max_info_fields)` | Returns `gymnasium.spaces.Dict` with all sub-spaces |
| `build_observation(state_dict)` | Converts internal state dict to a Gymnasium-compatible observation dict (NumPy arrays) |
| `validate_observation(obs, space)` | Utility for tests; wraps `space.contains(obs)` |

**Design decision — separate `spaces.py` from the environment class:** Keeping space construction in its own module means tests can import and inspect spaces without instantiating the full environment. It also isolates the space-shaping logic from the step/reward logic, reducing the environment class size (NFR-01.5: <300 lines).

### 3.3 `env/conversation.py` — Simulated User & Conversation Logic

| Responsibility | Detail |
|---|---|
| `ProblemRegistry` | Holds all problem type definitions; samples a problem via the RNG |
| `ConversationState` | Mutable dataclass tracking: problem_type, turn_count, info_collected (dict), frustration_level, last_action, action_history, diagnosed (bool), proposed_solution (bool), solution_rejected (bool) |
| `UserStateMachine` | Given the current state and agent action, produces the user's response message type, updates info_collected, updates frustration, and signals resolution/rejection |

**Design decision — state machine pattern for the simulated user:** The user's response depends on both the agent's current action and the conversation history. A state machine with explicit transition rules is the simplest correct abstraction. Each agent action maps to a handler method that reads the current state, mutates it, and returns a response message type. This keeps the user simulation testable in isolation.

### 3.4 `env/customer_support_env.py` — Gymnasium Environment

| Responsibility | Detail |
|---|---|
| `__init__` | Validate `max_turns`, build action/observation spaces, initialize RNG, set `_terminated` flag |
| `reset(seed, options)` | Validate, reseed RNG, sample problem, init ConversationState, build initial observation, return `(obs, info)` |
| `step(action)` | Validate action + episode-active flag, delegate to UserStateMachine, compute reward, check termination/truncation, build observation, return 5-tuple |
| `render()` | Print human-readable conversation summary to console |
| `close()` | No-op cleanup (no external resources) |

**Design decision — environment holds ConversationState, not spaces:** The environment owns a `ConversationState` instance that it passes to `UserStateMachine`. The spaces themselves are built once in `__init__` and stored as `self.action_space` and `self.observation_space`. The environment never reconstructs spaces on each step — it only builds new observation dicts from the current state.

### 3.5 `agents/rule_based_agent.py` — Baseline Agent

| Responsibility | Detail |
|---|---|
| `__init__` | Stores action space info, initializes last-action tracking |
| `act(observation)` | Validates observation keys, applies priority policy, returns action int |
| `reset()` | Clears internal tracking (last action, consecutive count) for a new episode |

### 3.6 `evaluation/run_evaluation.py` — Evaluation Harness

| Responsibility | Detail |
|---|---|
| `run_evaluation(num_episodes, max_turns, seed, agent, output_path)` | Orchestrates full evaluation run |
| `run_single_episode(env, agent, seed)` | Runs one episode, returns per-episode record dict |
| `compute_metrics(episode_records)` | Computes all aggregate metrics from episode records |
| `write_report(metrics, episode_records, config, output_path)` | Writes JSON file, creates `results/` dir |
| `print_summary(metrics)` | Prints human-readable summary to stdout |
| `main()` | CLI entry point |

---

## 4. Environment Lifecycle

### 4.1 State Transitions

The environment has three lifecycle states:

```
                    ┌─────────┐
          init()    │  IDLE   │
        ──────────► │ (ready) │
                    └────┬────┘
                         │ reset()
                         ▼
                    ┌─────────────┐
                    │   ACTIVE     │
                    │ (episode     │
                    │  in progress)│
                    └─────┬───────┘
                          │ step() returns
                          │   terminated=True
                          │   OR truncated=True
                          ▼
                    ┌─────────────┐
                    │  ENDED      │
                    │ (episode    │
                    │  complete)  │
                    └─────┬───────┘
                          │ reset()
                          ▼
                    ┌─────────────┐
                    │   ACTIVE     │
                    └─────────────┘
```

### 4.2 Detailed Lifecycle

**Initialization (`__init__`):**
1. Validate `max_turns` is a positive integer (raises `ValueError` otherwise).
2. Store `max_turns` as instance attribute.
3. Build `self.action_space` via `spaces.build_action_space()`.
4. Build `self.observation_space` via `spaces.build_observation_space(...)`.
5. Initialize `self._rng = None` (set on first `reset`).
6. Set `self._episode_active = False` (no episode is running yet).
7. Initialize `self._state = None` (no conversation state until reset).

**Reset (`reset(seed=None, options=None)`):**
1. If `seed` is not None, create `self._rng = np.random.default_rng(seed)`. If `seed` is None and `self._rng` is None, create `self._rng = np.random.default_rng()`.
2. Sample a problem type index from the `ProblemRegistry` using `self._rng`.
3. Initialize a fresh `ConversationState` with the sampled problem.
4. Generate the initial user message type (always `MSG_INITIAL_PROBLEM` — the user states their issue).
5. Set `self._episode_active = True`.
6. Build observation from state, construct info dict with `{problem_type: name, turn_count: 0}`.
7. Return `(observation, info)`.

**Step (`step(action)`):**
1. If not `self._episode_active`: raise `RuntimeError("Episode has ended. Call reset() to start a new episode.")`.
2. Validate action: if `action` not in `self.action_space`: raise `ValueError(f"Invalid action {action}. Must be in [0, {self.action_space.n}).")`.
3. Increment `state.turn_count`.
4. Delegate to `UserStateMachine.process_action(state, action)` — this mutates the conversation state (updates info_collected, frustration, message type) and returns a `response` struct containing: new message type, whether info was newly gathered, whether solution was accepted/rejected, whether frustration increased.
5. Compute reward (see Section 7).
6. Check termination (see Section 8).
7. Build observation from current state.
8. If terminated or truncated: set `self._episode_active = False`.
9. Build info dict with `{turn_count, frustration_level, termination_reason (if terminal), info_gathered_this_step}`.
10. Return `(observation, reward, terminated, truncated, info)`.

**Render (`render()`):**
1. If no active episode and no state: print "No active episode."
2. Otherwise print: problem name, current turn, info collected fields, frustration level, last user message type, last action taken.

**Close (`close()`):**
1. Set `self._episode_active = False`, `self._state = None`. No external resources to release.

### 4.3 Key Design Decisions in the Lifecycle

**Why `_episode_active` flag?** Gymnasium does not mandate this, but FR-06.6 and AC-12.2 require that `step()` after termination raises an error. The flag is the simplest way to enforce this. It is set to `True` on `reset()` and `False` when any terminal condition is reached. Calling `step()` while `False` raises `RuntimeError`.

**Why validation before state mutation?** AC-12.8 requires that an invalid action does not corrupt internal state. By validating the action and episode-active flag before incrementing the turn counter or calling `UserStateMachine`, we guarantee the state is untouched on any error path.

**Why `reset()` is always allowed?** AC-12.3 permits mid-episode reset. This is standard Gymnasium behavior — `reset()` simply starts a fresh episode regardless of prior state. No error is raised.

---

## 5. Observation Design

### 5.1 Observation Space Definition

The observation space is a `gymnasium.spaces.Dict`:

```python
observation_space = spaces.Dict({
    "user_message_type":  spaces.Discrete(num_message_types, start=0),
    "turn_count":          spaces.Discrete(max_turns + 1, start=0),
    "problem_type":        spaces.Discrete(num_problems, start=0),
    "info_collected":      spaces.MultiBinary(max_info_fields),
    "frustration_level":   spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
})
```

### 5.2 Field-by-Field Rationale

| Field | Type | Range | Why This Type |
|---|---|---|---|
| `user_message_type` | `Discrete` | `[0, num_message_types)` | The user's latest message falls into discrete categories (initial problem, info provided, confirmation, rejection, frustration, escalation confirmation). A `Discrete` space is the natural fit and avoids the complexity of text encoding. |
| `turn_count` | `Discrete` | `[0, max_turns]` | Turn count is a bounded integer. Using `Discrete(max_turns + 1)` ensures the observation always satisfies `space.contains(obs)` since the count can never exceed `max_turns`. We use `Discrete` rather than `Box` because turns are inherently categorical-bounded integers, not continuous. |
| `problem_type` | `Discrete` | `[0, num_problems)` | Identifies which of the 5+ problem types was sampled. This is categorical — there is no ordinal relationship between problem types, so `Discrete` with one-hot semantics is correct. |
| `info_collected` | `MultiBinary` | shape `(max_info_fields,)` | A binary vector where each position corresponds to an information field. `1` = collected, `0` = not yet. `MultiBinary` is the canonical Gymnasium space for fixed-length binary vectors. The length is `max_info_fields` (the maximum number of required fields across all problem types), and unused positions are always `0`. |
| `frustration_level` | `Box` | `[0.0, 1.0]` | Frustration is a continuous float that increases and decreases based on agent behavior. `Box` with `shape=()` (scalar) is the correct space for a single bounded continuous value. Using `float32` matches NumPy default precision and Gymnasium conventions. |

### 5.3 Message Type Enum

```
0  MSG_INITIAL_PROBLEM     — User states their issue (first turn only)
1  MSG_INFO_PROVIDED       — User provides requested information
2  MSG_NO_RELEVANT_INFO    — User says they don't have that info / it's irrelevant
3  MSG_SOLUTION_ACCEPTED   — User confirms the solution works
4  MSG_SOLUTION_REJECTED   — User says the solution doesn't help
5  MSG_FRUSTRATED          — User expresses frustration
6  MSG_ESCALATION_CONFIRM  — User acknowledges escalation
7  MSG_DIAGNOSIS_ACK       — User acknowledges the diagnosis step
```

**Design decision — 8 message types:** This covers all meaningful user responses without redundancy. The initial message type is distinct from info-provided so the agent can tell whether it is the first turn. Frustration is a separate message type from rejection because a frustrated user may not be rejecting a specific solution — they are expressing a general emotional state. The agent can observe frustration_level as a float AND user_message_type as a category, giving redundant but complementary signals.

### 5.4 Observation Construction

The `build_observation(state)` function converts the internal `ConversationState` to a Gymnasium observation dict:

```
state.user_message_type   → np.int64 scalar
state.turn_count          → np.int64 scalar
state.problem_type        → np.int64 scalar
state.info_collected      → np.ndarray of shape (max_info_fields,), dtype=np.int8
state.frustration_level   → np.float32 scalar
```

All values are NumPy scalars or arrays, which is what Gymnasium's `space.contains()` expects.

**Design decision — fixed-length `info_collected` vector:** Different problem types have different numbers of required fields (e.g., billing needs 3, technical needs 4). Rather than variable-length vectors (which Gymnasium does not support in `MultiBinary`), we use a fixed-length vector sized to the maximum field count across all problems. Each problem type maps its required fields to specific positions in the vector. Positions not used by the current problem type are always 0. This keeps the observation space fixed and valid across all episodes.

---

## 6. Action-Space Design

### 6.1 Action Space

```python
action_space = spaces.Discrete(7)  # 7 actions: 0 through 6
```

### 6.2 Action Definitions

| Integer | Name | Description |
|---|---|---|
| 0 | `ASK_ACCOUNT_INFO` | Request account/identity verification (name, email, account ID) |
| 1 | `ASK_BILLING_INFO` | Request billing details (last charge, payment method, invoice number) |
| 2 | `ASK_TECHNICAL_INFO` | Request technical details (device, OS, error message, steps to reproduce) |
| 3 | `DIAGNOSE` | Attempt to identify the root cause based on collected info |
| 4 | `PROPOSE_SOLUTION` | Propose a fix or resolution to the user |
| 5 | `ESCALATE` | Hand off to a human agent |
| 6 | `APOLOGIZE` | Apologize to reduce user frustration |

### 6.3 Action-to-Info-Field Mapping

Each `ASK_*` action maps to specific information fields depending on the problem type. This mapping is defined in the problem type definitions in `config.py`:

| Problem Type | ASK_ACCOUNT_INFO gathers | ASK_BILLING_INFO gathers | ASK_TECHNICAL_INFO gathers |
|---|---|---|---|
| Billing dispute | account_id | last_charge, payment_method | — |
| Technical glitch | account_id | — | device_type, error_message |
| Account lockout | account_id, email | — | — |
| Order not received | account_id | order_number | — |
| Refund request | account_id | transaction_id, refund_reason | — |

**Design decision — `ASK_*` actions are problem-specific:** Rather than having a generic "ask for info" action, we split into three categories (account, billing, technical) that correspond to the types of information a real support agent would request. This gives the environment enough granularity to penalize irrelevant questions (asking for billing info on a technical glitch problem increases frustration) and reward targeted information gathering. The mapping from action to specific fields is per-problem-type, so the same action gathers different fields depending on the context.

### 6.4 Why `Discrete` Over `MultiBinary` or `Box`

The agent takes exactly one action per turn. `Discrete(n)` is the canonical Gymnasium space for this: it represents choosing one of `n` options. `MultiBinary` would imply the agent can take multiple actions simultaneously, and `Box` would imply continuous action values. Neither fits the turn-based conversation model.

### 6.5 Gymnasium `start` Parameter

Gymnasium's `Discrete` space accepts a `start` parameter (default 0). We use the default `start=0` so action integers are `[0, 1, 2, 3, 4, 5, 6]`, matching Python's 0-indexing convention and making array indexing natural.

---

## 7. Reward Design

### 7.1 Reward Constants (in `config.py`)

| Constant | Value | Trigger |
|---|---|---|
| `REWARD_RESOLVE` | `+10.0` | Correct solution proposed with all required info gathered, user accepts |
| `REWARD_INFO_GATHERED` | `+1.0` | A previously-uncollected required field is gathered this turn |
| `REWARD_TURN_PENALTY` | `-0.5` | Applied every turn (efficiency pressure) |
| `REWARD_WRONG_SOLUTION` | `-3.0` | Solution proposed but wrong (missing info or wrong problem) |
| `REWARD_REPEATED_ACTION` | `-2.0` | Same action repeated without progress (no new info gathered), except APOLOGIZE |
| `REWARD_ESCALATE` | `+2.0` | Agent escalates to human |
| `REWARD_TRUNCATE` | `-5.0` | Turn limit reached without resolution |
| `REWARD_APOLOGIZE_HIGH_FRUSTRATION` | `+0.5` | Apologize when frustration > 0.5 |
| `REWARD_APOLOGIZE_LOW_FRUSTRATION` | `0.0` | Apologize when frustration <= 0.5 (no benefit) |

### 7.2 Reward Computation Algorithm

The reward for a single `step()` is computed as a sum of applicable components. The computation follows a priority order — terminal events are checked first, and if a terminal event fires, non-terminal rewards for the same step are still applied except as noted in AC-05.10.

```
reward = REWARD_TURN_PENALTY   # always applied

if action == last_action and action != APOLOGIZE and not new_info_gathered:
    reward += REWARD_REPEATED_ACTION   # progress-aware: only fires without progress

if new_info_gathered:
    reward += REWARD_INFO_GATHERED

if action == PROPOSE_SOLUTION:
    if all_info_collected and solution_correct:
        reward += REWARD_RESOLVE
        # episode terminates — no further rewards this step
    else:
        reward += REWARD_WRONG_SOLUTION

if action == ESCALATE:
    reward += REWARD_ESCALATE
    # episode terminates

if action == APOLOGIZE:
    if frustration_level > APOLOGY_THRESHOLD:
        reward += REWARD_APOLOGIZE_HIGH_FRUSTRATION
    else:
        reward += REWARD_APOLOGIZE_LOW_FRUSTRATION

if turn_count >= max_turns and not resolved and not escalated:
    reward += REWARD_TRUNCATE
    # episode truncates
```

### 7.3 Design Decisions in Reward Design

**Why turn penalty is always applied, even on resolution:** The `-0.5` turn penalty creates a constant pressure toward efficiency. An agent that resolves in 5 turns earns more than one that resolves in 10 turns, even if both reach `+10.0` for resolution. Without a per-turn penalty, the agent has no incentive to be efficient. Applying it even on the terminal step means the total reward for an episode is `10.0 - 0.5 * num_turns + other_bonuses`.

**Why `REWARD_RESOLVE` replaces `REWARD_WRONG_SOLUTION` when correct:** AC-05.10 states the terminal step includes only the terminal event reward plus the turn penalty, not stacked bonuses. When the solution is correct, we add `REWARD_RESOLVE` (not `REWARD_WRONG_SOLUTION`). When wrong, we add `REWARD_WRONG_SOLUTION`. These are mutually exclusive — the solution is either accepted or rejected, never both.

**Why the repeated-action penalty is progress-aware:** Some scenarios require the same ASK action to be called multiple times to gather different fields (e.g., Scenario 3 Password Reset uses `ASK_ACCOUNT_INFO` to collect `account_id`, `email`, and `security_question` on three separate turns). Penalising these productive repetitions would incorrectly punish valid information-gathering behaviour. The penalty therefore only fires when the same action is repeated AND `response.info_gathered` is `False` — meaning the action made no progress. Truly unproductive repeats (e.g., asking for billing info on a technical problem, or proposing a solution twice) are still penalised.

**Why repeated-action penalty excludes APOLOGIZE:** The rule-based agent may need to apologize multiple times if frustration stays high. Penalizing repeated apologies would discourage de-escalation. The agent's anti-repetition logic (FR-08.3) also exempts APOLOGIZE from its consecutive-limit rule, keeping both systems consistent.

**Why escalation gets positive but lower reward than resolution:** Escalation is a valid outcome — sometimes the agent genuinely cannot solve the problem. `+2.0` makes it better than doing nothing (which would accumulate turn penalties until truncation at `-5.0`) but worse than resolution (`+10.0`). This teaches the agent that escalation is a fallback, not a primary strategy.

**Why truncation penalty is separate from turn penalties:** Truncation (`-5.0`) is a terminal penalty applied once when the turn limit is hit, on top of the accumulated turn penalties. It signals a clear failure: the agent ran out of turns without resolving or escalating. This is distinct from the per-turn efficiency penalty, which is about pacing, not failure.

**Reward bounds:** The minimum possible episode reward is achieved by hitting the turn limit with all unproductive wrong actions: `15 * (-0.5) + 14 * (-2.0) + (-5.0) = -40.5`. The maximum is achieved by resolving in the minimum turns with all-progress info gathering (gather all info, diagnose, propose) with no unproductive repeats: `3 * (1.0 - 0.5) + (0.0 - 0.5) + (10.0 - 0.5) = +10.5` for a 3-field problem resolved in 5 turns. With the progress-aware penalty, the baseline rule-based agent consistently achieves `+10.5` across all scenarios. These bounds ensure rewards are always finite (AC-05.9).

---

## 8. Episode Termination Rules

### 8.1 `terminated` vs. `truncated` — Gymnasium Convention

Gymnasium distinguishes two reasons an episode can end:

| Flag | Meaning | In This Project |
|---|---|---|
| `terminated = True` | The episode reached a natural conclusion. The MDP has a terminal state. | Problem resolved, or agent escalated. |
| `truncated = True` | The episode was cut short by an external time/step limit, not by reaching a terminal state in the MDP. | Turn limit reached without resolution. |

These are **separate booleans**. In our environment, they are mutually exclusive for any single step:

| Condition | `terminated` | `truncated` | `termination_reason` |
|---|---|---|---|
| Correct solution accepted | `True` | `False` | `"resolved"` |
| Agent escalates | `True` | `False` | `"escalated"` |
| Turn limit reached | `False` | `True` | `"truncated"` |
| Non-terminal step | `False` | `False` | `None` (key absent from info) |

### 8.2 Why They Are Never Both True

In Gymnasium's formal model, `terminated` and `truncated` can theoretically both be `True` (e.g., a time limit coincides with a terminal state). However, in our design, truncation only fires when the agent has NOT resolved or escalated. If the agent resolves or escalates on the final allowed turn, `terminated` is `True` and `truncated` is `False` — the episode ended naturally at the boundary, not by truncation. Truncation is the fallback for when no natural terminal state was reached. This makes the three terminal conditions mutually exclusive, simplifying metric computation (FR-10.3: `success_rate + escalation_rate + truncation_rate == 1.0`).

### 8.3 Termination Check Order in `step()`

```
1. If action == PROPOSE_SOLUTION and solution accepted:
     terminated = True, reason = "resolved"

2. Else if action == ESCALATE:
     terminated = True, reason = "escalated"

3. Else if turn_count >= max_turns:
     truncated = True, reason = "truncated"

4. Else:
     terminated = False, truncated = False
```

The order matters: resolution and escalation are checked before truncation. This ensures that if the agent resolves on the final turn, it gets credit for resolution (and the `+10.0` reward) rather than being truncated (and getting `-5.0`).

### 8.4 Post-Termination Protection

After any step where `terminated` or `truncated` is `True`, `self._episode_active` is set to `False`. Any subsequent `step()` call raises `RuntimeError` (AC-12.2). The only way to continue is `reset()`.

---

## 9. Customer Scenario Data Model

### 9.1 Problem Type Definition

Each problem type is defined as a frozen dataclass:

```python
@dataclass(frozen=True)
class ProblemType:
    id: int                    # Unique integer identifier (0, 1, 2, ...)
    name: str                  # Human-readable name
    required_fields: tuple[str, ...]  # Ordered tuple of field names
    resolution_action: int     # The action that resolves this problem (always PROPOSE_SOLUTION = 4)
    ask_field_mapping: dict[int, str | None]  # Maps action_id -> field_name or None
    # e.g., {ASK_ACCOUNT_INFO: "account_id", ASK_BILLING_INFO: "last_charge", ASK_TECHNICAL_INFO: None}
```

**Design decision — `resolution_action` is always `PROPOSE_SOLUTION`:** The action that resolves any problem is always action 4 (propose a solution). What makes the solution "correct" is not the action itself but the state: all required info must be collected before proposing. This keeps the action space uniform — the agent always proposes the same action, and the environment determines correctness from the conversation state. This avoids the need for problem-specific resolution actions, which would require the agent to somehow know the problem type's "magic" resolution action.

### 9.2 Defined Problem Types

| ID | Name | Required Fields | Relevant ASK Actions |
|---|---|---|---|
| 0 | Billing Dispute | `account_id`, `last_charge`, `payment_method` | ASK_ACCOUNT_INFO → `account_id`, ASK_BILLING_INFO → `last_charge` (first call) then `payment_method` (second call) |
| 1 | Technical Glitch | `account_id`, `device_type`, `error_message` | ASK_ACCOUNT_INFO → `account_id`, ASK_TECHNICAL_INFO → `device_type` (first) then `error_message` (second) |
| 2 | Account Lockout | `account_id`, `email` | ASK_ACCOUNT_INFO → `account_id` (first) then `email` (second) |
| 3 | Order Not Received | `account_id`, `order_number` | ASK_ACCOUNT_INFO → `account_id`, ASK_BILLING_INFO → `order_number` |
| 4 | Refund Request | `account_id`, `transaction_id`, `refund_reason` | ASK_ACCOUNT_INFO → `account_id`, ASK_BILLING_INFO → `transaction_id` (first) then `refund_reason` (second) |

### 9.3 `ask_field_mapping` Multi-Field Logic

A single `ASK_*` action can gather multiple fields for a problem type, but only one per turn. The mapping uses a list of field names per action, and the `UserStateMachine` returns the next uncollected field in the list:

```python
ask_field_mapping = {
    ASK_ACCOUNT_INFO: ["account_id"],
    ASK_BILLING_INFO: ["last_charge", "payment_method"],
    ASK_TECHNICAL_INFO: [],
}
```

When the agent calls `ASK_BILLING_INFO` the first time, it gets `last_charge`. The second time, it gets `payment_method`. If called a third time (both already collected), the user responds with `MSG_NO_RELEVANT_INFO` and frustration increases.

**Design decision — one field per turn per action:** This mirrors real support conversations — you ask one question, get one answer. It also creates a natural lower bound on episode length (you cannot gather 3 fields in fewer than 3 turns), which makes the reward landscape more interesting and prevents trivially short optimal episodes.

### 9.4 `ConversationState` Data Model

```python
@dataclass
class ConversationState:
    problem: ProblemType          # The sampled problem definition
    turn_count: int               # Current turn (starts at 0)
    info_collected: dict[str, bool]  # field_name -> collected flag
    frustration_level: float      # [0.0, 1.0], starts at 0.0
    last_action: int | None       # Previous action (for repeat detection)
    consecutive_action_count: int # How many times last_action was repeated
    diagnosed: bool               # Whether DIAGNOSE has been performed
    solution_proposed: bool        # Whether PROPOSE_SOLUTION has been attempted
    solution_rejected: bool       # Whether the last solution was rejected
    user_message_type: int        # Latest user message type (for observation)
```

### 9.5 `UserStateMachine` Behavior Rules

| Agent Action | Condition | User Response | State Change |
|---|---|---|---|
| `ASK_ACCOUNT_INFO` | Mapped field uncollected | `MSG_INFO_PROVIDED` | Mark field collected, frustration unchanged |
| `ASK_ACCOUNT_INFO` | All mapped fields collected | `MSG_NO_RELEVANT_INFO` | Frustration += 0.1 |
| `ASK_ACCOUNT_INFO` | No mapping for this problem | `MSG_NO_RELEVANT_INFO` | Frustration += 0.15 |
| `ASK_BILLING_INFO` | Mapped field uncollected | `MSG_INFO_PROVIDED` | Mark field collected |
| `ASK_BILLING_INFO` | All mapped fields collected | `MSG_NO_RELEVANT_INFO` | Frustration += 0.1 |
| `ASK_BILLING_INFO` | No mapping | `MSG_NO_RELEVANT_INFO` | Frustration += 0.15 |
| `ASK_TECHNICAL_INFO` | (same pattern) | (same pattern) | (same pattern) |
| `DIAGNOSE` | All info collected | `MSG_DIAGNOSIS_ACK` | `diagnosed = True`, frustration -= 0.05 |
| `DIAGNOSE` | Info incomplete | `MSG_FRUSTRATED` | `diagnosed = True`, frustration += 0.1 |
| `PROPOSE_SOLUTION` | All info collected and `diagnosed == True` | `MSG_SOLUTION_ACCEPTED` | Resolution! |
| `PROPOSE_SOLUTION` | All info collected but `diagnosed == False` | `MSG_SOLUTION_REJECTED` | `solution_rejected = True`, frustration += 0.2 |
| `PROPOSE_SOLUTION` | Info incomplete | `MSG_SOLUTION_REJECTED` | `solution_rejected = True`, frustration += 0.2 |
| `ESCALATE` | (always) | `MSG_ESCALATION_CONFIRM` | Episode terminates |
| `APOLOGIZE` | Frustration > 0.0 | `MSG_INFO_PROVIDED` (ack) | Frustration -= 0.3 (clamped at 0.0) |
| `APOLOGIZE` | Frustration == 0.0 | `MSG_INFO_PROVIDED` (ack) | No change |

**Design decision — diagnosis is required before resolution:** A solution proposed without diagnosis is rejected even if all info is gathered. This forces the agent to follow a realistic support workflow: gather info → diagnose → propose. The rule-based agent naturally follows this order. A learned agent would need to discover this ordering, making the environment non-trivial. The frustration increase for premature proposals (+0.2) is significant enough to make skipping diagnosis clearly suboptimal.

**Design decision — frustration clamped at [0.0, 1.0]:** Frustration is bounded. It can never go negative (you cannot be "anti-frustrated") and never exceed 1.0 (capped at maximum frustration). Clamping happens after every update. This keeps the observation valid (`Box([0.0], [1.0])`) and prevents unbounded reward from repeated apologies.

---

## 10. Baseline Agent Decision Logic

### 10.1 Policy Priority Order

The rule-based agent implements a strict priority cascade. The first matching condition determines the action:

```
def act(observation):
    1. Validate observation keys (raise KeyError if missing)

    2. If frustration_level > FRUSTRATION_HIGH_THRESHOLD (0.7):
         → return APOLOGIZE

    3. If not all required info fields collected:
         → return the ASK_* action that maps to the next uncollected field

    4. If all info collected and not diagnosed:
         → return DIAGNOSE

    5. If diagnosed and not solution_proposed:
         → return PROPOSE_SOLUTION

    6. If solution_rejected (proposed but rejected):
         → return ESCALATE

    7. Fallback (should not occur in normal flow):
         → return ESCALATE
```

### 10.2 Anti-Repetition Logic

The agent tracks `last_action` and `consecutive_count`. Before returning an action (except `APOLOGIZE`), it checks:

```
if action == last_action and action != APOLOGIZE and consecutive_count >= 2:
    # Would be the 3rd consecutive same action — pick an alternative
    # The fallback is ESCALATE (safe exit)
    action = ESCALATE
```

**Design decision — ESCALATE as the anti-repetition fallback:** If the agent would repeat the same non-apologize action three times, something is wrong with its logic (e.g., it keeps asking for info that's already collected). Rather than looping, it escalates. This is a safe, terminal action that prevents infinite loops. In practice, the rule-based agent's priority cascade should rarely trigger this — the anti-repetition check is a safety net.

### 10.3 Info-Gathering Action Selection

The agent needs to know which `ASK_*` action to use for each uncollected field. It uses a reverse mapping derived from the problem type definition:

```
problem = PROBLEM_TYPES[observation["problem_type"]]
uncollected_fields = [f for f in problem.required_fields if not info_collected[f]]
if uncollected_fields:
    next_field = uncollected_fields[0]  # Pick first uncollected
    action = FIELD_TO_ACTION[next_field]  # Reverse lookup: field -> action
```

The `FIELD_TO_ACTION` mapping is a global dict built at import time from the problem type definitions. For example:

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

### 10.4 Agent State Management

The agent maintains internal state across turns within an episode:

| Field | Purpose |
|---|---|
| `last_action` | For anti-repetition check |
| `consecutive_count` | How many times `last_action` was repeated |
| `has_diagnosed` | Whether the agent has already performed DIAGNOSE this episode |
| `has_proposed` | Whether the agent has already proposed a solution |

The agent exposes a `reset()` method that clears this internal state, called by the evaluation harness before each episode.

**Design decision — agent keeps minimal state:** The agent could theoretically derive everything from the observation (the observation contains `info_collected`, `frustration_level`, `turn_count`). However, `diagnosed` and `solution_proposed` are not in the observation space — they are internal conversation state that the agent must track itself. This is a deliberate observation design choice: including `diagnosed` would leak the solution path to the agent, making the problem trivial. The agent must infer from the observation whether diagnosis has occurred (e.g., by tracking its own action history).

---

## 11. Evaluation Architecture

### 11.1 Evaluation Flow

```
run_evaluation(num_episodes=100, max_turns=15, seed=42, agent=RuleBasedAgent())
│
├── 1. Validate inputs (num_episodes > 0, seed is int or None)
│
├── 2. Create environment: CustomerSupportEnv(max_turns=max_turns)
│
├── 3. For episode_id in range(num_episodes):
│      ├── episode_seed = seed + episode_id  (deterministic per-episode seeding)
│      ├── agent.reset()
│      ├── obs, info = env.reset(seed=episode_seed)
│      ├── episode_reward = 0.0
│      ├── terminated = False, truncated = False
│      │
│      ├── while not terminated and not truncated:
│      │     ├── action = agent.act(obs)
│      │     ├── obs, reward, terminated, truncated, info = env.step(action)
│      │     └── episode_reward += reward
│      │
│      └── record = {
│             "episode_id": episode_id,
│             "total_reward": episode_reward,
│             "num_turns": env._state.turn_count,
│             "termination_reason": info.get("termination_reason", "unknown"),
│             "success": info.get("termination_reason") == "resolved"
│         }
│
├── 4. metrics = compute_metrics(episode_records)
├── 5. write_report(metrics, episode_records, config_dict, output_path)
└── 6. print_summary(metrics)
```

### 11.2 Per-Episode Seeding Strategy

Each episode uses a deterministic seed derived from the base seed: `episode_seed = seed + episode_id`. This ensures:
- Reproducibility: the same `seed` always produces the same sequence of problems.
- Independence: changing one episode's seed does not affect others.
- Deterministic overlap avoidance: episodes never reuse the same RNG state.

**Design decision — additive seeding over sequential RNG draws:** An alternative is to create one RNG at the start and draw from it for each episode. However, if we later change `num_episodes`, the RNG stream shifts and all episodes change. With additive seeding (`seed + episode_id`), episode 5 always uses seed 47 regardless of how many total episodes are run. This makes per-episode results independently reproducible.

### 11.3 Error Handling in Evaluation

Per AC-09.6, the evaluation harness catches exceptions per-episode without halting:

```python
for episode_id in range(num_episodes):
    try:
        record = run_single_episode(env, agent, episode_seed)
    except Exception as e:
        record = {
            "episode_id": episode_id,
            "total_reward": 0.0,
            "num_turns": 0,
            "termination_reason": "error",
            "success": False,
            "error": str(e),
        }
        logger.warning(f"Episode {episode_id} failed: {e}")
    episode_records.append(record)
```

Episodes that error are recorded with `termination_reason = "error"` and zero reward. They do not count toward success/escalation/truncation rates (the rates sum to 1.0 only over non-error episodes, or we normalize accordingly). In practice, with the rule-based agent, errors should not occur — this is a defensive measure for future agents.

### 11.4 CLI Entry Point

```bash
python -m evaluation.run_evaluation [--episodes 100] [--max-turns 15] [--seed 42] [--output results/evaluation_results.json]
```

The `main()` function parses `argparse` arguments, constructs the agent and calls `run_evaluation()`.

---

## 12. Metrics Calculation

### 12.1 Per-Episode Data

Each episode produces a record:

```python
{
    "episode_id": int,
    "total_reward": float,
    "num_turns": int,
    "termination_reason": str,  # "resolved", "escalated", "truncated", "error"
    "success": bool,
}
```

### 12.2 Aggregate Metrics

All metrics are computed from the list of episode records using NumPy:

| Metric | Computation | NumPy Function |
|---|---|---|
| `mean_reward` | Mean of `total_reward` across all episodes | `np.mean(rewards)` |
| `std_reward` | Population std of `total_reward` | `np.std(rewards)` |
| `min_reward` | Minimum `total_reward` | `np.min(rewards)` |
| `max_reward` | Maximum `total_reward` | `np.max(rewards)` |
| `success_rate` | Count of `reason == "resolved"` / total episodes | `np.mean(reasons == "resolved")` |
| `mean_turns` | Mean of `num_turns` | `np.mean(turns)` |
| `std_turns` | Population std of `num_turns` | `np.std(turns)` |
| `escalation_rate` | Count of `reason == "escalated"` / total episodes | `np.mean(reasons == "escalated")` |
| `truncation_rate` | Count of `reason == "truncated"` / total episodes | `np.mean(reasons == "truncated")` |
| `num_episodes` | Total count | `len(records)` |

### 12.3 Validation Constraint

`success_rate + escalation_rate + truncation_rate` must equal 1.0 (within `np.isclose` tolerance), because every episode ends in exactly one of these three terminal states. If any episode has `termination_reason == "error"`, this constraint is relaxed to `success_rate + escalation_rate + truncation_rate + error_rate == 1.0`.

**Design decision — population std (`ddof=0`) over sample std (`ddof=1`):** With 100 episodes, the difference is negligible. We use population std (`np.std` default `ddof=0`) for consistency and simplicity. This is a design choice, not a statistical claim — we are describing the observed sample, not estimating a population parameter.

### 12.4 Metrics Return Type

The `compute_metrics()` function returns a plain dict with all keys as native Python types (float, int) rather than NumPy scalars, to ensure JSON serialization works without custom encoders:

```python
metrics = {
    "mean_reward": float(np.mean(rewards)),
    "std_reward": float(np.std(rewards)),
    ...
}
```

---

## 13. JSON Report Structure

### 13.1 Top-Level Structure

```json
{
    "config": {
        "num_episodes": 100,
        "max_turns": 15,
        "seed": 42,
        "agent_type": "RuleBasedAgent"
    },
    "metrics": {
        "mean_reward": 7.35,
        "std_reward": 2.12,
        "min_reward": -5.0,
        "max_reward": 10.5,
        "success_rate": 0.78,
        "mean_turns": 6.2,
        "std_turns": 1.8,
        "escalation_rate": 0.15,
        "truncation_rate": 0.07,
        "num_episodes": 100
    },
    "episodes": [
        {
            "episode_id": 0,
            "total_reward": 10.5,
            "num_turns": 5,
            "termination_reason": "resolved",
            "success": true
        },
        {
            "episode_id": 1,
            "total_reward": 2.0,
            "num_turns": 1,
            "termination_reason": "escalated",
            "success": false
        },
        ...
    ]
}
```

### 13.2 Design Decisions

**Why `config` is first:** JSON key order is not semantically meaningful, but placing `config` first makes the file human-readable — you see the experimental setup before the results. Python dicts preserve insertion order, and `json.dump` respects it.

**Why per-episode data is included:** The aggregate metrics tell you the overall picture, but per-episode data lets you inspect outliers (which episodes failed, which took the most turns). This is critical for debugging the agent and the environment. The `episodes` array is 100 entries, which is small enough to be human-scannable and machine-parseable.

**Why `success` is a separate boolean:** `termination_reason == "resolved"` is the definition of success, but having a precomputed boolean makes filtering in downstream analysis trivial (e.g., `sum(e["success"] for e in data["episodes"])`).

**Why no nested metrics nesting:** Metrics are a flat dict at `data["metrics"]`. A flat structure is easier to parse and less error-prone than nested hierarchies. Each metric name is self-descriptive.

### 13.3 Console Output Format

The `print_summary()` function outputs a formatted table:

```
=== Evaluation Results ===
Episodes:           100
Agent:              RuleBasedAgent
Seed:               42
Max Turns:          15

--- Reward ---
Mean:               7.35
Std:                2.12
Min:                -5.00
Max:                10.50

--- Outcomes ---
Success Rate:       78.0%
Escalation Rate:    15.0%
Truncation Rate:     7.0%

--- Efficiency ---
Mean Turns:         6.2
Std Turns:          1.8

Report saved to: results/evaluation_results.json
```

---

## 14. Error Handling

### 14.1 Error Handling Matrix

| Component | Error Condition | Exception | Message Pattern |
|---|---|---|---|
| Env constructor | `max_turns <= 0` | `ValueError` | `"max_turns must be a positive integer, got {value}"` |
| Env constructor | `max_turns` not int | `ValueError` | `"max_turns must be an integer, got {type}"` |
| `reset()` | `seed` not int or None | `ValueError` | `"seed must be an integer or None, got {type}"` |
| `step()` | Episode not active | `RuntimeError` | `"Episode has ended. Call reset() to start a new episode."` |
| `step()` | Action out of range | `ValueError` | `"Invalid action {action}. Must be in [0, {n})."` |
| `step()` | Action not integer | `ValueError` | `"Action must be an integer, got {type}"` |
| Agent `act()` | Missing observation key | `KeyError` | `"Observation missing required key: '{key}'"` |
| Agent `act()` | Invalid observation value type | `ValueError` | `"Observation field '{key}' has invalid type: expected {expected}, got {actual}"` |
| Evaluation | `num_episodes <= 0` | `ValueError` | `"num_episodes must be a positive integer, got {value}"` |
| Evaluation | `seed` not int or None | `ValueError` | `"seed must be an integer or None, got {type}"` |
| Evaluation | Per-episode exception | (logged, not raised) | `"Episode {id} failed: {exception}"` |

### 14.2 Validation Before Mutation Principle

Every mutating operation validates all inputs before making any change to internal state. This is enforced structurally:

```
step(action):
    1. Check _episode_active     → raise if False (no state change)
    2. Validate action type       → raise if not int (no state change)
    3. Validate action range      → raise if out of bounds (no state change)
    4. NOW mutate state           → increment turn, update conversation
    5. Compute reward
    6. Check termination
    7. Build observation
    8. Return
```

Steps 1-3 are pure validation. If any fails, the function raises before reaching step 4. The environment state is guaranteed untouched on any error path. This satisfies AC-12.8.

### 14.3 Exception Types Rationale

| Exception | Why This Type |
|---|---|
| `ValueError` | Used for invalid argument values (wrong type, out of range). Standard Python convention for "you passed a bad value." |
| `RuntimeError` | Used for invalid state (calling `step()` after termination). The operation is valid in principle but the object is in the wrong state. |
| `KeyError` | Used by the agent when an observation dict is missing a required key. This is the standard Python exception for missing dict keys. |

We do not use custom exception classes. The three standard exceptions cover all cases, and using standard exceptions means callers can catch them without importing custom classes. This simplifies the API and reduces coupling.

### 14.4 No Bare Except Clauses (NFR-05.1)

All `try/except` blocks catch specific exception types. In the evaluation harness, the per-episode catch uses `except Exception as e:` (not bare `except:`), which catches all non-system-exiting exceptions. This is the broadest acceptable pattern — bare `except:` would also catch `KeyboardInterrupt` and `SystemExit`, which we do not want to suppress.

---

## 15. Testing Strategy

### 15.1 Test File Structure

```
tests/
├── conftest.py          # Shared pytest fixtures
├── test_env.py          # Environment tests (FR-01 through FR-07, FR-12)
├── test_agent.py        # Agent tests (FR-08, FR-12)
└── test_evaluation.py   # Evaluation tests (FR-09, FR-10, FR-11, FR-12)
```

### 15.2 Shared Fixtures (`conftest.py`)

| Fixture | Scope | Returns | Purpose |
|---|---|---|---|
| `env` | function | `CustomerSupportEnv(max_turns=15)` | Fresh env per test |
| `seeded_env` | function | `CustomerSupportEnv` after `reset(seed=42)` | Deterministic env state |
| `agent` | function | `RuleBasedAgent()` | Fresh agent per test |
| `small_env` | function | `CustomerSupportEnv(max_turns=3)` | Short episodes for truncation tests |
| `fixed_seed` | session | `42` | Shared seed constant |

### 15.3 Test Categories

#### `test_env.py` — Environment API and Logic

| Test Name | Requirement | What It Verifies |
|---|---|---|
| `test_reset_returns_tuple` | FR-01 (AC-01.2) | `reset()` returns 2-tuple `(obs, info)` |
| `test_step_returns_5tuple` | FR-01 (AC-01.3) | `step()` returns 5-tuple |
| `test_env_subclass` | FR-01 (AC-01.1) | `isinstance(env, gymnasium.Env)` |
| `test_action_space_discrete` | FR-04 (AC-04.1) | `isinstance(env.action_space, spaces.Discrete)` |
| `test_observation_space_dict` | FR-03 (AC-03.1) | `isinstance(env.observation_space, spaces.Dict)` |
| `test_observation_contains_all_fields` | FR-03 (AC-03.2-03.6) | All 5 fields present with correct types |
| `test_observation_in_space` | FR-03 (AC-03.7) | `observation_space.contains(obs)` is True after reset and step |
| `test_action_names_mapping` | FR-04 (AC-04.3) | Each action int maps to a readable name |
| `test_invalid_action_raises` | FR-12 (AC-12.1) | Out-of-range action raises `ValueError` |
| `test_step_after_termination_raises` | FR-12 (AC-12.2) | `step()` after termination raises `RuntimeError` |
| `test_invalid_max_turns_raises` | FR-07 (AC-07.3) | `max_turns=0` and `max_turns=-1` raise `ValueError` |
| `test_turn_count_never_exceeds_max` | FR-07 (AC-07.4) | Observation turn_count <= max_turns |
| `test_truncated_at_max_turns` | FR-07 (AC-07.2) | `truncated=True` when turn limit reached |
| `test_resolved_termination` | FR-06 (AC-06.1) | Correct solution → `terminated=True`, reason="resolved" |
| `test_escalate_termination` | FR-06 (AC-06.2) | Escalate → `terminated=True`, reason="escalated" |
| `test_termination_reason_in_info` | FR-06 (AC-06.5) | `info["termination_reason"]` present on terminal steps |
| `test_reward_resolution` | FR-05 (AC-05.1) | Resolution reward is `+10.0` (plus turn penalty) |
| `test_reward_info_gathered` | FR-05 (AC-05.2) | Info gathering gives `+1.0` |
| `test_reward_turn_penalty` | FR-05 (AC-05.3) | Each step has `-0.5` base |
| `test_reward_wrong_solution` | FR-05 (AC-05.4) | Wrong solution gives `-3.0` |
| `test_reward_repeated_action` | FR-05 (AC-05.5) | Unproductive repeat (no progress) → `-2.0` |
| `test_reward_escalate` | FR-05 (AC-05.6) | Escalate gives `+2.0` |
| `test_reward_truncate` | FR-05 (AC-05.7) | Truncation gives `-5.0` |
| `test_reward_apologize_high` | FR-05 (AC-05.8) | Apologize at high frustration → `+0.5` |
| `test_reward_apologize_low` | FR-05 (AC-05.8) | Apologize at low frustration → `0.0` |
| `test_reset_mid_episode` | FR-12 (AC-12.3) | `reset()` during active episode works without error |
| `test_no_state_mutation_on_error` | FR-12 (AC-12.8) | Invalid action leaves state unchanged |
| `test_render_output` | FR-01 | `render()` prints to stdout (capture with `capsys`) |

#### `test_agent.py` — Agent Policy Logic

| Test Name | Requirement | What It Verifies |
|---|---|---|
| `test_act_returns_valid_action` | FR-08 (AC-08.1, 08.5) | Action is int in `[0, 6]` |
| `test_apologize_when_frustrated` | FR-08 (AC-08.2.1) | frustration > 0.7 → action == APOLOGIZE |
| `test_gathers_info_first` | FR-08 (AC-08.2.2) | Incomplete info → ASK_* action |
| `test_diagnoses_after_info` | FR-08 (AC-08.2.3) | All info + not diagnosed → DIAGNOSE |
| `test_proposes_after_diagnosis` | FR-08 (AC-08.2.4) | Diagnosed + not proposed → PROPOSE_SOLUTION |
| `test_escalates_after_rejection` | FR-08 (AC-08.2.5) | Solution rejected → ESCALATE |
| `test_no_excessive_repeats` | FR-08 (AC-08.3) | No non-apologize action repeated > 2 times |
| `test_deterministic` | FR-08 (AC-08.4) | Same observation → same action, always |
| `test_missing_key_raises` | FR-12 (AC-12.5) | Missing observation key → `KeyError` |
| `test_agent_reset_clears_state` | FR-08 | `reset()` clears internal tracking |

#### `test_evaluation.py` — Evaluation Harness

| Test Name | Requirement | What It Verifies |
|---|---|---|
| `test_run_evaluation_returns_dict` | FR-09 | Returns dict with `metrics`, `episodes`, `config` |
| `test_num_episodes_correct` | FR-09 | `len(episodes) == num_episodes` |
| `test_metrics_all_present` | FR-10 (AC-10.1) | All 10 metric keys present |
| `test_success_rate_range` | FR-10 (AC-10.2) | `0.0 <= success_rate <= 1.0` |
| `test_rates_sum_to_one` | FR-10 (AC-10.3) | `success + escalation + truncation ≈ 1.0` |
| `test_reproducible` | NFR-02 (AC-NFR-02.2) | Same seed → identical metrics |
| `test_json_valid` | FR-11 (AC-11.5) | `json.load()` on output file succeeds |
| `test_json_has_config` | FR-11 (AC-11.4) | JSON includes config with all 4 keys |
| `test_json_has_episodes` | FR-11 (AC-11.3) | Episodes array with correct per-episode keys |
| `test_results_dir_created` | FR-11 (AC-11.6) | `results/` exists after evaluation |
| `test_invalid_num_episodes` | FR-12 (AC-12.6) | `num_episodes=0` → `ValueError` |
| `test_invalid_seed` | FR-12 (AC-12.7) | `seed="abc"` → `ValueError` |
| `test_immediate_escalation` | FR-09 (AC-09.5) | 1-turn episode (immediate escalate) does not crash |
| `test_small_num_episodes` | FR-09 | `num_episodes=5` works correctly |

### 15.4 Testing Design Decisions

**Why function-scoped fixtures:** Each test gets a fresh environment and agent. This prevents cross-test contamination where one test's episode state leaks into the next. Function scope is slightly slower than module scope but guarantees isolation. With 100-episode evaluation tests, the overhead is still well under the 30-second limit (NFR-03.6).

**Why fixed seeds in tests:** Every test that involves randomness uses `seed=42` (or a derived value). This makes tests deterministic and reproducible. If a test fails, it fails the same way every time. The `fixed_seed` session fixture provides the shared constant.

**Why float comparisons use tolerance:** Per NFR-03.7, all float assertions use `pytest.approx()` or `np.isclose()`. For example, `assert metrics["success_rate"] == pytest.approx(0.78, abs=0.01)`. This prevents flaky tests from floating-point representation issues.

**Why edge case tests matter:** The tests for immediate escalation (1-turn episode), truncation (3-turn max), and repeated actions cover the boundaries of the environment's logic. These are the scenarios most likely to expose off-by-one errors, state corruption, or incorrect reward computation. Each edge case maps to a specific acceptance criterion.

**Why no external dependencies in tests:** Tests create environments, agents, and evaluation runs entirely in-process. No network calls, no file system access outside the project `results/` directory. The `test_json_valid` test writes to a temporary path (via `tmp_path` fixture) to avoid polluting the actual `results/` directory during testing.

**Why evaluation tests use small episode counts:** Most evaluation tests run 5-10 episodes (not 100) to keep execution fast. The full 100-episode run is tested once in a separate test to verify it completes without error. This balances coverage with speed.

---

## 16. Design Decision Summary

| Decision | Rationale | Alternative Considered |
|---|---|---|
| Numeric observation (no NLP) | Keeps dependencies minimal, environment fast, extensible to learned agents | Text embeddings would add complexity and external deps (transformers, torch) |
| `Dict` observation space | Gymnasium-native structured space, each field independently meaningful | Flat `Box` vector would lose field semantics and be harder to debug |
| `Discrete` action space | One action per turn is the natural model | `MultiBinary` would imply multiple simultaneous actions; `Box` implies continuous control |
| Centralized `config.py` | Single source of truth for all constants, easy tuning | Constants scattered in modules would require hunting for magic numbers |
| Frozen dataclasses for problem types | Immutable, type-safe, clean equality for tests | Plain dicts would lack type hints and immutability guarantees |
| `terminated` and `truncated` mutually exclusive | Simplifies metrics (rates sum to 1.0), matches our domain (truncation only when no natural end) | Gymnasium allows both True, but conflating them complicates analysis |
| One info field per turn per action | Mirrors real conversations, creates meaningful episode length lower bound | All-at-once gathering would make episodes trivially short |
| Diagnosis required before resolution | Forces realistic workflow, makes the problem non-trivial | Allowing direct resolution would make the agent's job too easy |
| Additive per-episode seeding | Each episode independently reproducible regardless of total count | Sequential RNG draws would shift all episodes if `num_episodes` changes |
| Standard exceptions (no custom classes) | Callers need no imports, simpler API | Custom exceptions would add a module dependency for error handling |
| Population std (`ddof=0`) | We describe observed data, not estimate population params | Sample std (`ddof=1`) is more correct statistically but the difference is negligible at n=100 |
| Agent tracks `diagnosed`/`proposed` internally | Not in observation — prevents leaking solution path to the agent | Including them in observation would make the problem trivially solvable |
