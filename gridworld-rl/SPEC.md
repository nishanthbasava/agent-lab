# GridWorld RL — Project Spec

Signatures and contracts only. Every body is yours to write. The rule of thumb for the
whole project: **one data structure per era** — Block 1 consumes `P`, Blocks 2+ consume
the Gymnasium `step`/`reset` API, and `Q` is always a `np.ndarray [n_states, n_actions]`.
If you keep those interfaces stable, every later file plugs into every earlier one.

---

## env/gridworld.py — the one environment, two interfaces

```python
class GridWorld(gym.Env):
    # construction
    def __init__(self, grid: list[str], slip: float = 0.1, step_cost: float = -0.04)
    # attrs: nrows, ncols, n_states, n_actions=4, start_state
    # action_space = Discrete(4), observation_space = Discrete(n_states)

    # geometry (done / in progress)
    def _to_state(self, r, c) -> int
    def _to_rc(self, s) -> tuple[int, int]
    def _move(self, s, a) -> int          # deterministic; walls/edges -> stay put

    # interface 1: the model (for DP, Block 1)
    def build_P(self) -> dict             # P[s][a] = [(prob, s_next, reward, done)]
                                          # terminals (G, T) absorbing: self-loop, r=0
    def is_terminal(self, s) -> bool

    # interface 2: the simulator (for MC/TD/deep, Block 2+)
    def reset(self, seed=None) -> tuple[int, dict]           # -> (obs, info)
    def step(self, a) -> tuple[int, float, bool, bool, dict] # -> (obs, r, terminated, truncated, info)
                                          # step SAMPLES from the same distribution P describes.
                                          # sanity check: empirical step() frequencies ≈ P.
```

Design decisions that are yours: whether `build_P` merges duplicate `s'` entries; whether
`step` is implemented *using* `P` (sample from `P[s][a]`) or independently (then P vs.
step agreement is a real test, not a tautology).

---

## tabular/dp.py — Block 1

All functions take `P` (never the env object) so they work on any finite MDP.

```python
def q_from_v(P, V, s, gamma) -> np.ndarray            # Q(s,·) one-step lookahead; used everywhere below
def policy_evaluation(P, policy, gamma, theta) -> V   # Bellman expectation, iterate to tol theta
def policy_improvement(P, V, gamma) -> policy         # greedy w.r.t. V
def policy_iteration(P, gamma, theta) -> (V_star, pi_star)
def value_iteration(P, gamma, theta) -> (V_star, pi_star)
```

`policy` representation: `np.ndarray [n_states]` of action ints (deterministic) is enough.
**Acceptance:** policy_iteration and value_iteration agree on V* to within theta;
raising `slip` visibly pushes the greedy arrows away from the trap.

---

## viz.py — build once, reuse for every block

```python
def plot_v(env, V, policy=None, title="")   # heatmap + greedy arrows; walls/G/T marked
def plot_curves(runs: dict[str, np.ndarray], ylabel, title)  # label -> per-episode values,
                                            # averaged over seeds with shading
```

---

## tabular/mc.py, td.py, nstep.py — Block 2

Shared conventions, worth writing once in a small `tabular/common.py` if you like:

```python
def epsilon_greedy(Q, s, epsilon, rng) -> int
# every control algo returns the same shape:
#   (Q, stats)  where stats includes per-episode return and ||max_a Q - V*||_inf
#   (pass V* in from Block 1 for the error curve)
```

```python
# mc.py
def mc_control(env, n_episodes, gamma, epsilon, seed) -> (Q, stats)   # first-visit

# td.py — the three differ ONLY in their backup target; factor accordingly or not, your call
def sarsa(env, n_episodes, gamma, alpha, epsilon, seed) -> (Q, stats)
def q_learning(env, n_episodes, gamma, alpha, epsilon, seed) -> (Q, stats)
def expected_sarsa(env, n_episodes, gamma, alpha, epsilon, seed) -> (Q, stats)

# nstep.py
def nstep_sarsa(env, n, n_episodes, gamma, alpha, epsilon, seed) -> (Q, stats)
# n in {1, 2, 4, 8, inf}: n=1 must reproduce sarsa, n=inf must reproduce MC — that's the test
```

**Acceptance:** Q-learning's greedy path hugs the trap, SARSA's detours (cliff-walking
effect); error curves converge toward 0 against Block 1's V*.

---

## experiments/ — one script per question, all seeded

```
experiments/block1_dp.py        V* heatmap, slip sweep
experiments/block2_td_vs.py     Q-learning vs SARSA paths + error curves
experiments/block2_nstep.py     n sweep
experiments/block2_ablate.py    epsilon / alpha grids
```

Each: fix seeds (e.g. 5 seeds), save plots to `results/`, print a one-line conclusion.

---

## deep/dqn.py — Block 3, part 1 (15x15 grid)

```python
class QNetwork(nn.Module):       # obs -> Q-values [n_actions]; you choose obs encoding
                                 # (one-hot state, normalized (r,c), or 2D grid channels)
class ReplayBuffer:
    def push(self, s, a, r, s2, done)
    def sample(self, batch_size) -> batch

def train_dqn(env, cfg) -> (q_net, stats)
# cfg flags you'll ablate: use_replay: bool, use_target_net: bool, target_sync_every: int
```

**Acceptance:** full DQN learns; replay-off and target-net-off runs visibly degrade.

---

## deep/reinforce.py — Block 3, part 2: the road to PPO/GRPO

One file, four stages, each a small delta on the last. Keep all four runnable.

```python
class PolicyNet(nn.Module)       # obs -> action logits
class ValueNet(nn.Module)        # obs -> scalar V (enters at stage 2)

def collect_rollouts(env, policy, n_steps_or_episodes, seed) -> trajectories

def train_reinforce(env, cfg)            # stage 1: vanilla policy gradient, G_t returns
def train_reinforce_baseline(env, cfg)   # stage 2: subtract learned V(s) baseline
def train_actor_critic(env, cfg)         # stage 3: GAE(lambda) advantages
def train_ppo(env, cfg)                  # stage 4: clipped ratio objective,
                                         #   multiple epochs over the same rollout batch
```

**The GRPO punchline** (write this up in the README at the end): PPO needs ValueNet only
to build the advantage baseline. GRPO deletes the value net — sample a *group* of
episodes from the same start, use the group's mean return as the baseline, advantage =
(return − group mean) / group std, same clipped ratio loss. In LLM post-training the
"episode" is one sampled completion and the "group" is N completions of the same prompt.
Optional capstone: `train_grpo(env, cfg)` on the gridworld — it's ~20 lines of diff
from your PPO.

---

## Order of work (current position marked)

1. ▶ `_move` → `build_P` → hand-check P near the trap
2. policy_evaluation → policy_iteration → value_iteration → plot_v → slip sanity check
3. `reset`/`step` + agreement check vs P
4. SARSA / Q-learning / expected SARSA + error curves vs V*
5. MC control, n-step sweep, ablations
6. 15x15 grid, DQN + ablations
7. REINFORCE → baseline → A2C/GAE → PPO → (GRPO)
8. README with plots
```