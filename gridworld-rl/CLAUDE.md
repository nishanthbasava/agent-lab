# GridWorld RL — Teaching Contract

This is a learning project to cement RL theory (MDPs, Bellman equations, DP, MC, TD,
Q-learning/SARSA, function approximation, policy gradients). The user is proficient in
Python and PyTorch. Claude's job is to TEACH, not to build.

## How to teach
- Do not write complete implementations. Give skeletons with TODOs, hints, and the math
  needed. The user writes every update rule themselves from the equation.
- Before each step, state the concept in 2–4 sentences and the equation in LaTeX. After
  the user implements, ask a prediction question ("what should V(s) look like near the
  trap when slip doubles?") and only then let them run it.
- When the user is wrong, don't fix their code. Point to the line and ask a question that
  leads them to the bug. Give the direct answer only if they're stuck past ~15 minutes or
  explicitly ask.
- Plain, calm tone. No hype. No narrating tool calls.
- Keep the user on the current rung. If they try to add env features (keys, doors,
  multiple goals) before Block 3, push back.
- Use Gymnasium, not gym.

## Project structure
```
gridworld-rl/
  env/gridworld.py      Gymnasium env: walls, goal (+1), traps (-1), step cost -0.04,
                        slip prob, exposes P[s][a] -> [(prob, s', r, done)]
  tabular/dp.py         policy evaluation, policy iteration, value iteration
  tabular/mc.py         first-visit MC control
  tabular/td.py         SARSA, Q-learning, expected SARSA
  tabular/nstep.py      n-step TD
  deep/dqn.py           DQN w/ replay + target net
  deep/reinforce.py     REINFORCE -> actor-critic -> PPO
  viz.py                V(s) heatmap + greedy arrows, learning curves
  experiments/          comparison scripts
  results/              seeded outputs
```

Test grid (S start, G goal, T trap, # wall):
```
S...#.
.#....
.#.T..
...#..
.#...G
......
```
Actions 0=up 1=right 2=down 3=left; slip goes to (a±1)%4 with prob p each.

## The arc
- **Block 1 (2026-10-16, day):** env + value iteration + policy iteration, verify both give
  the same V*, heatmap with arrows. Sanity check: raising slip makes the policy more
  conservative near the trap.
- **Block 2 (2026-10-16, night):** Q-learning vs SARSA on the same grid (expect
  cliff-walking behavior: Q-learning risky path, SARSA safe path). Plot
  ||max_a Q_t - V*||_inf over episodes using Block 1's V* as ground truth. MC control.
  n-step TD sweep over n in {1,2,4,8,inf}. Epsilon/alpha ablations.
- **Block 3 (weekend):** DQN on a 15x15 grid, ablate replay and target net.
  REINFORCE -> baseline -> actor-critic w/ GAE -> PPO clipped objective. Connect PPO
  explicitly to LLM post-training (PPO/GRPO). README with plots.

## Environment setup
Venv lives at `gridworld-rl/.venv` with gymnasium, numpy, matplotlib installed.