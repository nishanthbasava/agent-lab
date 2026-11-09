# saycan

A small architectural demo of SayCan (Ahn et al., 2022, "Do As I Can, Not As I Say:
Grounding Language in Robotic Affordances", https://arxiv.org/abs/2204.01691).
No robots, no training, no sim — a symbolic kitchen world + the Anthropic API.

The paper's one idea: an LLM knows *what would be useful* ("Say") but not *what the
robot can actually do right now* ("Can"). Score every candidate skill with both,
multiply, take the argmax, repeat.

```
                 instruction ("I spilled my coke...")
                        │
        ┌───────────────┴────────────────┐
        │   for each skill in library:   │◄────────────┐
        │                                │             │
        │  say(skill)   ── Claude: "how useful         │
        │                   as the next step?"         │
        │  can(skill)   ── rules on symbolic state:    │
        │                   "possible right now?"      │
        │                                │             │
        │  combined = say * can          │             │
        └───────────────┬────────────────┘             │
                        │ argmax                       │
                 apply skill's effect ── updated state ┘
                        │
              until "done" or max steps
```

## Files → paper sections

| file        | paper                                                            |
|-------------|------------------------------------------------------------------|
| `world.py`  | the environment — the office-kitchen setting of Sec 5, reduced to a symbolic dataclass |
| `skills.py` | the skill library + affordances ("Can", Sec 3) — the paper's learned value functions become precondition rules here |
| `say.py`    | the LLM scoring ("Say", Sec 3) — the paper's token log-likelihoods become structured-JSON usefulness scores (see the comment in the file) |
| `run.py`    | the planning loop (Sec 3, Fig 2 / Alg 1): combine, argmax, append, repeat |

## Run

Uses the root venv (`anthropic`, `python-dotenv`, `rich` already installed) and
`ANTHROPIC_API_KEY` from `agent-lab/.env`.

```bash
cd saycan
../.venv/bin/python run.py --scene spill --instruction "I spilled my coke, can you help?"
```

Scenes: `spill`, `apple`, `snack`, `trash` (see `world.py`). Each step prints a
table of the top candidates with say / can / combined scores, the chosen skill,
and the updated world state; the full plan prints at the end.
