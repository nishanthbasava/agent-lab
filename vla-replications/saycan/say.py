"""The "Say" side (Sec 3): how useful is each skill as the next step?

HOW THIS DIFFERS FROM THE PAPER: SayCan scores skill s as p(s | prompt) — the
LLM's *generative log-likelihood* of the skill's text continuing a prompt like
"Robot: I would 1. ...". That needs token logprobs, which the Anthropic API
doesn't expose. So instead we ask Claude to act as a *critic*: one call rates
every candidate with a usefulness score in [0, 1], returned as structured JSON.
Two consequences worth noticing while reading Sec 3:
  - the paper's scores are a normalized distribution over skills; ours are
    independent judgments (they needn't sum to 1),
  - a critic can reason about the whole candidate list at once, which the
    likelihood trick cannot — arguably easier, and a sign of how much stronger
    instruction-tuned models are than the 2022 PaLM used in the paper.
"""

import anthropic

MODEL = "claude-opus-5"

SYSTEM = """You are the task-grounding half of a SayCan-style robot planner.
Given a human instruction, the steps already taken, and a list of candidate
skills, rate how useful each skill would be as the IMMEDIATE next step toward
completing the instruction. 1.0 = clearly the right next step, 0.0 = useless or
counterproductive. Rate usefulness only — ignore whether the skill is currently
feasible; a separate affordance model handles that. When the instruction is
already satisfied by the steps taken, give 'done' the highest score."""


def say_scores(client: anthropic.Anthropic, instruction: str, scene: str,
               steps_so_far: list[str], skill_names: list[str]) -> dict[str, float]:
    """One API call -> {skill name: score in [0, 1]} for every candidate.

    Note we pass a scene description. The paper's LLM is state-blind — grounding
    comes entirely from the value functions, which are graded and state-aware
    (e.g. "go to the counter" scores higher when the sponge task needs it).
    Our rule-based "can" is binary, so it can't break ties between feasible
    navigation skills; without the scene text the planner literally cannot know
    which location the spill is at and oscillates. The paper's own "describing
    the scene" variant (Sec 5.4) adds exactly this kind of text to the prompt."""
    # Constrain the output to exactly one number per skill, so there is no
    # parsing step and no skill can be silently skipped.
    schema = {
        "type": "object",
        "properties": {name: {"type": "number"} for name in skill_names},
        "required": skill_names,
        "additionalProperties": False,
    }
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_so_far)) or "(none yet)"
    prompt = (
        f"Instruction: {instruction}\n\n"
        f"Current scene: {scene}\n\n"
        f"Steps taken so far:\n{steps}\n\n"
        "Candidate skills to rate:\n" + "\n".join(f"- {n}" for n in skill_names)
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    import json
    scores = json.loads(next(b.text for b in response.content if b.type == "text"))
    # The schema can't enforce numeric ranges (min/max unsupported), so clamp.
    return {name: min(1.0, max(0.0, float(scores[name]))) for name in skill_names}
