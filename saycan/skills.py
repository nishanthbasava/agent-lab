"""The skill library + the "Can" side (Sec 3).

In the paper each skill is a trained policy, and "Can" is a learned value
function: the expected probability that running the skill *from this state*
succeeds. Here the rule-based `can(world)` stands in for that value function —
it returns 1.0 when the preconditions hold and 0.0 otherwise (a real value
function would be graded, e.g. 0.6 for "sponge visible but far away").
The product say * can is why SayCan never picks "pick up the sponge" from
across the room, no matter how useful the LLM thinks it is.
"""

from dataclasses import dataclass, replace
from typing import Callable

from world import World


@dataclass(frozen=True)
class Skill:
    name: str                          # the text the LLM scores — this IS the interface
    can: Callable[[World], float]      # affordance: feasible from this state?
    effect: Callable[[World], World]   # symbolic state transition if executed


def build_skills(world: World) -> list[Skill]:
    """Instantiate the library for a scene, like the paper's 551 skills
    generated from (verb, object) combinations — just much smaller."""
    skills = []

    for loc in world.locations:
        skills.append(Skill(
            name=f"go to the {loc}",
            can=lambda w, loc=loc: 0.0 if w.robot_at == loc else 1.0,
            effect=lambda w, loc=loc: replace(w, robot_at=loc),
        ))

    for obj in world.objects:
        # Pick up: must be co-located with the object, gripper free.
        skills.append(Skill(
            name=f"pick up the {obj}",
            can=lambda w, o=obj: 1.0 if w.holding is None and w.objects.get(o) == w.robot_at else 0.0,
            effect=lambda w, o=obj: replace(
                w, holding=o, objects={k: v for k, v in w.objects.items() if k != o}),
        ))
        # Put down: must be holding exactly this object.
        skills.append(Skill(
            name=f"put down the {obj}",
            can=lambda w, o=obj: 1.0 if w.holding == o else 0.0,
            effect=lambda w, o=obj: replace(
                w, holding=None, objects={**w.objects, o: w.robot_at}),
        ))

    # One composite skill so the spill scene has a satisfying ending.
    if any("spill" in o for o in world.objects):
        spill = next(o for o in world.objects if "spill" in o)
        skills.append(Skill(
            name="wipe up the spill",
            can=lambda w: 1.0 if w.holding == "sponge" and w.objects.get(spill) == w.robot_at else 0.0,
            effect=lambda w: replace(w, objects={k: v for k, v in w.objects.items() if k != spill}),
        ))

    # "done" is a real skill in the paper too — the LLM decides termination
    # by ranking it above everything else once the instruction is satisfied.
    skills.append(Skill(name="done", can=lambda w: 1.0, effect=lambda w: w))
    return skills
