"""A symbolic stand-in for the paper's office kitchen (Sec 5).

The real system grounds skills in camera images + a mobile manipulator; here the
whole world is one small dataclass. Other representations you could swap in:
a set of predicates like {"on(apple, table)", "at(robot, sink)"} (PDDL-style),
or a natural-language paragraph handed straight to the LLM. The dataclass keeps
the "Can" rules in skills.py trivial to write, which is the point of the demo.
"""

from dataclasses import dataclass, field


@dataclass
class World:
    locations: tuple        # places the robot can be
    robot_at: str
    holding: str | None     # at most one object in the gripper
    objects: dict           # object name -> location name

    def describe(self):
        held = self.holding or "nothing"
        objs = ", ".join(f"{o} ({loc})" for o, loc in self.objects.items()) or "none"
        return f"robot at the {self.robot_at}, holding {held} | objects: {objs}"


# A scene = initial state. Each is a tiny story the planner can resolve.
SCENES = {
    # The paper's headline example: sponge is elsewhere, spill is on the counter.
    "spill": World(
        locations=("counter", "sink", "table", "trash can"),
        robot_at="table",
        holding=None,
        objects={"spilled coke": "counter", "sponge": "sink", "apple": "table"},
    ),
    # Fetch task: "bring me the apple" (user is at the table).
    "apple": World(
        locations=("counter", "table", "cabinet"),
        robot_at="counter",
        holding=None,
        objects={"apple": "cabinet", "sponge": "table"},
    ),
    # "I'm hungry" — ambiguous instruction, the LLM must pick a reasonable goal.
    "snack": World(
        locations=("counter", "table", "cabinet"),
        robot_at="table",
        holding=None,
        objects={"bag of chips": "cabinet", "plate": "counter"},
    ),
    # Tidy-up: object needs to end up at the trash can.
    "trash": World(
        locations=("counter", "table", "trash can"),
        robot_at="counter",
        holding=None,
        objects={"empty coke can": "table", "sponge": "counter"},
    ),
}
