"""The SayCan planning loop (Sec 3, Fig 2 / Alg 1):

    repeat: score all skills with say * can -> execute argmax -> until "done".

Example:
    python run.py --scene spill --instruction "I spilled my coke, can you help?"
"""

import argparse
import pathlib

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from say import say_scores
from skills import build_skills
from world import SCENES

load_dotenv(pathlib.Path(__file__).parents[1] / ".env")  # ANTHROPIC_API_KEY

TOP_K = 8  # rows shown per step; all candidates are scored


def step_table(step, ranked, chosen_name):
    table = Table(title=f"step {step}", title_justify="left")
    for col in ("skill", "say", "can", "combined"):
        table.add_column(col, justify="right" if col != "skill" else "left")
    for name, say, can, combined in ranked[:TOP_K]:
        mark = " ◀" if name == chosen_name else ""
        style = "bold green" if name == chosen_name else ("dim" if can == 0 else "")
        table.add_row(name + mark, f"{say:.2f}", f"{can:.1f}", f"{combined:.2f}", style=style)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    console = Console()
    client = anthropic.Anthropic()
    world = SCENES[args.scene]
    plan = []

    console.print(Panel(f'[bold]"{args.instruction}"[/bold]\n{world.describe()}',
                        title=f"scene: {args.scene}"))

    for step in range(1, args.max_steps + 1):
        # Rebuild each step: held objects lose their pick-up skill, etc.
        skills = {s.name: s for s in build_skills(world)}

        say = say_scores(client, args.instruction, world.describe(), plan, list(skills))
        ranked = sorted(
            ((name, say[name], s.can(world), say[name] * s.can(world))
             for name, s in skills.items()),
            key=lambda r: r[3], reverse=True,
        )
        chosen_name = ranked[0][0]

        console.print(step_table(step, ranked, chosen_name))
        world = skills[chosen_name].effect(world)
        plan.append(chosen_name)
        console.print(f"  state: {world.describe()}\n")

        if chosen_name == "done":
            break
    else:
        console.print("[yellow]max steps reached without 'done'[/yellow]")

    console.print(Panel("\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)),
                        title="final plan"))


if __name__ == "__main__":
    main()
