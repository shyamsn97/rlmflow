"""Dynamically add skills to a running agent via a live prompt section.

Two pieces work together here:

1. A **callable prompt section**. ``Flow`` re-renders the system prompt every
   turn (``build_system_prompt`` -> ``prompt_builder.build``), so a section whose
   body is a function reflects whatever it reads *right now*. Point it at a
   mutable ``SkillLibrary`` and the rendered "Skills" section is always live.
2. An **``add_skill`` tool**. A plain ``@tool``-decorated function passed via
   ``tools=[...]`` lands in the REPL namespace and auto-documents itself in the
   prompt. Because it closes over the same ``SkillLibrary``, the agent can
   install a new skill mid-run and *see it in its own prompt on the next turn*.

That is the self-extending pattern: give the model a tool that writes skills,
and the prompt teaches it those skills one turn later.

    export OPENAI_API_KEY=...
    python examples/dynamic_skills.py --model gpt-4o-mini --print-prompt

With no ``OPENAI_API_KEY`` set, the run uses ``ScriptedLLM``: a deterministic
offline client that drives the *same* agent loop. It installs the skill on turn
one and, seeing that skill now in its own prompt, solves the task on turn two.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from rflow import DEFAULT_BUILDER, Flow, Graph, LLMUsage
from rflow.clients import OpenAIClient
from rflow.tools import tool


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


@dataclass(frozen=True)
class Skill:
    name: str
    body: str


class SkillLibrary:
    """A mutable set of skills the prompt section reads on every render."""

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills or []:
            self.add(skill.name, skill.body)

    def add(self, name: str, body: str) -> None:
        self._skills[name] = Skill(name, body.strip())

    def names(self) -> list[str]:
        return list(self._skills)

    def render(self) -> str:
        skills = list(self._skills.values())
        if not skills:
            return (
                "No skills installed yet. Call `add_skill(name, body)` to teach "
                "yourself one; it will appear here on your next turn."
            )
        blocks = [f"### {skill.name}\n\n{skill.body}" for skill in skills]
        return (
            "Use these skills when they match the task. Each was installed into "
            "this run via `add_skill` and is now part of your prompt.\n\n"
            + "\n\n".join(blocks)
        )

    def __len__(self) -> int:
        return len(self._skills)


def make_add_skill(library: SkillLibrary):
    """Build the ``add_skill`` tool bound to a specific library."""

    @tool(
        "Install a reusable skill (a name plus a markdown body of guidance) into "
        "your own skill library. It appears in the Skills section of your system "
        "prompt starting next turn."
    )
    def add_skill(name: str, body: str) -> str:
        library.add(name, body)
        return (
            f"installed skill {name!r} "
            f"(library now has {len(library)}: {library.names()})"
        )

    return add_skill


KADANE_SKILL = """\
Kadane's algorithm finds the maximum-sum contiguous subarray in O(n): scan left
to right keeping best_ending_here = max(x, best_ending_here + x) and track the
running maximum of that quantity. Never carry a negative running sum into the
next element. Verify small cases by brute force over all subarrays.
"""

QUERY = (
    "You can extend yourself with `add_skill(name, body)`, which writes a skill "
    "into your own system prompt (visible next turn). First, install a skill "
    "named 'kadane' whose body explains, in a short paragraph, Kadane's algorithm "
    "for the maximum contiguous subarray sum. Then apply it: compute the maximum "
    "subarray sum of [-2, 1, -3, 4, -1, 2, 1, -5, 4] in the REPL, verify it by "
    "brute force over all contiguous subarrays, and report the value."
)


class ScriptedLLM:
    """Deterministic offline client that drives the add_skill -> solve loop.

    It keys its reply on the *incoming system prompt*: turn one has an empty
    Skills section, so it installs the skill; turn two sees that skill now in
    the prompt (proof the section grew), so it solves and calls ``done``.
    """

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=64, output_tokens=16)
        system = messages[0]["content"] if messages else ""
        if "### kadane" not in system:
            return (
                "```repl\n"
                f'add_skill("kadane", """{KADANE_SKILL}""")\n'
                'print("installed kadane; it will be in my prompt next turn")\n'
                "```"
            )
        return (
            "```repl\n"
            "nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]\n"
            "best = cur = nums[0]\n"
            "for x in nums[1:]:\n"
            "    cur = max(x, cur + x)\n"
            "    best = max(best, cur)\n"
            "brute = max(\n"
            "    sum(nums[i:j])\n"
            "    for i in range(len(nums))\n"
            "    for j in range(i + 1, len(nums) + 1)\n"
            ")\n"
            "assert best == brute, (best, brute)\n"
            'print("kadane =", best, "brute =", brute)\n'
            "done(str(best))\n"
            "```"
        )


def build_flow(library: SkillLibrary, llm, *, max_iters: int) -> Flow:
    flow = Flow(llm, max_iters=max_iters, tools=[make_add_skill(library)])
    flow.prompt_builder = DEFAULT_BUILDER.section(
        "skills",
        lambda flow, graph: library.render(),
        title="Skills",
        before="tools",
    )
    return flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use.")
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the Skills section before and after the run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_example_run_dir(__file__, "dynamic-skills"),
        help="Save the final run here (default: examples/_runs/dynamic-skills/).",
    )
    args = parser.parse_args()

    library = SkillLibrary()
    if os.environ.get("OPENAI_API_KEY"):
        llm = OpenAIClient(model=args.model)
        print(f"Using live OpenAI client ({args.model}).")
    else:
        llm = ScriptedLLM()
        print("No OPENAI_API_KEY - using the deterministic ScriptedLLM offline.")
    flow = build_flow(library, llm, max_iters=args.max_iters)

    graph = Graph(query=QUERY)
    if args.print_prompt:
        print("\n--- Skills section BEFORE run (empty) ---\n")
        print(library.render())

    print("\n--- run ---\n")
    flow.run(graph)

    print(f"\nagent installed skills: {library.names()}")
    if args.print_prompt:
        print("\n--- Skills section AFTER run (grown by the agent) ---\n")
        print(library.render())

    print("\nresult:", graph.result())
    path = graph.save(args.out_dir)
    print(f"Graph saved to {path}")
    flow.close_repls(graph.graph_id)


if __name__ == "__main__":
    main()
