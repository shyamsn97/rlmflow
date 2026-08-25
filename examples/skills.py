"""Skills as on-disk artifacts plus a dynamic prompt section.

This example demonstrates the intended shape for user-authored skills:

1. A skill is just a file on disk you choose, e.g.
   ``skills/numpy-linear-algebra/SKILL.md``. There is no hardcoded directory.
2. A prompt-builder callable section reads selected skill files at prompt
   render time, so edits show up on the next turn.
3. The agent runs with a real LLM client; pass ``--print-prompt`` to inspect
   the rendered prompt before the LLM call.

    export OPENAI_API_KEY=...
    python examples/skills.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from rlmflow import (
    AgentConfig,
    Flow,
    SystemPromptBuilder,
)
from rlmflow.llm import OpenAIClient

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir, save_example_graph  # noqa: E402

MAX_ITERS = 5


NUMPY_LINEAR_ALGEBRA_SKILL = """\
# NumPy Linear Algebra

Use this skill when the user asks for a concrete matrix/vector computation,
least-squares fit, eigenvalue calculation, or numerical verification.

## Instructions

- Use NumPy in the REPL for the actual arithmetic. Do not do matrix algebra by
  hand when code can compute it exactly enough.
- Print the key intermediate arrays or scalars so the trace is auditable.
- Use `np.linalg.solve` for square nonsingular systems and `np.linalg.lstsq`
  for overdetermined systems.
- Verify the answer by computing a residual, reconstruction, or direct
  substitution check.
- In the final answer, include the numeric result and the verification residual.

## Example

For a least-squares line `y = m*x + b` through points:

```python
import numpy as np

x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([1, 2, 2, 4], dtype=float)
A = np.column_stack([x, np.ones_like(x)])

coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
m, b = coeffs
pred = A @ coeffs
residual = y - pred
residual_norm = np.linalg.norm(residual)

print("A =", A)
print("coeffs =", coeffs)
print("pred =", pred)
print("residual =", residual)
print("residual_norm =", residual_norm)
```

Then report `m`, `b`, `pred`, and `residual_norm`.
"""


def install_example_skill(skills_dir: Path) -> Path:
    """Write a sample SKILL.md as a normal file on disk."""
    path = skills_dir / "skills" / "numpy-linear-algebra" / "SKILL.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(NUMPY_LINEAR_ALGEBRA_SKILL)
    return path


def skills_section(skill_paths: list[Path]):
    """Return a callable prompt section that reads skill files dynamically."""

    def render(flow, agent) -> str:
        sections: list[str] = []
        for path in skill_paths:
            if not path.exists():
                continue
            body = path.read_text().strip()
            sections.append(f"### `{path.name}`\n\n{body}")
        if not sections:
            return ""
        return (
            "Use these skills when they match the user's task. Each skill is a "
            "file loaded into the prompt for this run.\n\n" + "\n\n".join(sections)
        )

    return render


def build_flow(skills_dir: Path, *, model: str) -> Flow:
    """Create a flow whose prompt includes the installed skill."""
    skill_path = install_example_skill(skills_dir)
    flow = Flow(OpenAIClient(model=model), root_config=AgentConfig(max_iters=MAX_ITERS))
    prompt = SystemPromptBuilder()
    prompt.sections.add("skills", skills_section([skill_path]), title="Skills", before="tools")
    flow.system_prompt = prompt
    return flow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help="Directory where the sample SKILL.md is written (default: a temp dir).",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use.")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the rendered prompt before making the LLM call.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=example_run_dir("skills"),
        help="Save the final run here (default: examples/_runs/skills/).",
    )
    args = parser.parse_args()

    tmp = None
    if args.skills_dir is None:
        tmp = tempfile.TemporaryDirectory()
        skills_dir = Path(tmp.name)
    else:
        skills_dir = args.skills_dir

    try:
        flow = build_flow(skills_dir, model=args.model)

        query = (
            "Fit the least-squares line y = m*x + b through points "
            "(0, 1), (1, 2), (2, 2), (3, 4), then report m, b, the "
            "predicted values, and the L2 residual norm."
        )
        root = flow.start(query)

        print(f"Wrote skill artifact under: {skills_dir}")
        print("- skills/numpy-linear-algebra/SKILL.md")
        if args.print_prompt:
            print("\n--- rendered system prompt ---\n")
            print(flow.system_prompt.render(flow, root))
            print("\n--- live run ---\n")

        async def drive() -> None:
            async for node in flow.run_streaming(root):
                print(f"{node.parent_agent.config.path}  {node.type}")
                root.save(args.out_dir)

        asyncio.run(drive())
        print(root.result())
        save_example_graph(root, "skills", out_dir=args.out_dir)
        flow.runtime.close_repls()
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
