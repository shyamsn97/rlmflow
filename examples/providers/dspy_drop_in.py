"""Use a Flow agent as the LM behind a DSPy program.

`DSPyFlow` wraps `Flow` with DSPy's LM interface, so every DSPy "LM call"
becomes a full recursive agent run.

Run with:
    export OPENAI_API_KEY=...
    pip install -e ".[openai,dspy]"
    python examples/providers/dspy_drop_in.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import dspy

from rlmflow import AgentConfig, Flow
from rlmflow.adapters import DSPyFlow, FlowLLM
from rlmflow.llm import OpenAIClient

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import save_example_graph  # noqa: E402


def main() -> None:
    agent = FlowLLM(
        Flow(
            OpenAIClient(model="gpt-4o-mini"),
            root_config=AgentConfig(max_depth=1, max_iters=5),
        )
    )

    dspy.configure(lm=DSPyFlow(agent, model="rlmflow/gpt-4o-mini"))

    qa = dspy.ChainOfThought("question -> answer")
    result = qa(question="What is 17 * 23? Show a short calculation.")
    print(result.answer)
    if agent.last_graph is not None:
        save_example_graph(agent.last_graph, "dspy-drop-in")

    agent.close()


if __name__ == "__main__":
    main()
