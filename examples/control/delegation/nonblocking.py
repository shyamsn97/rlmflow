"""Queue children now and await their persistent handles on a later turn.

This deterministic example uses a scripted model, so it needs no API key.

Run:
    python examples/control/delegation/nonblocking.py
"""

from __future__ import annotations

from rlmflow import AgentConfig, Flow, Graph, LLMUsage

ROOT_REPLY = """\
```repl
alpha = launch_agent("Return alpha.", name="alpha")
beta = launch_agent("Return beta.", name="beta")
print("queued:", alpha.name, beta.name)
```"""

WAIT_REPLY = """\
```repl
alpha_result = await alpha
beta_result = await beta
submit(f"{alpha_result}, {beta_result}")
```"""


class ScriptedLLM:
    last_usage = LLMUsage(input_tokens=20, output_tokens=10)

    def chat(self, messages):
        query = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
        if "Return alpha." in query:
            return "```repl\nsubmit('alpha finished')\n```"
        if "Return beta." in query:
            return "```repl\nsubmit('beta finished')\n```"
        if "queued:" in query:
            return WAIT_REPLY
        return ROOT_REPLY


def main() -> None:
    flow = Flow(ScriptedLLM(), workers=3)
    g = Graph()
    root = g.create_agent("Queue two children, then collect them on a later turn.")
    try:
        flow.run(g, root_config=AgentConfig(max_iters=3))
        result = g.result(root)
        child_results = {child.name: g.result(child) for child in g.children(root)}
        print("root:", result)
        print("children:", child_results)
        assert result == "alpha finished, beta finished"
        assert child_results == {
            "alpha": "alpha finished",
            "beta": "beta finished",
        }
    finally:
        flow.runtime.close()


if __name__ == "__main__":
    main()
