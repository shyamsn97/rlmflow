"""Queue children now and await their persistent handles on a later turn.

This deterministic example uses a scripted model, so it needs no API key.

Run:
    python examples/control/delegation/nonblocking.py
"""

from __future__ import annotations

from rlmflow import AgentConfig, Flow, LLMUsage

ROOT_REPLY = """\
```repl
alpha = launch_subagent(task="Return alpha.")
beta = launch_subagent(task="Return beta.")
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
        query = "\n\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        if "Return alpha." in query:
            return "```repl\nsubmit('alpha finished')\n```"
        if "Return beta." in query:
            return "```repl\nsubmit('beta finished')\n```"
        if "queued:" in query:
            return WAIT_REPLY
        return ROOT_REPLY


def main() -> None:
    flow = Flow(ScriptedLLM(), workers=3)
    root = flow.start(
        "Queue two children, then collect them on a later turn.",
        config=AgentConfig(max_depth=1, max_iters=3),
    )
    try:
        result = flow.run(root)
        child_results = {child.config.name: child.result() for child in root.sub_agents}
        print("root:", result)
        print("children:", child_results)
        assert result == "alpha finished, beta finished"
        assert child_results == {
            "child0": "alpha finished",
            "child1": "beta finished",
        }
    finally:
        flow.runtime.close_repls()


if __name__ == "__main__":
    main()
