"""Queue children now and await their persistent handles on a later turn.

This deterministic example uses a scripted model, so it needs no API key.

Run:
    python examples/control/delegation/nonblocking.py
"""

from __future__ import annotations

from rlmflow import Flow, LLMUsage

ROOT_REPLY = """\
```python
alpha = await launch_subagent("Return alpha.", model="default", name="alpha")
beta = await launch_subagent("Return beta.", model="default", name="beta")
print("queued:", alpha.name, beta.name)
```"""

WAIT_REPLY = """\
```python
alpha_result = await alpha.wait_for_result()
beta_result = await beta.wait_for_result()
finish(f"{alpha_result}, {beta_result}")
```"""


class ScriptedLLM:
    last_usage = LLMUsage(input_tokens=20, output_tokens=10)

    def chat(self, messages):
        query = messages[-1]["content"]
        if "Return alpha." in query:
            return "```python\nfinish('alpha finished')\n```"
        if "Return beta." in query:
            return "```python\nfinish('beta finished')\n```"
        if "queued:" in query:
            return WAIT_REPLY
        return ROOT_REPLY


def main() -> None:
    flow = Flow(ScriptedLLM(), workers=3)
    root = flow.start("Queue two children, then collect them on a later turn.", max_depth=1)
    try:
        result = flow.run(root)
        child_results = {child.config.name: child.result() for child in root.sub_agents}
        print("root:", result)
        print("children:", child_results)
        assert result == "alpha finished, beta finished"
        assert child_results == {
            "alpha": "alpha finished",
            "beta": "beta finished",
        }
    finally:
        flow.runtime.close_repls()


if __name__ == "__main__":
    main()
