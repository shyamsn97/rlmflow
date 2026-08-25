"""Share live Python state with one child while keeping another isolated.

This deterministic example uses a scripted model, so it needs no API key.

Run:
    python examples/control/delegation/reuse_repl.py
"""

from __future__ import annotations

from rlmflow import Flow, LLMUsage

ROOT_REPLY = """\
```python
shared = ["parent"]
parent_identity = id(shared)

shared_child = await launch_subagent(
    "Append 'shared-child' to the existing shared list, then return its id.",
    model="default",
    name="shared",
    reuse_repl=True,
)
isolated_child = await launch_subagent(
    "Report whether a global named 'shared' exists in your REPL.",
    model="default",
    name="isolated",
)
shared_identity = await shared_child.wait_for_result()
isolated_view = await isolated_child.wait_for_result()

finish(
    f"same_object={shared_identity == str(parent_identity)}; "
    f"values={shared}; isolated_sees_shared={isolated_view}"
)
```"""

SHARED_REPLY = """\
```python
shared.append("shared-child")
finish(str(id(shared)))
```"""

ISOLATED_REPLY = """\
```python
finish(str("shared" in globals()))
```"""

EXPECTED = "same_object=True; values=['parent', 'shared-child']; " "isolated_sees_shared=False"


class ScriptedLLM:
    """Return fixed cells so the runtime behavior is the only variable."""

    last_usage = LLMUsage(input_tokens=20, output_tokens=20)

    def chat(self, messages):
        query = messages[-1]["content"]
        if "Append 'shared-child'" in query:
            return SHARED_REPLY
        if "whether a global named 'shared'" in query:
            return ISOLATED_REPLY
        return ROOT_REPLY


def main() -> None:
    flow = Flow(ScriptedLLM())
    root = flow.start("Demonstrate explicit shared and isolated child REPLs.", max_depth=1)
    try:
        result = flow.run(root)
        print(result)
        assert result == EXPECTED
    finally:
        flow.runtime.close_repls()


if __name__ == "__main__":
    main()
