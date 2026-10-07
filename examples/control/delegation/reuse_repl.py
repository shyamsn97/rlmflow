"""Let Flow place delegated children in the parent's worker with isolated state.

This deterministic example uses a scripted model, so it needs no API key.

Run:
    python examples/control/delegation/reuse_repl.py
"""

from __future__ import annotations

from rlmflow import AgentConfig, Flow, LLMUsage

ROOT_REPLY = """\
```repl
shared = ["parent"]
parent_pid = os.getpid()

worker_child = launch_subagent(
    task="Report whether you share the parent worker but not its Python globals.",
    parent_pid=parent_pid,
)
report = await worker_child

submit(f"{report}; parent_values={shared}")
```"""

SHARED_REPLY = """\
```repl
submit(
    f"same_worker={os.getpid() == parent_pid}; "
    f"isolated={'shared' not in globals()}"
)
```"""

EXPECTED = "same_worker=True; isolated=True; parent_values=['parent']"


class ScriptedLLM:
    """Return fixed cells so the runtime behavior is the only variable."""

    last_usage = LLMUsage(input_tokens=20, output_tokens=20)

    def chat(self, messages):
        query = "\n\n".join(
            message["content"] for message in messages if message["role"] == "user"
        )
        if "Report whether" in query:
            return SHARED_REPLY
        return ROOT_REPLY


def main() -> None:
    flow = Flow(ScriptedLLM(), delegation_reuse_worker=True)
    root = flow.start(
        "Demonstrate isolated child namespaces in one worker.",
        config=AgentConfig(max_depth=1, max_iters=3),
    )
    try:
        result = flow.run(root)
        print(result)
        assert result == EXPECTED
    finally:
        flow.runtime.close_repls()


if __name__ == "__main__":
    main()
