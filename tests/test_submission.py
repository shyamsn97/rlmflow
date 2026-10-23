from helpers import StubLLM, agent_graph, counting_replies, repl, run_result, stepping

from rlmflow import (
    DoneOutput,
    Flow,
    LocalRuntime,
    RepairQuery,
    VerificationFailure,
)
from rlmflow.graph import reads
from rlmflow.tools import format_tool_docs
from rlmflow.tools.builtins import builtin_tools


def test_immediate_submission_adds_no_model_turn():
    llm = StubLLM(counting_replies(repl("submit('answer')")))
    flow = Flow(llm, runtime=LocalRuntime(repl_timeout=5))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root, close_repls=True) == "answer"
    assert sum(node.type == "llm_output" for node in reads.history(g, root)) == 1
    assert not any(node.type == "submission" for node in reads.history(g, root))
    assert isinstance(reads.frontier(g, root), DoneOutput)


def test_stdout_submission_is_accepted_without_an_extra_turn():
    llm = StubLLM(counting_replies(repl("print('count=4')\nsubmit('Answer: 4')")))
    flow = Flow(llm, runtime=LocalRuntime(repl_timeout=5))
    g, root = agent_graph("Count the records.")

    assert run_result(flow, g, root, close_repls=True) == "Answer: 4"
    assert sum(node.type == "llm_output" for node in reads.history(g, root)) == 1


def test_finish_remains_a_hidden_compatibility_alias():
    flow = Flow(
        StubLLM(counting_replies(repl("finish('legacy')"))),
        runtime=LocalRuntime(repl_timeout=5),
    )
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root, close_repls=True) == "legacy"
    with stepping(g, root):
        docs = format_tool_docs(builtin_tools(flow, reads.frontier(g, root)))
    assert "submit" in docs
    assert "finish" not in docs


def test_worker_owned_async_verifier_repairs_the_same_child():
    flow = Flow(
        StubLLM(
            counting_replies(
                repl(
                    "async def valid(value):\n"
                    "    await asyncio.sleep(0)\n"
                    "    return None if value.get('ok') is True else 'set ok to true'\n"
                    "child = launch_subagent("
                    "task='return an accepted object', verify=valid)\n"
                    "print(await child)"
                ),
                repl("submit({'ok': False})"),
                repl("submit({'ok': True})"),
                repl("submit({'accepted': True})"),
            )
        ),
        runtime=LocalRuntime(repl_timeout=5),
    )
    g, root = agent_graph("delegate", max_depth=1)

    assert run_result(flow, g, root, close_repls=True) == {"accepted": True}

    children = reads.children(g, root)
    assert len(children) == 1
    child = children[0]
    failure = next(
        node for node in reads.history(g, child) if isinstance(node, VerificationFailure)
    )
    assert failure.candidate == {"ok": False}
    assert failure.kind == "verifier"
    assert failure.feedback == "set ok to true"
    assert failure.verifier_hash == child.config.verifier_hash
    assert isinstance(
        reads.history(g, child)[reads.history(g, child).index(failure) + 1], RepairQuery
    )
    accepted = reads.frontier(g, child)
    assert isinstance(accepted, DoneOutput)
    assert accepted.result == {"ok": True}
    assert accepted.verified is True
    assert accepted.verifier_hash == child.config.verifier_hash
