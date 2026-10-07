import asyncio

from helpers import StubLLM, counting_replies, repl

from rlmflow import (
    AgentStart,
    DoneOutput,
    Flow,
    LocalRuntime,
    RepairQuery,
    Submission,
    VerificationFailure,
    persistence,
    start,
)
from rlmflow.engine.steps import accept_submission
from rlmflow.engine.transitions import TransitionFunction
from rlmflow.tools import format_tool_docs
from rlmflow.tools.builtins import builtin_tools


def test_immediate_submission_adds_no_model_turn():
    llm = StubLLM(counting_replies(repl("submit('answer')")))
    flow = Flow(llm, runtime=LocalRuntime(repl_timeout=5))
    root = start("Answer.")

    assert flow.run(root, close_repls=True) == "answer"
    assert sum(node.type == "llm_output" for node in root.transcript()) == 1
    assert any(isinstance(node, Submission) for node in root.transcript())
    assert isinstance(root.frontier, DoneOutput)


def test_stdout_submission_is_accepted_without_an_extra_turn():
    llm = StubLLM(counting_replies(repl("print('count=4')\nsubmit('Answer: 4')")))
    flow = Flow(llm, runtime=LocalRuntime(repl_timeout=5))
    root = start("Count the records.")

    assert flow.run(root, close_repls=True) == "Answer: 4"
    assert sum(node.type == "llm_output" for node in root.transcript()) == 1


def test_finish_remains_a_hidden_compatibility_alias():
    flow = Flow(
        StubLLM(counting_replies(repl("finish('legacy')"))),
        runtime=LocalRuntime(repl_timeout=5),
    )
    root = start("Answer.")

    assert flow.run(root, close_repls=True) == "legacy"
    docs = format_tool_docs(builtin_tools(flow, root))
    assert "submit" in docs
    assert "finish" not in docs


def test_persisted_submission_resolves_without_an_llm_call(tmp_path):
    root = start("query")
    submission = Submission(
        content="count=4",
        result="answer",
        schema_hash="schema",
        verifier_hash="verifier",
        verified=True,
    )
    root.append(submission)
    root.save(tmp_path / "run")

    loaded = persistence.load(tmp_path / "run")
    assert isinstance(loaded.frontier, Submission)
    done = asyncio.run(Flow(object()).step(loaded.frontier))
    assert isinstance(done, DoneOutput)
    assert (
        done.content,
        done.result,
        done.schema_hash,
        done.verifier_hash,
        done.verified,
    ) == ("count=4", "answer", "schema", "verifier", True)


def test_submission_acceptance_is_a_transition_function():
    assert isinstance(accept_submission, TransitionFunction)
    assert accept_submission.sources == (Submission,)


def test_submission_nodes_complete_in_flight_after_budget_is_spent():
    flow = Flow(object())
    root = start("query", max_budget=0)
    submission = Submission(result="answer")
    root.append(submission)

    assert flow.budget_exceeded(submission) is False


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
    root = start("delegate", max_depth=1)

    assert flow.run(root, close_repls=True) == {"accepted": True}

    children = [node for node in root.walk() if isinstance(node, AgentStart)][1:]
    assert len(children) == 1
    child = children[0]
    failure = next(
        node for node in child.transcript() if isinstance(node, VerificationFailure)
    )
    assert failure.candidate == {"ok": False}
    assert failure.kind == "verifier"
    assert failure.feedback == "set ok to true"
    assert failure.verifier_hash == child.config.verifier_hash
    assert isinstance(failure.next, RepairQuery)
    accepted = child.frontier
    assert isinstance(accepted, DoneOutput)
    assert accepted.result == {"ok": True}
    assert accepted.verified is True
    assert accepted.verifier_hash == child.config.verifier_hash
