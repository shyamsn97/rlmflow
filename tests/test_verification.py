from helpers import StubLLM, counting_replies

from rlmflow import (
    AgentStart,
    DoneOutput,
    Flow,
    LocalRuntime,
    RepairQuery,
    VerificationFailure,
    start,
)


def repl(code: str) -> str:
    return f"```repl\n{code}\n```"


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
                repl("finish({'ok': False})"),
                repl("finish({'ok': True})"),
                repl("finish({'accepted': True})"),
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
