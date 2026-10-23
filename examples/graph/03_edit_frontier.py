"""Send controller feedback to a finished agent, then branch from an earlier node."""

from __future__ import annotations

from rlmflow import DoneOutput, Graph, StaleFrontier, UserQuery, ops, reads


def main() -> None:
    g = Graph()
    root = ops.spawn(g, "draft a release plan")
    first_answer = ops.append(g, root, DoneOutput(result="ship immediately"))

    # A finished agent becomes active again when it receives a message. History
    # stays intact; the new turn extends the chain.
    ops.send(g, root, "Account for the failed canary first.")
    ops.deliver(g, root)
    ops.append(g, root, DoneOutput(result="rollback, fix, then retry the canary"))

    print("continued:", reads.result(g, root))
    print("history:", [node.type for node in reads.history(g, root)])

    # To change an earlier decision, fork there. The original run is untouched,
    # and the branch shares everything up to the fork point.
    branch = ops.fork(g, first_answer)
    ops.append(g, branch, UserQuery(content="Prefer the lowest-risk release plan."))
    ops.append(g, branch, DoneOutput(result="hold the release"))

    print("original:", reads.result(g, root))
    print("branch:", reads.result(g, branch))

    # Appending after stale history is rejected instead of silently rewriting it.
    try:
        ops.append(g, root, UserQuery(content="This is not the frontier."), after=first_answer)
    except StaleFrontier as error:
        print("stale edit rejected:", error)


if __name__ == "__main__":
    main()
