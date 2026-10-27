"""Send controller feedback to a finished agent, then branch from an earlier node."""

from __future__ import annotations

from rlmflow import CODE, DONE, Graph, Node, StaleFrontier


def main() -> None:
    g = Graph()
    root = g.spawn("draft a release plan", name="root")
    first_answer = g.append(root, Node(DONE, result="ship immediately"))

    # A finished agent becomes active again when it receives a message. History
    # stays intact; the new turn extends the chain.
    g.send(root, "Account for the failed canary first.")
    g.deliver(root)
    g.append(root, Node(DONE, result="rollback, fix, then retry the canary"))

    print("continued:", g.result(root))
    print("history:", [node.type for node in g.history(root)])

    # To change an earlier decision, fork there. The original run is untouched,
    # and the branch shares everything up to the fork point.
    branch = g.fork(first_answer)
    g.send(branch, "Prefer the lowest-risk release plan.")
    g.deliver(branch)
    g.append(branch, Node(DONE, result="hold the release"))

    print("original:", g.result(root))
    print("branch:", g.result(branch))

    # Appending after stale history is rejected instead of silently rewriting it.
    try:
        g.append(root, Node(CODE, code="print('late')", output="late"), after=first_answer)
    except StaleFrontier as error:
        print("stale edit rejected:", error)


if __name__ == "__main__":
    main()
