"""Attach agents prepared in another graph as children of an agent in this one."""

from __future__ import annotations

from rlmflow import AGENT_START, CODE, DONE, Graph, Node


def main() -> None:
    g = Graph()
    root = g.spawn("prepare a release decision", name="root")

    # A controller can prepare (or load) agents in a graph of their own, merge
    # that graph into this one, and give them a launcher. attach() tells the
    # launcher with a message; the attached agents report to it from then on.
    side = Graph()
    for name, task, verdict in (
        ("candidate", "audit the release candidate", {"safe": False, "reason": "canary failed"}),
        ("rollback", "audit the rollback path", {"safe": True}),
    ):
        audit = side.spawn(task, name=name)
        side.append(audit, Node(DONE, result=verdict))
    g.merge(side)
    audits = [agent for agent in g.nodes(AGENT_START) if agent is not root]
    g.attach(root, *audits)
    g.deliver(root)

    results = [g.result(audit) for audit in audits]
    g.append(root, Node(CODE, code="print(results)", output=str(results)))
    g.append(root, Node(DONE, result="release blocked"))

    for agent in g.tree(root):
        print(f"{g.path(agent)}: {g.result(agent)!r}")


if __name__ == "__main__":
    main()
