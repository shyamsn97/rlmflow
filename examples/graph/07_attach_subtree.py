"""Attach agents prepared in another graph as children of an agent in this one."""

from __future__ import annotations

from rlmflow import AgentStart, DoneOutput, ExecOutput, Graph, ops, reads


def main() -> None:
    g = Graph()
    root = ops.spawn(g, "prepare a release decision")

    # A controller can prepare (or load) agents in a graph of their own, merge
    # that graph into this one, and give them a launcher. attach() writes one launch action whose
    # receipts resolve to the attached agents, so replay never relaunches them.
    side = Graph()
    for task, verdict in (
        ("audit the release candidate", {"safe": False, "reason": "canary failed"}),
        ("audit the rollback path", {"safe": True}),
    ):
        audit = ops.spawn(side, task)
        ops.append(side, audit, DoneOutput(result=verdict))
    g.merge(side)
    audits = [agent for agent in g.nodes(AgentStart) if agent is not root]
    ops.attach(g, root, *audits, names=["candidate", "rollback"])

    results = [reads.result(g, audit) for audit in audits]
    ops.append(g, root, ExecOutput(content=str(results)))
    ops.append(g, root, DoneOutput(result="release blocked"))

    for agent in reads.tree(g, root):
        print(f"{reads.path(g, agent)}: {reads.result(g, agent)!r}")


if __name__ == "__main__":
    main()
