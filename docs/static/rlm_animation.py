"""rlmflow hero animation: a recursive run growing as a graph.

Run from the repo root.

Quick preview (480p, fast)::

    manim -pql docs/static/rlm_animation.py RecursiveFlowHero

High quality MP4 (1080p, the canonical render)::

    manim -qh docs/static/rlm_animation.py RecursiveFlowHero
    cp media/videos/rlm_animation/1080p60/RecursiveFlowHero.mp4 docs/static/rlm_animation.mp4

GIF for README / PyPI previews::

    ffmpeg -y -i docs/static/rlm_animation.mp4 \
      -vf "fps=20,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none" \
      docs/static/rlm_animation.gif
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ArcBetweenPoints,
    Circle,
    Create,
    DashedVMobject,
    FadeIn,
    FadeOut,
    Flash,
    Group,
    LaggedStart,
    Line,
    RegularPolygon,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    TransformFromCopy,
    VGroup,
)

BG = "#0B0F14"
WHITE_C = "#E6EDF3"
DIM = "#6E7681"

AGENT_C = "#58A6FF"
TURN_C = "#BC8CFF"
DONE_C = "#3FB950"
RETURN_C = "#58A6FF"
HOT = "#FFD60A"
LABEL_C = "#3FB950"

FS_HEADER = 24
FS_BODY = 14
FS_CAP = 12
FS_SMALL = 10
FS_TINY = 8

CODE_FONT = "Menlo"


def typed_node(kind, *, r=0.30):
    """One node: an agent's task, a model turn, a message, or the turn that submitted."""
    if kind == "agent":
        shape, fill = Circle(radius=r, color=AGENT_C, stroke_width=2), AGENT_C
    elif kind == "turn":
        shape = RegularPolygon(n=4, color=TURN_C, stroke_width=2).scale(r * 1.1)
        fill = TURN_C
    elif kind == "message":
        shape, fill = Circle(radius=r * 0.85, color=AGENT_C, stroke_width=2, stroke_opacity=0.9), BG
    elif kind == "done":
        shape, fill = RegularPolygon(n=6, color=DONE_C, stroke_width=2).scale(r * 1.05), DONE_C
    else:
        raise ValueError(kind)
    shape.fill_op = 1.0 if kind == "message" else 0.2
    shape.set_fill(fill, opacity=shape.fill_op)
    return VGroup(shape)


def chain(a, b, *, width=0.9):
    line = Line(a.get_bottom(), b.get_top(), color=DIM, stroke_width=width).set_opacity(0.7)
    line.stroke_op = 0.7
    return line.set_z_index(-1)


def spawn(a, b, *, width=0.9, opacity=0.55):
    line = Line(a.get_center(), b.get_center(), color=DIM, stroke_width=width).set_opacity(opacity)
    line.stroke_op = opacity
    return line.set_z_index(-2)


def ret(a, b, *, width=1.0, opacity=0.55, angle=-0.5):
    """A return edge: a dashed arc from a child's last node back to the waiting turn."""
    arc = ArcBetweenPoints(a.get_center(), b.get_center(), angle=angle)
    dashes = max(6, int(arc.get_arc_length() / 0.09)) if hasattr(arc, "get_arc_length") else 18
    line = DashedVMobject(arc, num_dashes=dashes).set_stroke(RETURN_C, width=width, opacity=opacity)
    for piece in line.get_family():
        piece.stroke_op = opacity
    return line.set_z_index(-2)


def restore(group):
    """Re-apply fills and stroke opacities after transforms that flatten them."""
    for mob in group.get_family():
        if hasattr(mob, "fill_op"):
            mob.set_fill(opacity=mob.fill_op)
            mob.set_stroke(opacity=1)
        elif hasattr(mob, "stroke_op"):
            mob.set_stroke(opacity=mob.stroke_op)


def agent_stack(label, kinds, *, x, top_y, r=0.10, dy=0.30, fs=FS_TINY, done_y=None):
    """One agent: its nodes top to bottom, joined in order, with its name above.

    Time runs downward, so a parent's last node (``done_y``) sits below its children's.
    """
    if label:
        label_mob = Text(label, font=CODE_FONT, font_size=fs, color=LABEL_C)
        label_mob.move_to(RIGHT * x + UP * (top_y + r + 0.14))
    else:
        label_mob = VGroup()
    nodes = [
        typed_node(kind, r=r).move_to(RIGHT * x + UP * (top_y - dy * i))
        for i, kind in enumerate(kinds)
    ]
    if done_y is not None:
        nodes[-1].move_to(RIGHT * x + UP * done_y)
    edges = [chain(nodes[i], nodes[i + 1], width=max(0.5, r * 6)) for i in range(len(nodes) - 1)]
    return nodes, edges, label_mob


def code_card(lines, *, width=5.6, height=2.0, font_size=12):
    box = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        color=DIM,
        stroke_width=1.0,
        fill_opacity=0.10,
    )
    header = Text("python", font=CODE_FONT, font_size=FS_SMALL, color=DIM)
    header.next_to(box.get_top(), DOWN, buff=0.16).align_to(box.get_left() + RIGHT * 0.22, LEFT)
    body = VGroup(*[Text(line, font=CODE_FONT, font_size=font_size, color=DIM) for line in lines])
    body.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    if body.width > width - 0.4:
        body.set_width(width - 0.4)
    body.next_to(header, DOWN, buff=0.20).align_to(box.get_left() + RIGHT * 0.22, LEFT)
    return VGroup(box, header, body)


def legend(*, font_size=8, r=0.065):
    items = VGroup()
    for kind, text in [
        ("agent", "agent"),
        ("turn", "model turn"),
        ("message", "message"),
        ("done", "submit"),
    ]:
        icon = typed_node(kind, r=r)
        items.add(
            VGroup(
                icon,
                Text(text, font=CODE_FONT, font_size=font_size, color=DIM).next_to(
                    icon, RIGHT, buff=0.08
                ),
            )
        )
    solid = Line(LEFT * 0.18, RIGHT * 0.18, color=DIM, stroke_width=1.2)
    items.add(
        VGroup(
            solid,
            Text("spawn", font=CODE_FONT, font_size=font_size, color=DIM).next_to(
                solid, RIGHT, buff=0.08
            ),
        )
    )
    dashed = DashedVMobject(Line(LEFT * 0.18, RIGHT * 0.18), num_dashes=4).set_stroke(
        RETURN_C, width=1.2
    )
    items.add(
        VGroup(
            dashed,
            Text("return", font=CODE_FONT, font_size=font_size, color=DIM).next_to(
                dashed, RIGHT, buff=0.08
            ),
        )
    )
    return items.arrange(RIGHT, buff=0.32)


def tree_graph():
    """root -> four agents -> two each, launched top-down, then results flowing back up."""
    root, root_e, root_l = agent_stack(
        "root", ["agent", "turn", "done"], x=0, top_y=2.35, r=0.13, dy=0.38, fs=FS_TINY, done_y=-1.9
    )
    stages = [VGroup(root_l, root[0]), VGroup(root_e[0], root[1])]
    mids, leaves = [], []
    for k, mx in enumerate((-4.5, -1.5, 1.5, 4.5)):
        name = f"worker-{k}"
        nodes, edges, label = agent_stack(
            name, ["agent", "turn", "done"], x=mx, top_y=1.05, r=0.11, dy=0.32, done_y=-1.4
        )
        mids.append((nodes, edges, label))
        for off in (-0.7, 0.7):
            lnodes, ledges, llabel = agent_stack(
                "",
                ["agent", "turn", "done"],
                x=mx + off,
                top_y=-0.2,
                r=0.085,
                dy=0.26,
                fs=6,
            )
            leaves.append((nodes, lnodes, ledges, llabel))

    def side(child, parent):
        return -0.5 if child.get_x() < parent.get_x() else 0.5

    stages.append(VGroup(*[VGroup(spawn(root[1], n[0]), lab, n[0]) for n, _e, lab in mids]))
    stages.append(VGroup(*[VGroup(e[0], n[1]) for n, e, _l in mids]))
    stages.append(
        VGroup(*[VGroup(spawn(p[1], n[0], width=0.7), lab, n[0]) for p, n, _e, lab in leaves])
    )
    stages.append(VGroup(*[VGroup(e[0], n[1]) for _p, n, e, _l in leaves]))
    stages.append(VGroup(*[VGroup(e[1], n[2]) for _p, n, e, _l in leaves]))
    stages.append(
        VGroup(*[ret(n[2], p[1], width=0.8, angle=side(n[2], p[1])) for p, n, _e, _l in leaves])
    )
    stages.append(VGroup(*[VGroup(e[1], n[2]) for n, e, _l in mids]))
    stages.append(
        VGroup(
            *[
                ret(n[2], root[1], width=0.8, opacity=0.45, angle=side(n[2], root[1]) * 0.6)
                for n, _e, _l in mids
            ]
        )
    )
    stages.append(VGroup(root_e[1], root[2]))
    return VGroup(*stages), stages


def step_parts():
    """The pieces of one small run, in the order the runtime adds them."""
    root, root_e, _ = agent_stack(
        "", ["agent", "turn", "done"], x=0, top_y=1.9, r=0.17, dy=0.55, done_y=-1.3
    )
    kids = [
        agent_stack("", ["agent", "turn", "done"], x=x, top_y=0.45, r=0.15, dy=0.46)
        for x in (-1.4, 1.4)
    ]
    return [
        (VGroup(root[0]), "root starts as one node: its task"),
        (VGroup(root_e[0], root[1]), "the model takes a turn: code, run in a REPL"),
        (
            VGroup(
                *[VGroup(spawn(root[1], n[0], width=1.1), n[0], e[0], n[1]) for n, e, _ in kids]
            ),
            "its code launches two agents; they run in parallel",
        ),
        (
            VGroup(
                *[
                    VGroup(e[1], n[2], ret(n[2], root[1], angle=-0.5 if n[2].get_x() < 0 else 0.5))
                    for n, e, _ in kids
                ]
            ),
            "each submits, and its result returns to the waiting turn",
        ),
        (VGroup(root_e[1], root[2]), "with both results in, the root's next turn submits"),
    ]


def fork_tree(x):
    """A root with two finished children, centered at x; returns (group, refs)."""
    root, root_e, root_l = agent_stack(
        "root", ["agent", "turn", "done"], x=x, top_y=1.6, r=0.18, dy=0.50, fs=FS_CAP, done_y=-1.3
    )
    kids = [
        agent_stack("", ["agent", "turn", "done"], x=x + dx, top_y=0.2, r=0.14, dy=0.42)
        for dx in (-0.8, 0.8)
    ]
    after = VGroup(root_e[0], root[1], root_e[1], root[2])
    for n, e, _ in kids:
        after.add(
            spawn(root[1], n[0]),
            *n,
            *e,
            ret(n[2], root[1], angle=-0.5 if n[2].get_x() < x else 0.5),
        )
    head = VGroup(root_l, root[0])
    return VGroup(head, after), {"agent": root[0], "head": head, "after": after}


def dense_tree():
    """A deep tree, launched top-down; then every parent submits below its children, bottom-up."""
    branching = [2, 2, 2, 2]
    style = [
        dict(top_y=2.45, r=0.13, dy=0.24, fs=7),
        dict(top_y=1.65, r=0.11, dy=0.22, fs=6),
        dict(top_y=0.90, r=0.095, dy=0.20, fs=5),
        dict(top_y=0.20, r=0.082, dy=0.19, fs=4),
        dict(top_y=-0.50, r=0.07, dy=0.18, fs=4),
    ]
    done_y = {3: -1.20, 2: -1.55, 1: -1.90, 0: -2.25}
    tree = []

    def add(label, depth, parent):
        idx = len(tree)
        tree.append({"label": label, "depth": depth, "parent": parent, "children": []})
        if parent is not None:
            tree[parent]["children"].append(idx)
        if depth < len(branching):
            for i in range(branching[depth]):
                add(f"{label}.{chr(ord('a') + i)}", depth + 1, idx)

    add("root", 0, None)
    cursor = [0]

    def place(idx):
        node = tree[idx]
        if not node["children"]:
            node["x"] = cursor[0] * 0.78
            cursor[0] += 1
        else:
            for c in node["children"]:
                place(c)
            node["x"] = sum(tree[c]["x"] for c in node["children"]) / len(node["children"])

    place(0)
    center = tree[0]["x"]
    launch = [[VGroup(), VGroup()] for _ in style]
    finish = {d: VGroup() for d in range(len(style))}
    for node in tree:
        d, st = node["depth"], style[node["depth"]]
        nodes, edges, label = agent_stack(
            node["label"] if d <= 1 else "",
            ["agent", "turn", "done"],
            x=node["x"] - center,
            top_y=st["top_y"],
            r=st["r"],
            dy=st["dy"],
            fs=st["fs"],
            done_y=done_y.get(d) if node["children"] else None,
        )
        node["nodes"] = nodes
        if node["parent"] is not None:
            launch[d][0].add(
                spawn(
                    tree[node["parent"]]["nodes"][1],
                    nodes[0],
                    width=max(0.3, 0.6 - 0.08 * d),
                    opacity=0.45 - 0.03 * d,
                )
            )
        launch[d][0].add(label, nodes[0])
        launch[d][1].add(edges[0], nodes[1])
        finish[d].add(edges[1], nodes[2])
    stages = [ring for level in launch for ring in level]
    stages += [finish[d] for d in sorted(finish, reverse=True)]
    graph = VGroup(*stages).move_to(DOWN * 0.25)
    return graph, stages


def center_head(text, *, buff=0.62):
    return Text(text, font=CODE_FONT, font_size=FS_HEADER, color=WHITE_C).to_edge(UP, buff=buff)


class RecursiveFlowHero(Scene):
    def construct(self):
        self.camera.background_color = BG
        brand = Text("rlmflow", font=CODE_FONT, font_size=26, color=WHITE_C).to_corner(
            UP + LEFT, buff=0.30
        )
        self.add(brand)

        def reveal(stage):
            return LaggedStart(
                *[FadeIn(item, shift=DOWN * 0.025) for item in stage], lag_ratio=0.08
            )

        # Phase 1: a recursive run grows as a graph, and results flow back up.
        head = center_head("Recursive agents as graphs")
        self.add(head)
        graph, stages = tree_graph()
        key = legend().to_edge(DOWN, buff=0.38)
        self.add(stages[0])
        for group, run_time in [
            (stages[1:2], 0.6),
            (stages[2:4], 1.3),
            (stages[4:6], 1.3),
            (stages[6:8], 1.3),
            (stages[8:], 1.6),
        ]:
            self.play(LaggedStart(*[reveal(s) for s in group], lag_ratio=0.45), run_time=run_time)
        self.play(FadeIn(key, shift=UP * 0.04), run_time=0.5)
        self.wait(1.0)

        # Phase 2: the runtime adds nodes as the run goes; step through it.
        title = center_head("Step through the live run")
        self.play(
            FadeOut(VGroup(graph, key), shift=DOWN * 0.05),
            ReplacementTransform(head, title),
            run_time=0.7,
        )
        head = title
        parts = step_parts()
        timeline = Line(
            LEFT * 4.8 + DOWN * 3.28, RIGHT * 4.8 + DOWN * 3.28, color=DIM, stroke_width=1.2
        )
        dots = VGroup(
            *[
                Circle(radius=0.07, color=DIM, fill_opacity=0.35, stroke_width=1.0).move_to(
                    LEFT * 4.8 + RIGHT * (9.6 * i / (len(parts) - 1)) + DOWN * 3.28
                )
                for i in range(len(parts))
            ]
        )
        self.play(Create(timeline), FadeIn(dots), run_time=0.4)
        shown, caption, marker = VGroup(), None, None
        for i, (piece, text) in enumerate(parts):
            new_caption = Text(text, font=CODE_FONT, font_size=FS_BODY, color=WHITE_C).move_to(
                DOWN * 2.62
            )
            new_marker = Circle(
                radius=0.11, color=HOT, fill_opacity=0.65, stroke_width=1.5
            ).move_to(dots[i])
            fades = [FadeOut(caption), FadeOut(marker)] if caption is not None else []
            self.play(
                *fades,
                reveal(piece),
                FadeIn(new_caption, shift=UP * 0.04),
                FadeIn(new_marker),
                run_time=0.9,
            )
            self.wait(0.8)
            shown.add(piece)
            caption, marker = new_caption, new_marker
        self.play(
            FadeOut(VGroup(timeline, dots, shown, caption, marker), shift=DOWN * 0.05),
            run_time=0.55,
        )

        # Phase 3: fork the run at a node, send the copy a message, and run it.
        code = code_card(
            [
                "branch = g.fork(root)     # cut root after its task",
                'branch.send(root, "try it another way")',
                "flow.run(branch)",
            ]
        ).to_edge(LEFT, buff=0.40)
        lines = code[2]

        def light(idx):
            return [lines[i].animate.set_color(HOT if i == idx else DIM) for i in range(len(lines))]

        orig, _ = fork_tree(0)
        orig.move_to(UP * 0.2)
        title = center_head("Every run is a graph you can edit")
        self.play(ReplacementTransform(head, title), run_time=0.55)
        head = title
        self.play(FadeIn(orig[0]), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(m, shift=DOWN * 0.02) for m in orig[1]], lag_ratio=0.05),
            run_time=1.2,
        )
        self.wait(0.4)

        title = center_head("Fork the run")
        branch, refs = fork_tree(0)
        branch.move_to(RIGHT * 5.0 + UP * 0.2)
        badge = VGroup(
            RoundedRectangle(
                corner_radius=0.10,
                width=1.45,
                height=0.42,
                color=HOT,
                stroke_width=1.3,
                fill_opacity=0.08,
            ),
            Text("branch", font=CODE_FONT, font_size=FS_CAP, color=HOT),
        ).next_to(refs["head"], UP, buff=0.18)
        self.play(
            ReplacementTransform(head, title),
            FadeIn(code, shift=UP * 0.06),
            *light(0),
            run_time=0.6,
        )
        head = title
        self.play(
            orig.animate.move_to(RIGHT * 1.7 + UP * 0.2),
            TransformFromCopy(orig, branch),
            run_time=1.1,
        )
        restore(orig)
        restore(branch)
        self.play(FadeIn(badge, shift=UP * 0.05), run_time=0.4)

        title = center_head("Fork at a node, cut what came after")
        ring = Circle(radius=0.30, color=HOT, stroke_width=2.4).move_to(refs["agent"])
        self.play(ReplacementTransform(head, title), Create(ring), run_time=0.6)
        head = title
        self.play(FadeOut(refs["after"], shift=DOWN * 0.2), run_time=0.6)
        self.wait(0.3)

        title = center_head("Send it a message")
        agent = refs["agent"]
        msg = typed_node("message", r=0.18).move_to(agent.get_center() + DOWN * 0.50)
        self.play(ReplacementTransform(head, title), *light(1), FadeOut(ring), run_time=0.55)
        head = title
        self.play(FadeIn(VGroup(chain(agent, msg), msg), shift=DOWN * 0.05), run_time=0.5)
        self.play(Flash(msg, color=AGENT_C, flash_radius=0.32), run_time=0.38)

        title = center_head("Run the copy: a different path")
        bx = agent.get_x()
        turn = typed_node("turn", r=0.18).move_to(msg.get_center() + DOWN * 0.50)
        kids = [
            agent_stack(
                "", ["agent", "turn", "done"], x=bx + dx, top_y=turn.get_y() - 0.9, r=0.11, dy=0.36
            )
            for dx in (-1.05, -0.35, 0.35, 1.05)
        ]
        done = typed_node("done", r=0.18).move_to(RIGHT * bx + UP * (kids[0][0][2].get_y() - 0.6))
        self.play(ReplacementTransform(head, title), *light(2), run_time=0.55)
        head = title
        self.play(FadeIn(VGroup(chain(msg, turn), turn), shift=DOWN * 0.04), run_time=0.45)
        self.play(
            LaggedStart(
                *[FadeIn(VGroup(spawn(turn, n[0]), n[0], e[0], n[1])) for n, e, _ in kids],
                lag_ratio=0.3,
            ),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                *[
                    FadeIn(VGroup(e[1], n[2], ret(n[2], turn, angle=-0.5 if dx < 0 else 0.5)))
                    for (n, e, _), dx in zip(kids, (-1, -1, 1, 1))
                ],
                lag_ratio=0.3,
            ),
            run_time=0.8,
        )
        self.play(FadeIn(VGroup(chain(turn, done), done), shift=DOWN * 0.04), run_time=0.45)
        self.play(Flash(done, color=DONE_C, flash_radius=0.30), run_time=0.38)
        self.wait(1.0)

        # Phase 4: the same pieces at any depth.
        finale = center_head("Nested agent trees, any depth", buff=0.72)
        dense, dense_stages = dense_tree()
        fork_mobs = Group(*[m for m in self.mobjects if m is not brand and m is not head])
        self.play(
            FadeOut(head), FadeOut(fork_mobs), FadeIn(finale, shift=DOWN * 0.05), run_time=0.75
        )
        for stage in dense_stages:
            self.play(reveal(stage), run_time=0.18)
        self.play(Flash(dense, color=DONE_C, flash_radius=0.35), run_time=0.28)
        self.wait(1.2)
