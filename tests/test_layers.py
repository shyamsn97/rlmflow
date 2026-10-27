"""Imports inside ``rlmflow`` only point down the layer stack."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "rlmflow"

LAYERS = {
    "rlmflow.utils": 0,
    "rlmflow.schema": 0,
    "rlmflow.graph": 1,
    "rlmflow.llm": 2,
    "rlmflow.harnesses.repl.runtime": 2,
    "rlmflow.integrations": 2,
    "rlmflow.engine": 3,
    "rlmflow.agent_tools": 3,
    "rlmflow.harnesses": 4,
    "rlmflow.flow": 5,
    "rlmflow.cli": 6,
    "rlmflow.consumers": 6,
    "rlmflow.view": 6,
    "rlmflow.rao": 6,
    "rlmflow.__main__": 6,
    "rlmflow": 6,
}

# Modules that sit lower than their package. Layer-0 modules import nothing
# from rlmflow.
OVERRIDES: dict[str, int] = {}


def module_name(path: Path) -> str:
    parts = path.relative_to(ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def layer(module: str) -> int:
    if module in OVERRIDES:
        return OVERRIDES[module]
    while module not in LAYERS:
        module = module.rpartition(".")[0]
    return LAYERS[module]


def is_type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def runtime_imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and is_type_checking(node):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found.append((node.lineno, f"{node.module}.{alias.name}"))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return [(line, name) for line, name in found if name.split(".")[0] == "rlmflow"]


def target(name: str) -> str:
    """The deepest real module ``name`` refers to (``from pkg import mod`` or a name)."""
    while name != "rlmflow":
        path = ROOT.parent / name.replace(".", "/")
        if path.with_suffix(".py").exists() or (path / "__init__.py").exists():
            return name
        name = name.rpartition(".")[0]
    return name


def test_imports_only_point_down() -> None:
    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        source = module_name(path)
        for line, name in runtime_imports(ast.parse(path.read_text(), str(path))):
            imported = target(name)
            if layer(imported) > layer(source):
                violations.append(
                    f"{path.relative_to(ROOT.parent)}:{line} {source} (layer {layer(source)}) "
                    f"imports {imported} (layer {layer(imported)})"
                )
    assert not violations, "\n".join(violations)


def test_leaves_import_nothing_from_rlmflow() -> None:
    leaves = {**LAYERS, **OVERRIDES}
    for module in (name for name, level in leaves.items() if level == 0):
        path = ROOT.parent / f"{module.replace('.', '/')}.py"
        imports = runtime_imports(ast.parse(path.read_text(), str(path)))
        assert not imports, f"{module} must stay importable on its own: {imports}"


#: What the REPL worker may import: it runs inside Docker and Modal images.
WORKER_IMPORTS = ("rlmflow.harnesses.repl.runtime.repl", "rlmflow.schema", "rlmflow.utils")


def test_the_repl_worker_imports_only_leaf_modules():
    path = ROOT / "harnesses" / "repl" / "runtime" / "repl_server.py"
    found = [
        f"{path.relative_to(ROOT.parent)}:{line} imports {name}"
        for line, name in runtime_imports(ast.parse(path.read_text(encoding="utf-8")))
        if not name.startswith(WORKER_IMPORTS)
    ]
    assert not found, "\n".join(found)


def test_the_worker_copy_of_the_agent_reference_key_matches_the_graph():
    from rlmflow.graph.nodes import AGENT_REF_KEY
    from rlmflow.harnesses.repl.runtime import repl

    assert repl.AGENT_REF_KEY == AGENT_REF_KEY
