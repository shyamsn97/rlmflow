from __future__ import annotations

import tomllib
from pathlib import Path

from rlmflow.graph.persistence import VERSION as PERSISTENCE_VERSION

ROOT = Path(__file__).parents[1]


def test_release_version_and_persistence_docs_are_consistent():
    package_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    upgrading = (ROOT / "UPGRADING.md").read_text(encoding="utf-8")
    internals = (ROOT / "docs" / "internals.md").read_text(encoding="utf-8")

    assert package_version == "0.6.0"
    assert f"## [{package_version}]" in changelog
    assert f"[{package_version}]:" in changelog
    assert f"# Upgrading to {package_version.rsplit('.', 1)[0]}" in upgrading
    assert PERSISTENCE_VERSION == 5
    assert "strict v5 flat JSON shape" in internals
