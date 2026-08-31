from pathlib import Path

from benchmarks.eval.runners.official_rlm import (
    OfficialRLMRunner,
    _official_script,
    _render_context,
)
from benchmarks.eval.runners.rlmflow import (
    RLMFlowLocalRunner,
    _example_inputs,
    _example_tools,
)
from benchmarks.eval.runners.shared import fact_lookup_data, materialize_fixtures
from benchmarks.eval.types import Example


def test_official_rlm_does_not_inline_fact_lookup_data_into_context():
    example = Example(
        id="parallel",
        prompt="Compute the result.",
        metadata={
            "tool": "fact_lookup",
            "facts": {"A": "10", "B": "20"},
        },
    )

    context = _render_context(example)

    assert context == {}

    script = _official_script(
        context_file="/tmp/context.json",
        task_file="/tmp/task.txt",
        model="model",
        log_dir=Path("/tmp/logs"),
        max_iters=20,
        max_depth=2,
        max_budget=100_000,
        facts={"a": "10", "b": "20"},
    )
    assert "def fact_lookup(entity):" in script
    assert "custom_tools=custom_tools" in script
    assert "FIXED FACT LOOKUP" not in script
    compile(script, "<official-rlm-script>", "exec")
    assert _example_tools(example)[0]("A") == "10"
    assert fact_lookup_data(example) == {"a": "10", "b": "20"}


def test_official_rlm_preserves_the_same_input_mapping_as_rlmflow():
    example = Example(id="context", prompt="Answer.", context="Evidence")

    assert _render_context(example) == {"context": "Evidence"}
    assert _render_context(example) == _example_inputs(example)


def test_both_runners_receive_identical_filesystem_fixtures(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "facts.txt").write_text("evidence")
    example = Example(
        id="files",
        prompt="Read the fixture.",
        metadata={"fixture_paths": [str(source)]},
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    materialize_fixtures(example, left)
    materialize_fixtures(example, right)

    assert (left / "source" / "facts.txt").read_text() == "evidence"
    assert (right / "source" / "facts.txt").read_text() == "evidence"


def test_comparison_runners_share_iteration_depth_and_budget_defaults():
    official = OfficialRLMRunner()
    rlmflow = RLMFlowLocalRunner()

    assert official.max_iters == rlmflow.max_iters == 20
    assert official.max_depth == rlmflow.max_depth == 2
    assert official.max_budget == rlmflow.max_budget == 100_000
    assert rlmflow.child_max_iters == rlmflow.max_iters
    assert rlmflow.use_llm_query
    assert not rlmflow.use_agent_tree
