from __future__ import annotations

import ast
import asyncio
import json
import re
import sys
from types import SimpleNamespace

import pytest
from helpers import StubLLM

from benchmarks.eval import DATASETS
from benchmarks.eval.delegation.align import ChildAlignment, score_alignment
from benchmarks.eval.delegation.annotations import ANNOTATIONS, annotation_for
from benchmarks.eval.delegation.conditions import (
    CapabilityOnlyStartStep,
    DelegationCondition,
    apply_condition,
    system_prompt_for,
)
from benchmarks.eval.delegation.faults import FAULT_KINDS, fault_spec
from benchmarks.eval.delegation.manifest import FAULT_SEEDS, PROBLEM_IDS, SLOTS
from benchmarks.eval.delegation.metrics import delegation_metrics
from benchmarks.eval.delegation.report import delegation_report
from benchmarks.eval.runners.rlmflow import (
    _example_inputs,
    _example_tools,
    _prediction_answer,
)
from benchmarks.eval.tasks._delegation_utils import exact_or_alias
from benchmarks.eval.tasks.arc_agi import ArcAgiTaskGraphDataset
from benchmarks.eval.tasks.dabstep import DABstepTaskGraphDataset
from benchmarks.eval.tasks.delegation_codeqa import (
    FROZEN_INDEX as CODEQA_INDEX,
)
from benchmarks.eval.tasks.delegation_codeqa import DelegationCodeQADataset
from benchmarks.eval.tasks.delegation_iteration import (
    REGRESSION_TASKS,
    SELECTED_TASKS,
    TEN_TASKS,
    DelegationIterationDataset,
    DelegationIterationTenDataset,
    DelegationRegressionDataset,
)
from benchmarks.eval.tasks.delegation_sudoku import SOLUTION as SUDOKU_SOLUTION
from benchmarks.eval.tasks.delegation_sudoku import DelegationSudokuDataset
from benchmarks.eval.tasks.entailmentbank import FROZEN_ID as ENTAILMENT_ID
from benchmarks.eval.tasks.entailmentbank import EntailmentBankTaskGraphDataset
from benchmarks.eval.tasks.grsqa import GRSQATaskGraphDataset
from benchmarks.eval.tasks.musique import FROZEN_ID as MUSIQUE_ID
from benchmarks.eval.tasks.musique import MuSiQueTaskGraphDataset
from benchmarks.eval.tasks.natural_plan import FROZEN_IDS as NATURAL_PLAN_IDS
from benchmarks.eval.tasks.natural_plan import NaturalPlanTaskGraphDataset
from benchmarks.eval.tasks.parallelqa import FROZEN_IDS, ParallelQATaskGraphDataset
from benchmarks.eval.tasks.planbench import PlanBenchTaskGraphDataset
from benchmarks.eval.tasks.twowiki import FROZEN_IDS as TWOWIKI_IDS
from benchmarks.eval.tasks.twowiki import TwoWikiTaskGraphDataset
from benchmarks.eval.types import Example, Prediction, Row, Score
from rlmflow import (
    AgentStart,
    DoneOutput,
    ExecAction,
    ExecOutput,
    Flow,
    LLMOutput,
    PlanQuery,
    start,
)
from rlmflow.graph.nodes import PLANNING_ACTION


def test_delegation_manifest_has_exactly_twenty_ordered_problems():
    problems = [problem for slot in SLOTS for problem in slot.problems]

    assert problems == list(range(1, 21))
    assert sum(slot.count for slot in SLOTS) == 20
    assert sorted(PROBLEM_IDS) == problems
    assert all(len(slot.revision) == 40 and slot.license for slot in SLOTS)
    assert DATASETS.expand(["delegation-suite"]) == [slot.dataset for slot in SLOTS]


@pytest.mark.parametrize(
    ("answer", "expected", "correct"),
    [
        ("5931.8", "5931.80", True),
        ("1e3", "1000.0", True),
        ("5930.8", "5931.80", False),
        ("NaN", "NaN", True),
        ("Puerto Rico Trench", "Puerto Rico Trench", True),
        # An answer key of "9810.79" claims two decimals; float round-tripping past
        # that precision is not a wrong answer. These exact strings were graded 0.
        ("9810.79000000000126", "9810.79", True),
        ("8408.629999999999475", "8408.63", True),
        ("5931.80000000000080", "5931.80", True),
        # Rounding stays at the key's precision and never coarser than whole numbers.
        ("9810.794", "9810.79", True),
        ("9810.80", "9810.79", False),
        ("772.77", "9810.79", False),
        ("3.4", "3", False),
        ("3.0000000000001", "3", True),
    ],
)
def test_exact_or_alias_grades_numbers_at_the_answer_keys_precision(answer, expected, correct):
    assert exact_or_alias(answer, expected) is correct


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("plain", "plain"),
        ({"answer": 1}, '{"answer": 1}'),
        ([1, "é"], '[1, "é"]'),
        (None, "null"),
    ],
)
def test_rlmflow_runner_serializes_typed_results_at_text_boundary(result, expected):
    assert _prediction_answer(result) == expected


def test_capability_only_keeps_agent_api_but_removes_policy_sections():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        system_prompt=system_prompt_for(DelegationCondition.CAPABILITY_ONLY),
    )
    apply_condition(flow, DelegationCondition.CAPABILITY_ONLY)
    root = flow.start("solve", inputs={"task": "one coherent problem"}, max_depth=2)
    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert flow.get_step_fn(root) is CapabilityOnlyStartStep
    assert "launch_subagent" in prompt
    assert "Delegate substantial independent workstreams" not in prompt
    assert "## Examples" not in prompt


def test_default_prompt_is_depth_aware_and_bounded():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("solve", inputs={"task": "{}"}, max_depth=2)
    child = start("scope", inputs={"task": "{}"}, depth=1, max_depth=2)
    leaf = start("scope", inputs={"task": "{}"}, depth=2, max_depth=2)

    root_prompt = flow.build_messages(root.frontier)[0]["content"]
    child_prompt = flow.build_messages(child.frontier)[0]["content"]
    leaf_prompt = flow.build_messages(leaf.frontier)[0]["content"]

    local_heading = "**Local work — probe, print the candidate, then submit it.**"
    delegation_heading = "**Delegation — a source too large to read here.**"

    # The source is official-rlm's base prompt and orchestrator addendum, adapted
    # for rlmflow's APIs plus the explicit print-then-finish contract.
    assert len(root_prompt) <= 9_500
    # A spawn-capable child carries the delegation trigger list but not the example.
    assert len(child_prompt) <= 9_000
    assert len(leaf_prompt) <= 6_500
    assert local_heading in root_prompt
    assert delegation_heading in root_prompt
    assert local_heading in child_prompt
    assert delegation_heading not in child_prompt
    assert "launch_subagent" in child_prompt
    assert "launch_subagent" not in leaf_prompt
    assert "output_schema=" not in root_prompt
    assert "AgentHandle" in root_prompt
    assert 'inputs={"path": INPUTS["transactions_path"]}' in root_prompt

    # Local leads: print-then-submit is what every row depends on, while delegation
    # fired on 3 of 34 rows, all on the one task that scored 0.000 nine times.
    assert root_prompt.index(local_heading) < root_prompt.index(delegation_heading)

    compact = " ".join(root_prompt.lower().split())
    for phrase in ("must delegate", "must launch", "must wait", "must retrieve"):
        assert phrase not in compact

    repl_blocks = re.findall(r"```repl\s*\n(.*?)\n```", root_prompt, re.DOTALL)
    # Six: both examples demonstrate separate work / inspect / finish turns. Cutting
    # the local example to two blocks — dropping the probe and its assert — was part
    # of a rewrite that lost 0.240 paired, so the three-step shape is load-bearing.
    assert len(repl_blocks) == 6
    for code in repl_blocks:
        compile(
            ast.parse(code),
            "<prompt-example>",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )


def test_default_prompt_shows_schema_only_when_explicit():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    plain = start("solve", max_depth=1)
    structured = start(
        "solve",
        max_depth=1,
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    plain_prompt = flow.build_messages(plain.frontier)[0]["content"]
    structured_prompt = flow.build_messages(structured.frontier)[0]["content"]

    assert "This run requires structured output" not in plain_prompt
    assert "This run requires structured output" in structured_prompt
    assert '"answer"' in structured_prompt


def test_default_prompt_uses_minimal_opening_action():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = flow.start("solve", inputs={"task": "data"}, max_depth=2)
    child = start("scope", inputs={"task": "data"}, depth=1, max_depth=2)

    root_action = asyncio.run(flow.step(root)).created
    child_action = asyncio.run(flow.step(child)).created
    assert root_action.content == PLANNING_ACTION
    assert child_action.content == PLANNING_ACTION


def test_prompt_only_benchmarks_do_not_receive_fabricated_inputs():
    example = Example(id="prompt-only", prompt="Solve the complete task.")
    inputs = _example_inputs(example)
    root = start(example.prompt, inputs=inputs, max_depth=2)
    flow = Flow(StubLLM(lambda _messages: "unused"))

    assert inputs == {}
    assert isinstance(asyncio.run(flow.step(root)).created, PlanQuery)


def test_parallelqa_fact_lookup_is_documented_in_the_rlmflow_prompt():
    example = Example(
        id="lookup",
        prompt="look up Alpha",
        metadata={"tool": "fact_lookup", "facts": {"Alpha": 42}},
    )
    tools = _example_tools(example)
    flow = Flow(StubLLM(lambda _messages: "unused"), tools=tools)
    root = flow.start(example.prompt, max_depth=1)

    prompt = flow.build_messages(root.frontier)[0]["content"]

    assert "`fact_lookup(" in prompt
    assert "Required task-data source" in prompt
    assert "preserve the returned numeric precision" in prompt
    assert tools[0]("Alpha") == "42"


def test_delegation_metrics_capture_launch_batches_failures_and_usage():
    root = start("solve", max_depth=2)
    action = root.append(LLMOutput(content="launch")).append(ExecAction(code="launch"))
    child_a = action.append(AgentStart(content="branch a", config=root.config.child("a")))
    child_b = action.append(AgentStart(content="branch b", config=root.config.child("b")))
    child_a.append(DoneOutput(result="A"))
    child_b.append(DoneOutput(result="[child failed: boom]"))
    action.mark_agent_retrieved(child_a.id)
    action.append(ExecOutput(content="children done")).append(DoneOutput(result="root"))

    metrics = delegation_metrics(root)

    assert metrics["children"] == 2
    assert metrics["max_launch_batch"] == 2
    assert metrics["concurrent_launch_batches"] == 1
    assert metrics["failed_children"] == 1
    assert metrics["retrieved_child_results"] == 1
    assert metrics["unretrieved_child_results"] == 1
    assert metrics["echoed_child_results"] == 0
    assert "consumed_child_results" not in metrics
    assert not metrics["root_finished_with_unfinished_children"]
    assert len(metrics["child_runs"]) == 2
    assert metrics["delegated_routing_decisions"] == 1
    assert metrics["local_routing_decisions"] == 2
    assert metrics["routing_decisions"][0]["outcome"] == "delegated"
    assert metrics["routing_decisions"][0]["retrieved_child_results"] == 1
    assert all(len(run["goal_hash"]) == 64 for run in metrics["child_runs"])
    assert all("result_echoed" in run for run in metrics["child_runs"])
    assert [run["result_retrieved"] for run in metrics["child_runs"]] == [True, False]


def test_private_annotations_are_complete_acyclic_and_never_model_inputs():
    assert sorted(ANNOTATIONS) == list(range(1, 21))
    assert annotation_for(18).preferred_local
    assert all(
        obligation.description
        for annotation in ANNOTATIONS.values()
        for obligation in annotation.obligations
    )


def test_alignment_rewards_successful_local_restraint_without_child_count():
    root = start("solve locally", max_depth=2)
    root.append(DoneOutput(result="solved"))

    score = score_alignment(root, annotation_for(18), [], outcome=1.0)

    assert score["trajectory"] == 1.0
    assert score["outcome_gated_trajectory"] == 1.0


def test_alignment_uses_external_labels_and_outcome_gate():
    root = start("solve", max_depth=2)
    action = root.append(ExecAction(code="launch"))
    child = action.append(AgentStart(content="find facts", config=root.config.child("facts")))
    child.append(DoneOutput(result="facts"))
    labels = [
        ChildAlignment(
            agent_id=child.id,
            obligations=("o1",),
            result_used=True,
        )
    ]

    score = score_alignment(root, annotation_for(1), labels, outcome=0.0)

    assert score["coverage"] == 0.5
    assert score["outcome_gated_trajectory"] == 0.0


def test_fault_seeds_map_once_to_each_recovery_fault():
    specs = [fault_spec(1, seed) for seed in FAULT_SEEDS]

    assert [spec.kind for spec in specs] == list(FAULT_KINDS)
    assert specs == [fault_spec(1, seed) for seed in FAULT_SEEDS]


def test_report_preserves_per_problem_repetitions():
    prediction = Prediction(
        answer="ok",
        metrics={
            "delegation": {
                "condition": "current_policy",
                "children": 2,
                "model_calls": 4,
                "retrieved_child_results": 2,
                "unretrieved_child_results": 0,
            }
        },
    )
    rows = [
        Row(
            run_id=f"run-{index}",
            dataset="delegation_parallelqa",
            example_id="delegation_parallelqa_11",
            runner="rlmflow",
            model="model",
            seed=index,
            prediction=prediction,
            score=Score(value=1.0, correct=True),
        )
        for index in range(2)
    ]

    report = delegation_report(rows)

    assert report["rows"] == 2
    assert report["problems"][0]["attempts"] == 2
    assert report["problems"][0]["delegation_rate"] == 1.0
    assert report["problems"][0]["mean_retrieved_child_results"] == 2.0
    assert report["problems"][0]["mean_unretrieved_child_results"] == 0.0
    assert report["conditions"][0]["macro_pass_at_1"] == 1.0


def test_parallelqa_freezes_four_native_rows_and_scores_exactly(tmp_path):
    dataset = ParallelQATaskGraphDataset(data_dir=str(tmp_path))
    dataset._rows = [
        {"id": source_id, "question": f"question {source_id}", "answer": source_id, "branch": 2}
        for source_id in FROZEN_IDS
    ]

    examples = dataset.examples(split="test", limit=None, seed=0)

    assert [example.metadata["source_id"] for example in examples] == list(FROZEN_IDS)
    assert dataset.score(examples[0], Prediction(answer="11")).correct
    assert not dataset.score(examples[0], Prediction(answer="12")).correct


def test_iteration_suite_selects_five_stable_roles(monkeypatch, tmp_path):
    dataset = DelegationIterationDataset(data_dir=str(tmp_path))
    parallelqa, planbench, _sudoku, arc = dataset._datasets
    parallelqa._rows = [
        {"id": source_id, "question": f"question {source_id}", "answer": source_id, "branch": 2}
        for source_id in FROZEN_IDS
    ]
    monkeypatch.setattr(planbench, "_download", lambda name: f"({name})")
    monkeypatch.setattr(
        arc,
        "_load",
        lambda _source_id: {
            "train": [],
            "test": [{"input": [[0]], "output": [[0]]}],
        },
    )

    examples = dataset.examples(split="test", limit=None, seed=0)

    assert [example.id for example in examples] == [task[0] for task in SELECTED_TASKS]
    assert [example.metadata["iteration_role"] for example in examples] == [
        task[1] for task in SELECTED_TASKS
    ]
    assert len(dataset.examples(split="test", limit=2, seed=0)) == 2


def test_ten_task_iteration_suite_is_stable(tmp_path):
    dataset = DelegationIterationTenDataset(data_dir=str(tmp_path))

    assert dataset.selected_tasks == TEN_TASKS
    assert len(TEN_TASKS) == 10
    assert len({example_id for example_id, _role in TEN_TASKS}) == 10


def test_regression_suite_targets_five_observed_failures(tmp_path):
    dataset = DelegationRegressionDataset(data_dir=str(tmp_path))

    assert dataset.selected_tasks == REGRESSION_TASKS
    assert len(REGRESSION_TASKS) == 5
    assert len({example_id for example_id, _role in REGRESSION_TASKS}) == 5


def test_musique_scores_answer_and_support_jointly():
    dataset = MuSiQueTaskGraphDataset()
    decomposition = [
        {"paragraph_support_idx": index, "question": f"hop {index}", "answer": str(index)}
        for index in range(4)
    ]
    paragraphs = [
        {"idx": index, "title": f"P{index}", "paragraph_text": f"Evidence {index}"}
        for index in range(20)
    ]
    dataset._rows = [
        {
            "id": "positive",
            "question": "What follows?",
            "question_decomposition": decomposition,
            "paragraphs": paragraphs,
            "answer": "result",
            "answer_aliases": ["the result"],
            "answerable": True,
        },
        {
            "id": "negative",
            "question": "What follows?",
            "question_decomposition": decomposition,
            "paragraphs": paragraphs,
            "answer": "",
            "answer_aliases": [],
            "answerable": False,
        },
    ]
    example = dataset.examples(split="test", limit=None, seed=0)[0]
    answer = json.dumps({"answer": "result", "supporting_paragraphs": ["0", "1", "2", "3"]})

    score = dataset.score(example, Prediction(answer=answer))

    assert score.correct
    assert score.value == 1.0


def test_musique_loader_ignores_duplicate_rows_for_one_pair_side(
    tmp_path,
    monkeypatch,
):
    rows = [
        {"id": MUSIQUE_ID, "answerable": True, "question": "q"},
        {"id": MUSIQUE_ID, "answerable": True, "question": "q"},
        {"id": MUSIQUE_ID, "answerable": False, "question": "q"},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: iter(rows)),
    )

    loaded = MuSiQueTaskGraphDataset(data_dir=str(tmp_path))._load()

    assert [row["answerable"] for row in loaded] == [True, False]


def test_twowiki_scores_answer_support_and_evidence():
    dataset = TwoWikiTaskGraphDataset()
    base = {
        "question": "Compare the two entities.",
        "answer": "yes",
        "context": [["A", ["A fact."]], ["B", ["B fact."]]],
        "supporting_facts": [["A", 0], ["B", 0]],
        "evidences": [["A", "same", "B"]],
    }
    dataset._rows = [
        {**base, "id": TWOWIKI_IDS["inference"], "type": "inference"},
        {
            **base,
            "id": TWOWIKI_IDS["bridge_comparison"],
            "type": "bridge_comparison",
        },
    ]
    example = dataset.examples(split="test", limit=None, seed=0)[0]
    answer = json.dumps(
        {
            "answer": "yes",
            "supporting_facts": ["A:0", "B:0"],
            "evidence": ["A|same|B"],
        }
    )

    assert dataset.score(example, Prediction(answer=answer)).correct


def test_planbench_validates_the_frozen_eleven_action_plan(monkeypatch):
    problem = """
    (define (problem x) (:domain logistics-strips)
    (:init (TRUCK t0) (TRUCK t1) (AIRPLANE a0)
      (AIRPORT l0-0) (AIRPORT l1-0)
      (in-city l0-2 c0) (in-city l0-1 c0) (in-city l0-0 c0)
      (in-city l1-0 c1) (in-city l1-2 c1)
      (at t0 l0-2) (at t1 l1-0) (at p0 l0-1) (at a0 l1-0))
    (:goal (and (at p0 l1-2))))"""
    dataset = PlanBenchTaskGraphDataset()
    monkeypatch.setattr(
        dataset,
        "_download",
        lambda filename: "(domain)" if filename == "generated_domain.pddl" else problem,
    )
    example = dataset.examples(split="test", limit=None, seed=0)[0]
    plan = """
    (drive-truck t0 l0-2 l0-1 c0)
    (load-truck p0 t0 l0-1)
    (drive-truck t0 l0-1 l0-0 c0)
    (unload-truck p0 t0 l0-0)
    (fly-airplane a0 l1-0 l0-0)
    (load-airplane p0 a0 l0-0)
    (fly-airplane a0 l0-0 l1-0)
    (unload-airplane p0 a0 l1-0)
    (load-truck p0 t1 l1-0)
    (drive-truck t1 l1-0 l1-2 c1)
    (unload-truck p0 t1 l1-2)
    """

    score = dataset.score(example, Prediction(answer=plan))

    assert score.correct
    assert score.details["optimality_gap"] == 0


def test_arc_requires_exact_test_grids(monkeypatch):
    dataset = ArcAgiTaskGraphDataset()
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    monkeypatch.setattr(dataset, "_load", lambda _source_id: task)
    example = dataset.examples(split="test", limit=1, seed=0)[0]

    assert dataset.score(example, Prediction(answer="[[[4]]]")).correct
    assert not dataset.score(example, Prediction(answer="[[[3]]]")).correct


def test_remaining_adapters_select_frozen_sources_and_use_native_scores():
    entailment = EntailmentBankTaskGraphDataset()
    entailment._rows = [
        {
            "id": ENTAILMENT_ID,
            "hypothesis": "C.",
            "context": "sent1: A. sent2: B.",
            "proof": "sent1 & sent2 -> int1: C.",
            "depth_of_proof": 1,
            "length_of_proof": 1,
        }
    ]
    proof = entailment.examples(split="test", limit=None, seed=0)[0]
    assert entailment.score(
        proof,
        Prediction(answer="sent1 & sent2 -> int1: C."),
    ).correct

    dabstep = DABstepTaskGraphDataset(materialize_context=False)
    dabstep._rows = [
        {
            "task_id": source_id,
            "question": "Return the number.",
            "guidelines": "",
            "answer": "12.5",
            "level": "hard",
        }
        for source_id in ("2536", "2769")
    ]
    data_example = dabstep.examples(split="test", limit=1, seed=0)[0]
    assert dabstep.score(data_example, Prediction(answer="12.500000")).correct

    sudoku = DelegationSudokuDataset()
    sudoku_example = sudoku.examples(split="test", limit=None, seed=0)[0]
    assert "r4c0" in sudoku_example.prompt
    assert "r5c1" in sudoku_example.prompt
    assert "r6c0" not in sudoku_example.prompt
    assert "r7c1" not in sudoku_example.prompt
    assert sudoku.score(sudoku_example, Prediction(answer=SUDOKU_SOLUTION)).correct


def test_grsqa_keeps_hard_negative_private_from_gold_evidence():
    dataset = GRSQATaskGraphDataset()
    dataset._rows = [
        {
            "question": "Are both native?",
            "answer": ["no"],
            "pos_graph": {
                "nodes": [
                    {"node_id": 1, "evidence": "A is native.", "type": "comparison"},
                    {"node_id": 2, "evidence": "B is not native.", "type": "comparison"},
                ]
            },
            "neg_graph": [
                {"nodes": [{"node_id": 3, "evidence": "Distractor.", "type": "neg_para"}]}
            ],
        }
    ]
    example = dataset.examples(split="test", limit=None, seed=0)[0]

    assert example.expected["evidence"] == ["1", "2"]
    assert "neg:3" in example.context["context"]
    assert dataset.score(
        example,
        Prediction(answer='{"answer":"no","evidence":["1","2"]}'),
    ).correct


def test_natural_plan_freezes_each_task_type_and_native_parser():
    dataset = NaturalPlanTaskGraphDataset()
    dataset._rows = {
        "trip": [
            {
                "_source_id": NATURAL_PLAN_IDS["trip"],
                "prompt_0shot": "Plan.",
                "golden_plan": "",
                "cities": "A**B",
                "durations": "2**2",
            }
        ],
        "meeting": [
            {
                "_source_id": NATURAL_PLAN_IDS["meeting"],
                "prompt_0shot": "Meet.",
                "golden_plan": ["You start at X at 9:00AM"],
                "constraints": [["X", "9:00AM"]],
                "dist_matrix": {},
            }
        ],
        "calendar": [
            {
                "_source_id": NATURAL_PLAN_IDS["calendar"],
                "prompt_0shot": "Schedule.",
                "golden_plan": "Monday, 9:00 - 9:30",
            }
        ],
    }
    examples = dataset.examples(split="test", limit=None, seed=0)
    trip_answer = (
        "European cities for 3 days\n"
        "**Day 1-2:** Visit A\n"
        "On Day 2, fly from A to B\n"
        "**Day 2-3:** Visit B"
    )

    assert dataset.score(examples[0], Prediction(answer=trip_answer)).correct
    assert dataset.score(
        examples[1],
        Prediction(answer="You start at X at 9:00AM."),
    ).correct
    assert dataset.score(
        examples[2],
        Prediction(answer="Monday, 9:00 - 9:30"),
    ).correct


def test_codeqa_wrapper_freezes_the_selected_source_index():
    dataset = DelegationCodeQADataset()
    dataset._rows = [
        {
            "_source_index": CODEQA_INDEX,
            "question": "Which behavior is correct?",
            "context": "repository",
            "choice_A": "A behavior",
            "choice_B": "B behavior",
            "choice_C": "C behavior",
            "choice_D": "D behavior",
            "answer": "A",
        }
    ]
    example = dataset.examples(split="test", limit=None, seed=0)[0]

    assert example.metadata["source_id"] == f"train:{CODEQA_INDEX}"
    assert dataset.score(example, Prediction(answer="A")).correct


@pytest.mark.parametrize(
    ("value", "condition"),
    [
        ("local", DelegationCondition.LOCAL),
        ("capability-only", DelegationCondition.CAPABILITY_ONLY),
        ("current_policy", DelegationCondition.CURRENT_POLICY),
    ],
)
def test_delegation_condition_parser(value, condition):
    assert DelegationCondition.parse(value) is condition
