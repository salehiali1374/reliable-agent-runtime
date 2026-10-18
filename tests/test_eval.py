import json
from pathlib import Path

import pytest

from agent_runtime import MemoryTracer, ToolCall
from agent_runtime.eval.__main__ import main
from agent_runtime.eval.metrics import compute_metrics
from agent_runtime.eval.runner import run_benchmark, run_task, scripted_llm
from agent_runtime.eval.tasks import (
    Benchmark,
    EvalTask,
    Fault,
    PricingConfig,
    live_tasks,
    load_benchmark,
)
from agent_runtime.eval.workspace import build_registry

BENCHMARK = Path(__file__).parent.parent / "benchmarks" / "triage.json"
EXPECTED_FAILURES = {"T05-wrong-diagnosis", "T06-ungrounded-answer", "R05-gives-up"}


@pytest.fixture(scope="module")
def benchmark() -> Benchmark:
    return load_benchmark(BENCHMARK)


@pytest.fixture(scope="module")
def outcomes(benchmark):
    return run_benchmark(benchmark, scripted_llm, measure_time=False)


def test_benchmark_file_is_well_formed(benchmark):
    ids = [t.id for t in benchmark.tasks]
    assert len(ids) == len(set(ids))
    assert {t.category for t in benchmark.tasks} == {"task", "recovery", "boundary"}
    assert all(t.script for t in benchmark.tasks)


def test_each_task_gets_its_designed_verdict(outcomes):
    failed = {o.task_id for o in outcomes if not o.passed}
    assert failed == EXPECTED_FAILURES


def test_failing_tasks_report_why(outcomes):
    by_id = {o.task_id: o for o in outcomes}
    assert any("answer lacks" in f for f in by_id["T05-wrong-diagnosis"].failures)
    assert by_id["T06-ungrounded-answer"].tool_calls == 0


def test_metrics_match_hand_computed_values(outcomes):
    m = compute_metrics(outcomes)
    assert m["tasks"] == {"total": 16, "task": 6, "recovery": 5, "boundary": 5}
    assert m["task_success_rate"] == pytest.approx(8 / 11)
    assert m["recovery_rate"] == pytest.approx(4 / 5)
    assert m["boundary_pass_rate"] == 1.0
    assert m["tool_call_validity"] == pytest.approx(32 / 34)
    assert m["tool_selection_accuracy"] == pytest.approx(12 / 15)
    assert m["cost_usd_total"] is None
    assert m["elapsed_s_per_task"] is None  # time is not measured for scripted runs


def test_simulated_run_is_deterministic(benchmark):
    first = [o.to_dict() for o in run_benchmark(benchmark, scripted_llm, measure_time=False)]
    second = [o.to_dict() for o in run_benchmark(benchmark, scripted_llm, measure_time=False)]
    assert first == second


def test_recovery_task_without_any_failure_is_not_counted_as_recovered(benchmark):
    task = next(t for t in benchmark.tasks if t.id == "R03-transient-fault")
    no_fault = task.model_copy(update={"faults": []})
    outcome = run_task(no_fault, benchmark, scripted_llm(no_fault))
    assert not outcome.passed
    assert any("nothing was recovered" in f for f in outcome.failures)


def test_cost_is_computed_when_pricing_is_configured(benchmark):
    priced = benchmark.model_copy(
        update={"pricing": PricingConfig(prompt_per_million=1000.0, completion_per_million=2000.0)}
    )
    task = priced.tasks[0]
    outcome = run_task(task, priced, scripted_llm(task))
    expected = (outcome.prompt_tokens * 1000 + outcome.completion_tokens * 2000) / 1e6
    assert outcome.cost_usd == pytest.approx(expected)


def test_task_without_script_cannot_run_simulated():
    task = EvalTask(id="x", category="task", prompt="p")
    with pytest.raises(ValueError, match="no script"):
        scripted_llm(task)


def test_workspace_refuses_paths_outside_it(benchmark):
    registry = build_registry(benchmark.workspace, [])
    for path in ("../secret", "/etc/passwd", "calc/../../x"):
        result = registry.execute(ToolCall(id="1", name="read_file", arguments={"path": path}))
        assert result.is_error
        assert "outside the workspace" in result.content


def test_workspace_search_reports_path_and_line(benchmark):
    registry = build_registry(benchmark.workspace, [])
    result = registry.execute(ToolCall(id="1", name="search_code", arguments={"pattern": "a - b"}))
    assert result.content == "calc/ops.py:2: return a - b"


def test_faults_are_fresh_for_every_registry(benchmark):
    faults = [Fault(tool="read_file", transient_failures=1)]
    for _ in range(2):
        registry = build_registry(benchmark.workspace, faults)
        result = registry.execute(
            ToolCall(id="1", name="read_file", arguments={"path": "calc/ops.py"})
        )
        assert result.attempts == 2 and not result.is_error


def test_faults_must_name_real_tools(benchmark):
    with pytest.raises(ValueError, match="unknown tools"):
        build_registry(benchmark.workspace, [Fault(tool="nope", transient_failures=1)])


def test_events_can_be_collected_through_an_extra_tracer(benchmark):
    extra = MemoryTracer()
    task = benchmark.tasks[0]
    run_task(task, benchmark, scripted_llm(task), tracer=extra)
    assert {e.type for e in extra.events} >= {"llm_call", "tool_call", "run_finished"}


def test_cli_writes_json_and_markdown(tmp_path, capsys):
    assert main([str(BENCHMARK), "--out", str(tmp_path), "--trace-db", str(tmp_path / "t.db")]) == 0
    result = json.loads((tmp_path / "sim_triage.json").read_text())
    assert result["mode"] == "simulated"
    assert result["metrics"]["tasks"]["total"] == 16
    assert (tmp_path / "sim_triage.md").read_text().startswith("# Benchmark `triage` (simulated)")
    assert (tmp_path / "t.db").exists()


def test_live_mode_requires_endpoint_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    assert main([str(BENCHMARK), "--mode", "live", "--out", str(tmp_path)]) == 2
    assert "AGENT_LLM_BASE_URL" in capsys.readouterr().err


def test_live_selection_keeps_only_tasks_a_real_model_can_be_judged_on(benchmark):
    live_ids = {t.id for t in live_tasks(benchmark)}
    assert live_ids == {
        "T01-add-bug",
        "T02-mean-cascade",
        "T03-empty-mean",
        "T04-full-report",
        "R03-transient-fault",
        "R04-fallback-after-outage",
    }
    assert all(t.category != "boundary" for t in live_tasks(benchmark))
