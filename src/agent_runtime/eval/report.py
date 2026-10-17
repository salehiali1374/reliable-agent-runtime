"""Markdown rendering of a benchmark result."""

from __future__ import annotations

from typing import Any

from agent_runtime.eval.runner import TaskOutcome

MODE_NOTES = {
    "simulated": (
        "Simulated run: the model is a fixed script, so these numbers verify the runtime's "
        "handling of tools, errors and limits. They say nothing about the quality of any "
        "real model."
    ),
    "live": "Live run: a real model produced every decision. Results depend on the model.",
}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _stat(summary: dict[str, float] | None, fmt: str = "{:.1f}") -> str:
    if summary is None:
        return "n/a"
    return f"mean {fmt.format(summary['mean'])}, median {fmt.format(summary['median'])}"


def render_markdown(
    name: str, mode: str, model: str | None, metrics: dict[str, Any], outcomes: list[TaskOutcome]
) -> str:
    t = metrics["tasks"]
    total_cost = metrics["cost_usd_total"]
    cost = "n/a (no pricing)" if total_cost is None else f"{total_cost:.4f}"
    lines = [
        f"# Benchmark `{name}` ({mode})",
        "",
        MODE_NOTES[mode],
        "",
        f"Model: {model or 'scripted'}. Tasks: {t['total']} "
        f"({t['task']} task, {t['recovery']} recovery, {t['boundary']} boundary).",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Task success rate | {_pct(metrics['task_success_rate'])} |",
        f"| Recovery rate | {_pct(metrics['recovery_rate'])} |",
        f"| Boundary pass rate | {_pct(metrics['boundary_pass_rate'])} |",
        f"| Tool-call validity | {_pct(metrics['tool_call_validity'])} |",
        f"| Tool-selection accuracy | {_pct(metrics['tool_selection_accuracy'])} |",
        f"| Steps per task | {_stat(metrics['steps'])} |",
        f"| Tool calls per task | {_stat(metrics['tool_calls_per_task'])} |",
        f"| Tokens per task | {_stat(metrics['tokens_per_task'], '{:.0f}')} |",
        f"| Latency per task (s) | {_stat(metrics['elapsed_s_per_task'], '{:.2f}')} |",
        f"| Cost (USD, total) | {cost} |",
        "",
        "## Tasks",
        "",
        "| Task | Category | Result | Status | Steps | Calls | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for o in outcomes:
        status = o.status + (f" ({o.stop_reason})" if o.stop_reason else "")
        notes = "; ".join(o.failures) if o.failures else ""
        lines.append(
            f"| {o.task_id} | {o.category} | {'pass' if o.passed else 'FAIL'} | {status} "
            f"| {o.steps} | {o.tool_calls} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)
