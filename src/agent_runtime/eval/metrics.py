"""Aggregate task outcomes into the benchmark metrics."""

from __future__ import annotations

import statistics
from typing import Any

from agent_runtime.eval.runner import TaskOutcome


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def compute_metrics(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    answerable = [o for o in outcomes if o.category in ("task", "recovery")]
    recovery = [o for o in outcomes if o.category == "recovery"]
    boundary = [o for o in outcomes if o.category == "boundary"]
    calls = sum(o.tool_calls for o in outcomes)
    expected = sum(o.expected_calls for o in outcomes)
    costs = [o.cost_usd for o in outcomes if o.cost_usd is not None]

    return {
        "tasks": {
            "total": len(outcomes),
            "task": sum(o.category == "task" for o in outcomes),
            "recovery": len(recovery),
            "boundary": len(boundary),
        },
        "task_success_rate": _rate(sum(o.passed for o in answerable), len(answerable)),
        "recovery_rate": _rate(sum(o.passed for o in recovery), len(recovery)),
        "boundary_pass_rate": _rate(sum(o.passed for o in boundary), len(boundary)),
        "tool_call_validity": _rate(calls - sum(o.invalid_calls for o in outcomes), calls),
        "tool_selection_accuracy": _rate(sum(o.matched_calls for o in outcomes), expected),
        "tool_calls_total": calls,
        "steps": _summary([o.steps for o in outcomes]),
        "tool_calls_per_task": _summary([o.tool_calls for o in outcomes]),
        "tokens_per_task": _summary([o.prompt_tokens + o.completion_tokens for o in outcomes]),
        "cost_usd_total": sum(costs) if costs else None,
        "elapsed_s_per_task": _summary([o.elapsed_s for o in outcomes if o.elapsed_s is not None]),
    }
