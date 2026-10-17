"""Run benchmark tasks and judge each outcome against its declared criteria."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from agent_runtime.budget import Budget, Pricing
from agent_runtime.eval.tasks import Benchmark, EvalTask
from agent_runtime.eval.workspace import build_registry
from agent_runtime.llm import LLMClient, ScriptedLLM
from agent_runtime.messages import AssistantTurn, ToolCall, Usage
from agent_runtime.runtime import Agent
from agent_runtime.state import RunState
from agent_runtime.trace import MemoryTracer, TraceEvent, Tracer

DEFAULT_BUDGET = Budget(max_steps=15, max_tokens=60_000, max_seconds=300)
INVALID_CALL_KINDS = {"invalid_arguments", "unknown_tool"}


@dataclass
class TaskOutcome:
    task_id: str
    category: str
    passed: bool
    failures: list[str]
    status: str
    stop_reason: str | None
    steps: int
    tool_calls: int
    invalid_calls: int
    error_results: int
    expected_calls: int
    matched_calls: int
    retried_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    elapsed_s: float | None
    answer: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def scripted_llm(task: EvalTask) -> ScriptedLLM:
    if task.script is None:
        raise ValueError(f"task {task.id} has no script; it cannot run in simulated mode")
    turns = [
        AssistantTurn(
            content=turn.content,
            tool_calls=[
                ToolCall(id=f"c{i}_{j}", name=c.name, arguments=c.arguments)
                for j, c in enumerate(turn.tool_calls)
            ],
            usage=Usage(prompt_tokens=turn.prompt_tokens, completion_tokens=turn.completion_tokens),
        )
        for i, turn in enumerate(task.script)
    ]
    return ScriptedLLM(turns)


def _matches(expected: dict, actual: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _judge(task: EvalTask, state: RunState, tool_events: list[TraceEvent]) -> list[str]:
    expect = task.expect
    problems = []
    if state.status != expect.status:
        problems.append(f"status {state.status!r}, expected {expect.status!r}")
    if expect.stop_reason is not None and state.stop_reason != expect.stop_reason:
        problems.append(f"stop_reason {state.stop_reason!r}, expected {expect.stop_reason!r}")
    answer = (state.answer or "").lower()
    for needle in expect.answer_contains:
        if needle.lower() not in answer:
            problems.append(f"answer lacks {needle!r}")
    for needle in expect.answer_not_contains:
        if needle.lower() in answer:
            problems.append(f"answer contains forbidden {needle!r}")
    outputs = " ".join(str(e.data.get("output", "")) for e in tool_events).lower()
    for needle in expect.tool_output_contains:
        if needle.lower() not in outputs:
            problems.append(f"no tool output contains {needle!r}")
    return problems


def run_task(
    task: EvalTask,
    benchmark: Benchmark,
    llm: LLMClient,
    *,
    tracer: Tracer | None = None,
    measure_time: bool = True,
) -> TaskOutcome:
    memory = MemoryTracer()
    pricing = (
        Pricing(benchmark.pricing.prompt_per_million, benchmark.pricing.completion_per_million)
        if benchmark.pricing
        else None
    )
    budget = Budget(**{**DEFAULT_BUDGET.__dict__, **task.budget})
    agent = Agent(
        llm,
        build_registry(benchmark.workspace, task.faults),
        budget=budget,
        pricing=pricing,
        tracer=_Fanout(memory, tracer) if tracer else memory,
        # a scripted model has no meaningful latency; a frozen clock keeps results reproducible
        **({} if measure_time else {"clock": lambda: 0.0}),
    )
    state = agent.run(task.prompt, run_id=f"{benchmark.name}:{task.id}")

    events = memory.for_run(state.run_id)
    tool_events = [e for e in events if e.type == "tool_call"]
    problems = _judge(task, state, tool_events)

    # calls skipped by the tool-call budget were never executed
    actual_calls = [e.data for e in tool_events if e.data["error_kind"] != "budget"]
    matched = sum(
        any(
            c["name"] == want.name and _matches(want.arguments, c["arguments"])
            for c in actual_calls
        )
        for want in task.expected_calls
    )
    retried = sum(1 for c in actual_calls if c["attempts"] > 1)
    errors = sum(1 for c in actual_calls if c["is_error"])
    if task.category == "recovery" and not (retried or errors):
        problems.append("no failure occurred, so nothing was recovered from")

    return TaskOutcome(
        task_id=task.id,
        category=task.category,
        passed=not problems,
        failures=problems,
        status=state.status,
        stop_reason=state.stop_reason,
        steps=state.steps,
        tool_calls=len(actual_calls),
        invalid_calls=sum(1 for c in actual_calls if c["error_kind"] in INVALID_CALL_KINDS),
        error_results=errors,
        expected_calls=len(task.expected_calls),
        matched_calls=matched,
        retried_calls=retried,
        prompt_tokens=state.usage.prompt_tokens,
        completion_tokens=state.usage.completion_tokens,
        cost_usd=pricing.cost(state.usage) if pricing else None,
        elapsed_s=state.elapsed_seconds if measure_time else None,
        answer=state.answer,
    )


class _Fanout:
    def __init__(self, *tracers: Tracer):
        self._tracers = tracers

    def emit(self, run_id: str, type: str, **data) -> None:
        for t in self._tracers:
            t.emit(run_id, type, **data)


def run_benchmark(
    benchmark: Benchmark,
    llm_for: Callable[[EvalTask], LLMClient],
    *,
    tracer: Tracer | None = None,
    measure_time: bool = True,
) -> list[TaskOutcome]:
    return [
        run_task(t, benchmark, llm_for(t), tracer=tracer, measure_time=measure_time)
        for t in benchmark.tasks
    ]
