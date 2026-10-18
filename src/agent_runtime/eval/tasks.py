"""Benchmark definition: tasks, success criteria and the fixture workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExpectedCall(BaseModel):
    """A tool call the task should lead to; ``arguments`` need only be a subset of the real ones."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Fault(BaseModel):
    """Make a tool raise a transient error this many times before it works."""

    tool: str
    transient_failures: int


class ScriptedCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ScriptedTurn(BaseModel):
    content: str = ""
    tool_calls: list[ScriptedCall] = Field(default_factory=list)
    prompt_tokens: int = 100
    completion_tokens: int = 50


class Expect(BaseModel):
    status: str = "completed"
    stop_reason: str | None = None
    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    # some tool output in the run must contain each of these strings
    tool_output_contains: list[str] = Field(default_factory=list)


class EvalTask(BaseModel):
    id: str
    # task: needs a correct answer; recovery: same, after an injected or induced failure;
    # boundary: a limit must stop the run
    category: Literal["task", "recovery", "boundary"]
    description: str = ""
    prompt: str
    expect: Expect = Field(default_factory=Expect)
    expected_calls: list[ExpectedCall] = Field(default_factory=list)
    faults: list[Fault] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    # scripted model behaviour, used only by simulated runs
    script: list[ScriptedTurn] | None = None
    # the task depends on the scripted model misbehaving on purpose (wrong answer, bad
    # arguments, endless loop), which a real model cannot be forced to do; skipped in live runs
    scripted_only: bool = False


class PricingConfig(BaseModel):
    prompt_per_million: float
    completion_per_million: float


class Benchmark(BaseModel):
    name: str
    description: str = ""
    workspace: dict[str, str]
    pricing: PricingConfig | None = None
    tasks: list[EvalTask]


def load_benchmark(path: str | Path) -> Benchmark:
    return Benchmark.model_validate_json(Path(path).read_text(encoding="utf-8"))


def live_tasks(benchmark: Benchmark) -> list[EvalTask]:
    return [t for t in benchmark.tasks if not t.scripted_only]
