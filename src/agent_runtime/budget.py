"""Limits that keep a run bounded."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from agent_runtime.messages import Usage
from agent_runtime.state import RunState


@dataclass(frozen=True)
class Pricing:
    """Price per million tokens, used to turn token usage into cost."""

    prompt_per_million: float
    completion_per_million: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.prompt_tokens * self.prompt_per_million
            + usage.completion_tokens * self.completion_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class Budget:
    """Hard limits for one run. ``None`` disables a limit."""

    max_steps: int = 20
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    # stop when the same tool is called with the same arguments this many times
    max_identical_calls: int | None = 3

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

    def check(self, state: RunState, pricing: Pricing | None = None) -> str | None:
        """Return the reason the run must stop now, or ``None`` if it may continue."""
        if state.steps >= self.max_steps:
            return "max_steps"
        if self.max_tool_calls is not None and state.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.max_tokens is not None:
            used = state.usage.prompt_tokens + state.usage.completion_tokens
            if used >= self.max_tokens:
                return "max_tokens"
        if self.max_cost_usd is not None and pricing is not None:
            if pricing.cost(state.usage) >= self.max_cost_usd:
                return "max_cost"
        if self.max_seconds is not None and state.elapsed_seconds >= self.max_seconds:
            return "max_seconds"
        if self.max_identical_calls is not None and _most_repeated_call(state) >= (
            self.max_identical_calls
        ):
            return "repeated_call"
        return None


def _most_repeated_call(state: RunState) -> int:
    counts = Counter(
        (call.name, json.dumps(call.arguments, sort_keys=True))
        for message in state.messages
        for call in message.tool_calls
    )
    return max(counts.values(), default=0)
