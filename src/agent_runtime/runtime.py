"""The agent loop: ask the model, execute the tools it requests, feed results back."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from agent_runtime.budget import Budget, Pricing
from agent_runtime.llm import LLMClient, LLMError, TransientLLMError
from agent_runtime.messages import Message
from agent_runtime.reliability import RetryFailed, RetryPolicy, retry_call
from agent_runtime.state import CheckpointStore, RunState
from agent_runtime.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a task-solving agent. Use the available tools to gather evidence, "
    "and answer only from what the tools returned. When you have the answer, "
    "reply without calling a tool."
)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        budget: Budget | None = None,
        pricing: Pricing | None = None,
        retry: RetryPolicy | None = None,
        store: CheckpointStore | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.budget = budget or Budget()
        if self.budget.max_cost_usd is not None and pricing is None:
            raise ValueError("max_cost_usd needs a pricing")
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.pricing = pricing
        self.retry = retry or RetryPolicy()
        self.store = store
        self.clock = clock

    def run(self, task: str, *, run_id: str | None = None) -> RunState:
        state = RunState(
            run_id=run_id or uuid.uuid4().hex,
            task=task,
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=task),
            ],
        )
        self._save(state)
        return self._loop(state)

    def resume(self, run_id: str) -> RunState:
        """Continue a run from its last checkpoint. Finished runs are returned unchanged."""
        if self.store is None:
            raise ValueError("resume needs a checkpoint store")
        state = self.store.load(run_id)
        if state is None:
            raise KeyError(f"no checkpoint for run {run_id!r}")
        if state.status != "running":
            return state
        return self._loop(state)

    def _loop(self, state: RunState) -> RunState:
        started = self.clock()
        elapsed_before = state.elapsed_seconds
        specs = self.registry.specs()

        while True:
            state.elapsed_seconds = elapsed_before + (self.clock() - started)
            reason = self.budget.check(state, self.pricing)
            if reason:
                return self._finish(state, "budget_exceeded", stop_reason=reason)

            try:
                turn, _ = retry_call(
                    lambda: self.llm.complete(state.messages, specs),
                    self.retry,
                    lambda exc: isinstance(exc, TransientLLMError),
                )
            except RetryFailed as failure:
                if not isinstance(failure.last, LLMError):
                    raise failure.last from None
                return self._finish(state, "failed", error=str(failure), stop_reason="llm_error")

            state.steps += 1
            state.usage = state.usage + turn.usage
            state.messages.append(
                Message(role="assistant", content=turn.content, tool_calls=turn.tool_calls)
            )
            if not turn.tool_calls:
                state.answer = turn.content
                return self._finish(state, "completed")

            for call in turn.tool_calls:
                if (
                    self.budget.max_tool_calls is not None
                    and state.tool_calls >= self.budget.max_tool_calls
                ):
                    # every tool call needs a matching result to keep the history valid
                    state.messages.append(
                        Message(
                            role="tool",
                            content="tool call budget exhausted; this call was not executed",
                            tool_call_id=call.id,
                            is_error=True,
                        )
                    )
                    continue
                result = self.registry.execute(call)
                state.tool_calls += 1
                state.messages.append(
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.call_id,
                        is_error=result.is_error,
                    )
                )
            self._save(state)

    def _finish(
        self,
        state: RunState,
        status: str,
        *,
        stop_reason: str | None = None,
        error: str | None = None,
    ) -> RunState:
        state.status = status  # type: ignore[assignment]
        state.stop_reason = stop_reason
        state.error = error
        self._save(state)
        return state

    def _save(self, state: RunState) -> None:
        if self.store is not None:
            self.store.save(state)
