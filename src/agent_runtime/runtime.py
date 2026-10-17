"""The agent loop: ask the model, execute the tools it requests, feed results back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent_runtime.llm import LLMClient
from agent_runtime.messages import Message, Usage
from agent_runtime.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a task-solving agent. Use the available tools to gather evidence, "
    "and answer only from what the tools returned. When you have the answer, "
    "reply without calling a tool."
)


@dataclass
class RunResult:
    status: Literal["completed", "max_steps"]
    answer: str | None
    messages: list[Message]
    steps: int
    usage: Usage = field(default_factory=Usage)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, task: str) -> RunResult:
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task),
        ]
        usage = Usage()
        specs = self.registry.specs()

        for step in range(1, self.max_steps + 1):
            turn = self.llm.complete(messages, specs)
            usage = usage + turn.usage
            messages.append(
                Message(role="assistant", content=turn.content, tool_calls=turn.tool_calls)
            )
            if not turn.tool_calls:
                return RunResult("completed", turn.content, messages, step, usage)

            for call in turn.tool_calls:
                result = self.registry.execute(call)
                messages.append(
                    Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=result.call_id,
                        is_error=result.is_error,
                    )
                )

        return RunResult("max_steps", None, messages, self.max_steps, usage)
