"""Conversation data model shared by the runtime, the LLM adapters and the tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # set by adapters when the model's raw arguments could not be parsed as JSON
    parse_error: str | None = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # assistant messages: the tools the model asked for
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # tool messages: which call this is the result of
    tool_call_id: str | None = None
    is_error: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class AssistantTurn(BaseModel):
    """One model response: either tool calls to run or a final answer."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
