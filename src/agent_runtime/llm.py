"""Provider-independent model interface and a scripted implementation for tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent_runtime.messages import AssistantTurn, Message
from agent_runtime.tools import ToolSpec


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn: ...


class ScriptExhaustedError(RuntimeError):
    """The scripted model was asked for more turns than it was given."""


class ScriptedLLM:
    """Replays a fixed list of turns and records what it was shown."""

    def __init__(self, turns: Sequence[AssistantTurn]):
        self._turns = list(turns)
        self.requests: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        self.requests.append(list(messages))
        if len(self.requests) > len(self._turns):
            raise ScriptExhaustedError(f"script has {len(self._turns)} turns")
        return self._turns[len(self.requests) - 1]
