"""Tool definition, argument validation and the registry the runtime executes against."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_runtime.messages import ToolCall


class ToolSpec(BaseModel):
    """What the model sees about a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    call_id: str
    content: str
    is_error: bool = False


class ToolInputError(ValueError):
    """The model supplied arguments that do not match the tool's schema."""


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        fn: Callable[[Any], Any],
    ):
        if not description.strip():
            raise ValueError(f"tool {name!r} needs a description; the model selects tools by it")
        self.name = name
        self.description = description.strip()
        self.args_model = args_model
        self._fn = fn

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )

    def run(self, arguments: dict[str, Any]) -> str:
        try:
            args = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputError(_format_validation_error(exc)) from exc
        return str(self._fn(args))


def tool(
    args_model: type[BaseModel], *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[[Any], Any]], Tool]:
    """Turn a function taking a validated ``args_model`` instance into a ``Tool``."""

    def decorate(fn: Callable[[Any], Any]) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            args_model=args_model,
            fn=fn,
        )

    return decorate


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"tool {t.name!r} is already registered")
        self._tools[t.name] = t

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a call. Failures become error results the model can react to, never exceptions."""
        t = self._tools.get(call.name)
        if t is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return ToolResult(
                call_id=call.id,
                content=f"unknown tool {call.name!r}; available tools: {known}",
                is_error=True,
            )
        try:
            return ToolResult(call_id=call.id, content=t.run(call.arguments))
        except ToolInputError as exc:
            return ToolResult(call_id=call.id, content=f"invalid arguments: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - tool code is untrusted by the loop
            return ToolResult(
                call_id=call.id, content=f"{type(exc).__name__}: {exc}", is_error=True
            )
