"""Tool definition, argument validation and the registry the runtime executes against."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_runtime.messages import ToolCall
from agent_runtime.reliability import RetryFailed, RetryPolicy, retry_call


class ToolSpec(BaseModel):
    """What the model sees about a tool."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    call_id: str
    content: str
    is_error: bool = False
    attempts: int = 1
    # unknown_tool | invalid_arguments | timeout | transient | tool_error; None when successful
    error_kind: str | None = None


class ToolInputError(ValueError):
    """The model supplied arguments that do not match the tool's schema."""


class TransientToolError(Exception):
    """A tool failure worth retrying, such as a momentary network or resource error."""


class ToolTimeoutError(Exception):
    """A tool call exceeded its time limit."""


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, TransientToolError | ToolTimeoutError)


def _error_kind(exc: Exception) -> str:
    if isinstance(exc, ToolTimeoutError):
        return "timeout"
    if isinstance(exc, TransientToolError):
        return "transient"
    return "tool_error"


def _run_with_timeout(fn: Callable[[], str], timeout: float | None, name: str) -> str:
    """Run ``fn`` in a daemon thread and give up waiting after ``timeout`` seconds.

    A Python thread cannot be killed, so a timed-out call keeps running in the background.
    Tools that need hard cancellation should run their work in a subprocess.
    """
    if timeout is None:
        return fn()

    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the calling thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise ToolTimeoutError(f"tool {name!r} timed out after {timeout}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


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
        timeout: float | None = None,
    ):
        if not description.strip():
            raise ValueError(f"tool {name!r} needs a description; the model selects tools by it")
        self.name = name
        self.description = description.strip()
        self.args_model = args_model
        self.timeout = timeout
        self._fn = fn

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        try:
            return self.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInputError(_format_validation_error(exc)) from exc

    def invoke(self, args: BaseModel) -> str:
        return str(self._fn(args))

    def run(self, arguments: dict[str, Any]) -> str:
        return self.invoke(self.validate(arguments))


def tool(
    args_model: type[BaseModel],
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
) -> Callable[[Callable[[Any], Any]], Tool]:
    """Turn a function taking a validated ``args_model`` instance into a ``Tool``."""

    def decorate(fn: Callable[[Any], Any]) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            args_model=args_model,
            fn=fn,
            timeout=timeout,
        )

    return decorate


class ToolRegistry:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        default_timeout: float | None = None,
        retry: RetryPolicy | None = None,
    ):
        self.default_timeout = default_timeout
        self.retry = retry or RetryPolicy()
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
                error_kind="unknown_tool",
            )
        if call.parse_error:
            return ToolResult(
                call_id=call.id,
                content=f"invalid arguments: not valid JSON ({call.parse_error})",
                is_error=True,
                error_kind="invalid_arguments",
            )
        try:
            args = t.validate(call.arguments)
        except ToolInputError as exc:
            return ToolResult(
                call_id=call.id,
                content=f"invalid arguments: {exc}",
                is_error=True,
                error_kind="invalid_arguments",
            )

        timeout = t.timeout if t.timeout is not None else self.default_timeout
        try:
            content, attempts = retry_call(
                lambda: _run_with_timeout(lambda: t.invoke(args), timeout, t.name),
                self.retry,
                _is_transient,
            )
        except RetryFailed as failure:
            note = f" (gave up after {failure.attempts} attempts)" if failure.attempts > 1 else ""
            return ToolResult(
                call_id=call.id,
                content=f"{type(failure.last).__name__}: {failure.last}{note}",
                is_error=True,
                attempts=failure.attempts,
                error_kind=_error_kind(failure.last),
            )
        return ToolResult(call_id=call.id, content=content, attempts=attempts)
