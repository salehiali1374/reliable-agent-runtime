"""Read-only tools over an in-memory workspace, with optional fault injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agent_runtime.eval.tasks import Fault
from agent_runtime.reliability import RetryPolicy
from agent_runtime.tools import Tool, ToolRegistry, TransientToolError, tool


class PathArgs(BaseModel):
    path: str


class SearchArgs(BaseModel):
    pattern: str


class NoArgs(BaseModel):
    pass


def _check_path(path: str, files: dict[str, str]) -> str:
    if path.startswith("/") or ".." in path.split("/"):
        raise PermissionError(f"path {path!r} is outside the workspace")
    if path not in files:
        raise FileNotFoundError(f"no such file: {path}")
    return path


def workspace_tools(files: dict[str, str]) -> list[Tool]:
    @tool(NoArgs)
    def list_files(args: NoArgs) -> str:
        """List every file in the workspace."""
        return "\n".join(sorted(files))

    @tool(PathArgs)
    def read_file(args: PathArgs) -> str:
        """Read the full text of one workspace file."""
        return files[_check_path(args.path, files)]

    @tool(SearchArgs)
    def search_code(args: SearchArgs) -> str:
        """Find lines containing a text pattern; returns path:line: text for each match."""
        hits = [
            f"{path}:{number}: {line.strip()}"
            for path, text in sorted(files.items())
            for number, line in enumerate(text.splitlines(), start=1)
            if args.pattern in line
        ]
        return "\n".join(hits) or "no matches"

    return [list_files, read_file, search_code]


def _with_faults(t: Tool, failures: int) -> Tool:
    remaining = {"n": failures}
    inner: Callable[[Any], Any] = t._fn

    def flaky(args: Any) -> Any:
        if remaining["n"] > 0:
            remaining["n"] -= 1
            raise TransientToolError("temporary failure injected by the benchmark")
        return inner(args)

    return Tool(t.name, t.description, t.args_model, flaky, timeout=t.timeout)


def build_registry(files: dict[str, str], faults: list[Fault]) -> ToolRegistry:
    """Tools for one task run. Fault counters are fresh per call, so runs do not interact."""
    failures = {f.tool: f.transient_failures for f in faults}
    tools = [
        _with_faults(t, failures[t.name]) if t.name in failures else t
        for t in workspace_tools(files)
    ]
    unknown = set(failures) - {t.name for t in tools}
    if unknown:
        raise ValueError(f"faults reference unknown tools: {sorted(unknown)}")
    # in-process tools: backoff delays would only slow the benchmark down
    return ToolRegistry(tools, default_timeout=30.0, retry=RetryPolicy(sleep=lambda _: None))
