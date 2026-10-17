import pytest
from pydantic import BaseModel

from agent_runtime import AssistantTurn, ToolCall, ToolRegistry, Usage, tool


class ReadFileArgs(BaseModel):
    path: str


class SearchArgs(BaseModel):
    pattern: str
    max_results: int = 5


FILES = {"app.py": "def add(a, b):\n    return a - b\n", "notes.txt": "nothing here"}


@tool(ReadFileArgs)
def read_file(args: ReadFileArgs) -> str:
    """Read a file from the workspace."""
    if args.path not in FILES:
        raise FileNotFoundError(args.path)
    return FILES[args.path]


@tool(SearchArgs)
def search_code(args: SearchArgs) -> str:
    """Return the names of files containing the pattern."""
    hits = [name for name, text in FILES.items() if args.pattern in text]
    return ",".join(hits[: args.max_results]) or "no matches"


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([read_file, search_code])


def call_turn(call_id: str, name: str, **arguments) -> AssistantTurn:
    return AssistantTurn(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def answer_turn(text: str) -> AssistantTurn:
    return AssistantTurn(content=text, usage=Usage(prompt_tokens=10, completion_tokens=5))
