import pytest
from pydantic import BaseModel

from agent_runtime import Tool, ToolCall, ToolRegistry

from .conftest import ReadFileArgs, read_file


def test_spec_exposes_schema_from_args_model():
    spec = read_file.spec
    assert spec.name == "read_file"
    assert spec.description == "Read a file from the workspace."
    assert spec.parameters["properties"]["path"]["type"] == "string"
    assert spec.parameters["required"] == ["path"]


def test_tool_requires_description():
    with pytest.raises(ValueError, match="description"):
        Tool("t", "  ", ReadFileArgs, lambda a: "")


def test_duplicate_registration_is_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(read_file)


def test_execute_returns_tool_output(registry):
    result = registry.execute(ToolCall(id="1", name="read_file", arguments={"path": "notes.txt"}))
    assert result.content == "nothing here"
    assert not result.is_error
    assert result.call_id == "1"


def test_defaults_and_coercion_apply(registry):
    result = registry.execute(ToolCall(id="1", name="search_code", arguments={"pattern": "add"}))
    assert result.content == "app.py"


def test_missing_argument_becomes_error_result(registry):
    result = registry.execute(ToolCall(id="1", name="read_file", arguments={}))
    assert result.is_error
    assert "path" in result.content


def test_wrong_argument_type_becomes_error_result(registry):
    call = ToolCall(id="1", name="search_code", arguments={"pattern": "x", "max_results": "many"})
    result = registry.execute(call)
    assert result.is_error
    assert "max_results" in result.content


def test_unknown_tool_lists_available_tools(registry):
    result = registry.execute(ToolCall(id="1", name="delete_all", arguments={}))
    assert result.is_error
    assert "read_file" in result.content and "search_code" in result.content


def test_tool_exception_is_contained(registry):
    result = registry.execute(ToolCall(id="1", name="read_file", arguments={"path": "missing"}))
    assert result.is_error
    assert result.content.startswith("FileNotFoundError")


def test_decorator_uses_docstring_and_custom_name():
    class Args(BaseModel):
        x: int

    from agent_runtime import tool

    @tool(Args, name="double")
    def f(args: Args) -> int:
        """Double a number."""
        return args.x * 2

    assert f.name == "double"
    result = ToolRegistry([f]).execute(ToolCall(id="1", name="double", arguments={"x": 4}))
    assert result.content == "8"
