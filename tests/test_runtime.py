import pytest

from agent_runtime import Agent, ScriptedLLM
from agent_runtime.llm import ScriptExhaustedError

from .conftest import answer_turn, call_turn


def test_multi_step_task_completes(registry):
    llm = ScriptedLLM(
        [
            call_turn("c1", "search_code", pattern="add"),
            call_turn("c2", "read_file", path="app.py"),
            answer_turn("add() subtracts instead of adding (app.py)."),
        ]
    )
    result = Agent(llm, registry).run("why does add fail?")

    assert result.status == "completed"
    assert result.steps == 3
    assert result.answer == "add() subtracts instead of adding (app.py)."
    assert [m.role for m in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.messages[3].content == "app.py"
    assert result.messages[5].content.startswith("def add")


def test_model_sees_tool_specs_and_prior_results(registry):
    llm = ScriptedLLM([call_turn("c1", "read_file", path="notes.txt"), answer_turn("done")])
    Agent(llm, registry).run("read notes")

    second_request = llm.requests[1]
    tool_msg = second_request[-1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "c1"
    assert tool_msg.content == "nothing here"


def test_invalid_arguments_are_fed_back_and_model_recovers(registry):
    llm = ScriptedLLM(
        [
            call_turn("c1", "read_file"),  # missing path
            call_turn("c2", "read_file", path="app.py"),
            answer_turn("ok"),
        ]
    )
    result = Agent(llm, registry).run("read app")

    assert result.status == "completed"
    bad = result.messages[3]
    assert bad.is_error and "path" in bad.content
    assert not result.messages[5].is_error


def test_unknown_tool_does_not_crash_the_loop(registry):
    llm = ScriptedLLM([call_turn("c1", "rm_rf", path="/"), answer_turn("cannot do that")])
    result = Agent(llm, registry).run("clean up")

    assert result.status == "completed"
    assert result.messages[3].is_error


def test_stops_at_max_steps(registry):
    llm = ScriptedLLM([call_turn(f"c{i}", "read_file", path="notes.txt") for i in range(10)])
    result = Agent(llm, registry, max_steps=3).run("loop forever")

    assert result.status == "max_steps"
    assert result.answer is None
    assert result.steps == 3
    assert len(llm.requests) == 3


def test_usage_is_accumulated(registry):
    llm = ScriptedLLM([call_turn("c1", "read_file", path="notes.txt"), answer_turn("done")])
    result = Agent(llm, registry).run("read")

    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10


def test_answer_without_tools_finishes_in_one_step(registry):
    result = Agent(ScriptedLLM([answer_turn("hello")]), registry).run("hi")
    assert (result.status, result.steps) == ("completed", 1)


def test_multiple_tool_calls_in_one_turn_run_in_order(registry):
    from agent_runtime import AssistantTurn, ToolCall

    turn = AssistantTurn(
        tool_calls=[
            ToolCall(id="a", name="read_file", arguments={"path": "app.py"}),
            ToolCall(id="b", name="read_file", arguments={"path": "notes.txt"}),
        ]
    )
    result = Agent(ScriptedLLM([turn, answer_turn("both read")]), registry).run("read both")

    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b"]


def test_exhausted_script_raises(registry):
    with pytest.raises(ScriptExhaustedError):
        Agent(ScriptedLLM([call_turn("c1", "read_file", path="notes.txt")]), registry).run("x")


def test_max_steps_must_be_positive(registry):
    with pytest.raises(ValueError):
        Agent(ScriptedLLM([]), registry, max_steps=0)
