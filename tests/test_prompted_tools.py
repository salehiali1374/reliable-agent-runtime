import json

from agent_runtime import Agent, AssistantTurn, Message, ScriptedLLM, ToolCall, Usage
from agent_runtime.prompted_tools import PromptedToolsClient, extract_json_object

from .conftest import read_file, search_code


def text(content: str, tokens: int = 10) -> AssistantTurn:
    return AssistantTurn(content=content, usage=Usage(prompt_tokens=tokens, completion_tokens=1))


def ask(replies, history=None, tools=(read_file.spec, search_code.spec), **kwargs):
    inner = ScriptedLLM([text(r) for r in replies])
    client = PromptedToolsClient(inner, **kwargs)
    turn = client.complete(history or [Message(role="user", content="go")], list(tools))
    return turn, inner


def test_tool_call_is_parsed():
    turn, _ = ask(
        ['{"thought": "need source", "tool": "read_file", "arguments": {"path": "a.py"}}']
    )
    assert turn.content == "need source"
    assert turn.tool_calls == [ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})]


def test_answer_is_parsed():
    turn, _ = ask(['{"answer": "add() is wrong"}'])
    assert turn.content == "add() is wrong"
    assert turn.tool_calls == []


def test_json_inside_code_fence_and_prose_is_found():
    reply = 'Sure!\n```json\n{"tool": "search_code", "arguments": {"pattern": "x"}}\n```\nDone.'
    turn, _ = ask([reply])
    assert turn.tool_calls[0].name == "search_code"


def test_several_calls_in_one_reply():
    reply = json.dumps(
        {
            "calls": [
                {"tool": "read_file", "arguments": {"path": "a"}},
                {"tool": "read_file", "arguments": {"path": "b"}},
            ]
        }
    )
    turn, _ = ask([reply])
    assert [c.arguments["path"] for c in turn.tool_calls] == ["a", "b"]
    assert len({c.id for c in turn.tool_calls}) == 2


def test_plain_prose_is_a_final_answer_without_reasking():
    turn, inner = ask(["The bug is in ops.py."])
    assert turn.content == "The bug is in ops.py."
    assert len(inner.requests) == 1


def test_invalid_action_is_reasked_and_usage_is_summed():
    turn, inner = ask(['{"foo": 1}', '{"tool": "read_file", "arguments": {"path": "a"}}'])
    assert turn.tool_calls[0].name == "read_file"
    assert len(inner.requests) == 2
    assert turn.usage.prompt_tokens == 20
    repair = inner.requests[1][-1]
    assert repair.role == "user" and "not a valid action" in repair.content


def test_arguments_must_be_an_object():
    turn, inner = ask(['{"tool": "read_file", "arguments": "a.py"}', '{"answer": "gave up"}'])
    assert turn.content == "gave up"
    assert len(inner.requests) == 2


def test_gives_up_after_format_retries_and_returns_the_raw_text():
    turn, inner = ask(['{"foo": 1}'] * 3, max_format_retries=2)
    assert turn.tool_calls == []
    assert turn.content == '{"foo": 1}'
    assert len(inner.requests) == 3


def test_system_prompt_lists_tools_and_schemas():
    history = [Message(role="system", content="Be careful."), Message(role="user", content="go")]
    _, inner = ask(['{"answer": "x"}'], history=history)
    system = inner.requests[0][0].content
    assert system.startswith("Be careful.")
    assert "read_file" in system and "search_code" in system
    assert '"path"' in system  # argument schema is shown


def test_history_is_rewritten_as_plain_chat():
    history = [
        Message(role="system", content="s"),
        Message(role="user", content="task"),
        Message(
            role="assistant",
            content="look",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a"})],
        ),
        Message(role="tool", content="file text", tool_call_id="c1"),
        Message(role="assistant", tool_calls=[ToolCall(id="c2", name="read_file", arguments={})]),
        Message(role="tool", content="path: required", tool_call_id="c2", is_error=True),
    ]
    _, inner = ask(['{"answer": "x"}'], history=history)
    chat = inner.requests[0]
    assert [m.role for m in chat] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(chat[2].content) == {
        "thought": "look",
        "tool": "read_file",
        "arguments": {"path": "a"},
    }
    assert chat[3].content == "Result of read_file (c1):\nfile text"
    assert chat[5].content.startswith("ERROR from read_file (c2):")
    assert all(not m.tool_calls and m.tool_call_id is None for m in chat)


def test_inner_model_is_called_without_native_tools():
    seen = []

    class Spy(ScriptedLLM):
        def complete(self, messages, tools):
            seen.append(list(tools))
            return super().complete(messages, tools)

    PromptedToolsClient(Spy([text('{"answer": "x"}')])).complete(
        [Message(role="user", content="go")], [read_file.spec]
    )
    assert seen == [[]]


def test_agent_runs_end_to_end_through_the_adapter(registry):
    inner = ScriptedLLM(
        [
            text('{"tool": "search_code", "arguments": {"pattern": "add"}}'),
            text('{"tool": "read_file", "arguments": {"path": "app.py"}}'),
            text('{"answer": "add() subtracts instead of adding."}'),
        ]
    )
    result = Agent(PromptedToolsClient(inner), registry).run("why does add fail?")
    assert result.status == "completed"
    assert result.answer == "add() subtracts instead of adding."
    assert result.messages[3].content == "app.py"


def test_extract_json_object_ignores_non_objects():
    assert extract_json_object("no braces here") is None
    assert extract_json_object("[1, 2]") is None
    assert extract_json_object('x {bad json} y {"a": 1}') == {"a": 1}


def test_raw_newline_inside_a_json_string_is_tolerated():
    turn, inner = ask(['{"answer": "line one\nline two"}'])
    assert turn.content == "line one\nline two"
    assert len(inner.requests) == 1
