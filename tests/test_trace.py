import pytest

from agent_runtime import (
    Agent,
    AssistantTurn,
    Budget,
    CheckpointStore,
    MemoryTracer,
    ScriptedLLM,
    SQLiteTracer,
    ToolCall,
    Usage,
)

from .conftest import FakeClock, answer_turn, call_turn


def types(events):
    return [e.type for e in events]


def run(registry, turns, **kwargs):
    tracer = MemoryTracer()
    result = Agent(ScriptedLLM(turns), registry, tracer=tracer, **kwargs).run("task", run_id="r1")
    return result, tracer.for_run("r1")


def test_run_emits_ordered_events(registry):
    _, events = run(registry, [call_turn("c1", "read_file", path="notes.txt"), answer_turn("done")])
    assert types(events) == ["run_started", "llm_call", "tool_call", "llm_call", "run_finished"]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]


def test_llm_call_records_rationale_choice_and_usage(registry):
    turn = AssistantTurn(
        content="the log mentions app.py",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "app.py"})],
        usage=Usage(prompt_tokens=11, completion_tokens=4),
    )
    _, events = run(registry, [turn, answer_turn("x")])
    llm = events[1].data
    assert llm["content"] == "the log mentions app.py"
    assert llm["tool_calls"][0]["name"] == "read_file"
    assert llm["tool_calls"][0]["arguments"] == {"path": "app.py"}
    assert (llm["prompt_tokens"], llm["completion_tokens"]) == (11, 4)
    assert llm["step"] == 1


def test_tool_call_records_output_and_error_kind(registry):
    turns = [
        call_turn("c1", "read_file", path="notes.txt"),
        call_turn("c2", "read_file"),
        call_turn("c3", "nope"),
        call_turn("c4", "read_file", path="missing"),
        answer_turn("x"),
    ]
    _, events = run(registry, turns, budget=Budget(max_identical_calls=None))
    tools = [e.data for e in events if e.type == "tool_call"]
    assert [t["error_kind"] for t in tools] == [
        None,
        "invalid_arguments",
        "unknown_tool",
        "tool_error",
    ]
    assert tools[0]["output"] == "nothing here"
    assert tools[0]["is_error"] is False
    assert tools[1]["is_error"] is True


def test_latency_comes_from_the_injected_clock(registry):
    clock = FakeClock()

    class TimedLLM(ScriptedLLM):
        def complete(self, messages, tools):
            clock.advance(2.0)
            return super().complete(messages, tools)

    tracer = MemoryTracer()
    llm = TimedLLM([call_turn("c1", "read_file", path="notes.txt"), answer_turn("x")])
    Agent(llm, registry, tracer=tracer, clock=clock).run("task", run_id="r1")
    llm_events = [e for e in tracer.events if e.type == "llm_call"]
    assert [e.data["latency_s"] for e in llm_events] == [2.0, 2.0]


def test_run_finished_summarises_the_run(registry):
    result, events = run(
        registry,
        [call_turn(f"c{i}", "search_code", pattern=str(i)) for i in range(9)],
        budget=Budget(max_steps=2),
    )
    done = events[-1]
    assert done.type == "run_finished"
    assert done.data["status"] == "budget_exceeded"
    assert done.data["stop_reason"] == "max_steps"
    assert done.data["steps"] == result.steps == 2


def test_skipped_call_is_traced_as_budget_error(registry):
    turn = AssistantTurn(
        tool_calls=[
            ToolCall(id="a", name="read_file", arguments={"path": "app.py"}),
            ToolCall(id="b", name="read_file", arguments={"path": "notes.txt"}),
        ]
    )
    _, events = run(registry, [turn], budget=Budget(max_tool_calls=1))
    kinds = [e.data["error_kind"] for e in events if e.type == "tool_call"]
    assert kinds == [None, "budget"]


def test_long_output_is_clipped(registry):
    from .conftest import FILES

    FILES["big.txt"] = "x" * 5000
    try:
        _, events = run(registry, [call_turn("c1", "read_file", path="big.txt"), answer_turn("x")])
    finally:
        del FILES["big.txt"]
    output = next(e for e in events if e.type == "tool_call").data["output"]
    assert len(output) < 2100
    assert "chars cut" in output


def test_sqlite_tracer_round_trip(tmp_path, registry):
    path = tmp_path / "trace.sqlite3"
    with SQLiteTracer(path) as tracer:
        Agent(ScriptedLLM([answer_turn("hi")]), registry, tracer=tracer).run("t", run_id="r1")
    with SQLiteTracer(path) as reopened:
        events = reopened.for_run("r1")
        assert types(events) == ["run_started", "llm_call", "run_finished"]
        assert events[1].data["content"] == "hi"
        assert reopened.for_run("other") == []


def test_trace_continues_across_resume(registry):
    turns = [
        call_turn("c1", "read_file", path="app.py"),
        call_turn("c2", "read_file", path="notes.txt"),
        answer_turn("done"),
    ]

    class CrashOnThird(ScriptedLLM):
        def complete(self, messages, tools):
            if len(self.requests) == 2:
                raise RuntimeError("killed")
            return super().complete(messages, tools)

    with CheckpointStore() as store, SQLiteTracer() as tracer:
        with pytest.raises(RuntimeError):
            Agent(CrashOnThird(turns), registry, store=store, tracer=tracer).run("t", run_id="r1")
        Agent(ScriptedLLM(turns[2:]), registry, store=store, tracer=tracer).resume("r1")
        events = tracer.for_run("r1")

    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    started = [e.data["resumed"] for e in events if e.type == "run_started"]
    assert started == [False, True]
    assert types(events)[-1] == "run_finished"
