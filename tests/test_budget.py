import pytest

from agent_runtime import Agent, AssistantTurn, Budget, Pricing, RetryPolicy, ScriptedLLM, Usage
from agent_runtime.llm import LLMError, TransientLLMError

from .conftest import FakeClock, NoSleep, answer_turn, call_turn


def endless(n=30):
    return [call_turn(f"c{i}", "search_code", pattern=f"p{i}") for i in range(n)]


def run(registry, turns, **agent_kwargs):
    return Agent(ScriptedLLM(turns), registry, **agent_kwargs).run("task")


def test_max_tool_calls_stops_run_and_keeps_history_valid(registry):
    two_calls = AssistantTurn(
        tool_calls=[
            *call_turn("a", "search_code", pattern="x").tool_calls,
            *call_turn("b", "search_code", pattern="y").tool_calls,
            *call_turn("c", "search_code", pattern="z").tool_calls,
        ]
    )
    result = run(registry, [two_calls, answer_turn("never")], budget=Budget(max_tool_calls=2))

    assert (result.status, result.stop_reason) == ("budget_exceeded", "max_tool_calls")
    assert result.tool_calls == 2
    answered = {m.tool_call_id for m in result.messages if m.role == "tool"}
    requested = {c.id for m in result.messages for c in m.tool_calls}
    assert answered == requested  # the skipped call still has a (error) result
    assert result.messages[-1].is_error


def test_max_tokens_stops_run(registry):
    result = run(registry, endless(), budget=Budget(max_tokens=40, max_identical_calls=None))
    assert result.stop_reason == "max_tokens"
    assert result.steps == 3  # 15 tokens per step: 45 >= 40 after the third


def test_max_cost_stops_run(registry):
    pricing = Pricing(prompt_per_million=1000.0, completion_per_million=2000.0)
    # one step costs 10*1000/1e6 + 5*2000/1e6 = 0.02
    result = run(
        registry,
        endless(),
        budget=Budget(max_cost_usd=0.05, max_identical_calls=None),
        pricing=pricing,
    )
    assert result.stop_reason == "max_cost"
    assert result.steps == 3


def test_cost_limit_requires_pricing(registry):
    with pytest.raises(ValueError, match="pricing"):
        Agent(ScriptedLLM([]), registry, budget=Budget(max_cost_usd=1.0))


def test_wall_clock_limit_uses_injected_clock(registry):
    clock = FakeClock()

    class SlowLLM(ScriptedLLM):
        def complete(self, messages, tools):
            clock.advance(10)
            return super().complete(messages, tools)

    llm = SlowLLM(endless())
    result = Agent(
        llm, registry, budget=Budget(max_seconds=25, max_identical_calls=None), clock=clock
    ).run("task")
    assert result.stop_reason == "max_seconds"
    assert result.steps == 3
    assert result.elapsed_seconds == 30


def test_repeated_identical_calls_are_stopped(registry):
    same = [call_turn(f"c{i}", "read_file", path="notes.txt") for i in range(10)]
    result = run(registry, same, budget=Budget(max_identical_calls=3))
    assert result.stop_reason == "repeated_call"
    assert result.steps == 3


def test_different_arguments_are_not_counted_as_repeats(registry):
    turns = [
        call_turn("c1", "read_file", path="notes.txt"),
        call_turn("c2", "read_file", path="app.py"),
        call_turn("c3", "read_file", path="notes.txt"),
        answer_turn("done"),
    ]
    result = run(registry, turns, budget=Budget(max_identical_calls=3))
    assert result.status == "completed"


def test_transient_llm_error_is_retried(registry):
    class FlakyLLM(ScriptedLLM):
        failures = 2

        def complete(self, messages, tools):
            if self.failures:
                self.failures -= 1
                raise TransientLLMError("429 rate limited")
            return super().complete(messages, tools)

    sleep = NoSleep()
    agent = Agent(FlakyLLM([answer_turn("ok")]), registry, retry=RetryPolicy(sleep=sleep))
    result = agent.run("task")
    assert result.status == "completed"
    assert len(sleep.delays) == 2


def test_llm_that_never_recovers_fails_the_run_cleanly(registry):
    class DownLLM:
        def complete(self, messages, tools):
            raise TransientLLMError("503")

    agent = Agent(DownLLM(), registry, retry=RetryPolicy(max_attempts=2, sleep=NoSleep()))
    result = agent.run("task")
    assert (result.status, result.stop_reason) == ("failed", "llm_error")
    assert "503" in result.error


def test_permanent_llm_error_fails_without_retry(registry):
    calls = {"n": 0}

    class BadKeyLLM:
        def complete(self, messages, tools):
            calls["n"] += 1
            raise LLMError("401 invalid api key")

    result = Agent(BadKeyLLM(), registry, retry=RetryPolicy(sleep=NoSleep())).run("task")
    assert result.status == "failed"
    assert calls["n"] == 1


def test_unexpected_exception_from_model_adapter_propagates(registry):
    class BuggyLLM:
        def complete(self, messages, tools):
            raise KeyError("bug in adapter")

    with pytest.raises(KeyError):
        Agent(BuggyLLM(), registry, retry=RetryPolicy(sleep=NoSleep())).run("task")


def test_usage_totals_are_recorded_on_budget_stop(registry):
    result = run(registry, endless(), budget=Budget(max_steps=2, max_identical_calls=None))
    assert result.usage == Usage(prompt_tokens=20, completion_tokens=10)
