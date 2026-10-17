import pytest

from agent_runtime import Agent, Budget, CheckpointStore, RunState, ScriptedLLM

from .conftest import answer_turn, call_turn

TURNS = [
    call_turn("c1", "search_code", pattern="add"),
    call_turn("c2", "read_file", path="app.py"),
    answer_turn("add() subtracts."),
]


class CrashingLLM(ScriptedLLM):
    """Behaves like the script, then dies at the given call number (like a killed process)."""

    def __init__(self, turns, crash_on_call: int):
        super().__init__(turns)
        self.crash_on_call = crash_on_call

    def complete(self, messages, tools):
        if len(self.requests) + 1 == self.crash_on_call:
            raise RuntimeError("process killed")
        return super().complete(messages, tools)


def test_store_round_trip(tmp_path):
    state = RunState(run_id="r1", task="t", steps=2, status="completed", answer="x")
    path = tmp_path / "runs.sqlite3"
    with CheckpointStore(path) as store:
        store.save(state)
    with CheckpointStore(path) as reopened:  # a new connection sees what was persisted
        assert reopened.load("r1") == state
        assert reopened.load("missing") is None


def test_save_overwrites_the_previous_checkpoint():
    with CheckpointStore() as store:
        store.save(RunState(run_id="r1", task="t", steps=1))
        store.save(RunState(run_id="r1", task="t", steps=2))
        assert store.load("r1").steps == 2


def test_run_writes_a_checkpoint_after_every_step(registry):
    with CheckpointStore() as store:
        result = Agent(ScriptedLLM(TURNS), registry, store=store).run("t", run_id="r1")
        saved = store.load("r1")
    assert saved == result
    assert saved.status == "completed"


def test_resume_continues_after_a_crash_without_repeating_steps(registry):
    with CheckpointStore() as store:
        crashing = Agent(CrashingLLM(TURNS, crash_on_call=3), registry, store=store)
        with pytest.raises(RuntimeError):
            crashing.run("t", run_id="r1")

        interrupted = store.load("r1")
        assert interrupted.status == "running"
        assert interrupted.steps == 2

        remaining = ScriptedLLM(TURNS[2:])  # the model is only asked for the missing step
        resumed = Agent(remaining, registry, store=store).resume("r1")

    assert resumed.status == "completed"
    assert resumed.steps == 3
    assert resumed.answer == "add() subtracts."
    assert len(remaining.requests) == 1
    assert [m.role for m in resumed.messages].count("tool") == 2


def test_resumed_run_matches_an_uninterrupted_run(registry):
    clean = Agent(ScriptedLLM(TURNS), registry).run("t", run_id="r1")

    with CheckpointStore() as store:
        with pytest.raises(RuntimeError):
            Agent(CrashingLLM(TURNS, crash_on_call=2), registry, store=store).run("t", run_id="r1")
        resumed = Agent(ScriptedLLM(TURNS[1:]), registry, store=store).resume("r1")

    assert resumed.messages == clean.messages
    assert resumed.usage == clean.usage
    assert resumed.answer == clean.answer


def test_budget_counts_work_done_before_the_crash(registry):
    turns = [call_turn(f"c{i}", "search_code", pattern=f"p{i}") for i in range(10)]
    budget = Budget(max_steps=4, max_identical_calls=None)
    with CheckpointStore() as store:
        with pytest.raises(RuntimeError):
            Agent(CrashingLLM(turns, 3), registry, budget=budget, store=store).run("t", run_id="r")
        remaining = ScriptedLLM(turns[2:])
        resumed = Agent(remaining, registry, budget=budget, store=store).resume("r")

    assert resumed.stop_reason == "max_steps"
    assert resumed.steps == 4
    assert len(remaining.requests) == 2  # 2 steps before the crash + 2 after = the limit of 4


def test_resuming_a_finished_run_returns_it_unchanged(registry):
    with CheckpointStore() as store:
        done = Agent(ScriptedLLM(TURNS), registry, store=store).run("t", run_id="r1")
        never_called = ScriptedLLM([])
        again = Agent(never_called, registry, store=store).resume("r1")
    assert again == done
    assert never_called.requests == []


def test_resume_unknown_run_raises(registry):
    with CheckpointStore() as store:
        with pytest.raises(KeyError):
            Agent(ScriptedLLM([]), registry, store=store).resume("nope")


def test_resume_requires_a_store(registry):
    with pytest.raises(ValueError, match="store"):
        Agent(ScriptedLLM([]), registry).resume("r1")
