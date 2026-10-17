import pytest

from agent_runtime import RetryPolicy
from agent_runtime.reliability import RetryFailed, retry_call

from .conftest import NoSleep


class Flaky(Exception):
    pass


def make_fn(failures: int):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise Flaky("temporary")
        return "ok"

    return fn, calls


def is_flaky(exc: Exception) -> bool:
    return isinstance(exc, Flaky)


def test_success_on_first_attempt_does_not_sleep():
    sleep = NoSleep()
    fn, _ = make_fn(0)
    assert retry_call(fn, RetryPolicy(sleep=sleep), is_flaky) == ("ok", 1)
    assert sleep.delays == []


def test_transient_failures_are_retried_with_exponential_backoff():
    sleep = NoSleep()
    fn, _ = make_fn(2)
    result = retry_call(fn, RetryPolicy(max_attempts=4, base_delay=1.0, sleep=sleep), is_flaky)
    assert result == ("ok", 3)
    assert sleep.delays == [1.0, 2.0]


def test_gives_up_after_max_attempts():
    sleep = NoSleep()
    fn, calls = make_fn(10)
    with pytest.raises(RetryFailed) as info:
        retry_call(fn, RetryPolicy(max_attempts=3, sleep=sleep), is_flaky)
    assert info.value.attempts == 3
    assert isinstance(info.value.last, Flaky)
    assert calls["n"] == 3
    assert len(sleep.delays) == 2


def test_permanent_error_is_not_retried():
    def fn():
        raise ValueError("bad input")

    with pytest.raises(RetryFailed) as info:
        retry_call(fn, RetryPolicy(sleep=NoSleep()), is_flaky)
    assert info.value.attempts == 1


def test_delay_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0)
    assert [policy.delay(n) for n in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
