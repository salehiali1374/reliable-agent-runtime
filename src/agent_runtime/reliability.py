"""Bounded retry with exponential backoff."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


class RetryFailed(Exception):
    """The call failed for good: a permanent error, or transient errors past the attempt limit."""

    def __init__(self, attempts: int, last: Exception):
        super().__init__(f"gave up after {attempts} attempt(s): {last}")
        self.attempts = attempts
        self.last = last


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    sleep: Callable[[float], None] = field(default=time.sleep, compare=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def delay(self, attempt: int) -> float:
        """Delay before the retry that follows failed attempt number ``attempt`` (1-based)."""
        return min(self.max_delay, self.base_delay * 2 ** (attempt - 1))


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy,
    is_transient: Callable[[Exception], bool],
) -> tuple[T, int]:
    """Call ``fn`` until it succeeds, returning ``(result, attempts)``.

    Only exceptions for which ``is_transient`` is true are retried. Every failure is raised as
    ``RetryFailed`` carrying the attempt count and the last underlying exception.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001 - classification is delegated to the caller
            if not is_transient(exc) or attempt >= policy.max_attempts:
                raise RetryFailed(attempt, exc) from exc
            policy.sleep(policy.delay(attempt))
