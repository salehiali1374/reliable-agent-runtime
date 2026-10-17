"""Run state and its SQLite checkpoint store."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent_runtime.messages import Message, Usage

RunStatus = Literal["running", "completed", "budget_exceeded", "failed"]


class RunState(BaseModel):
    """Everything needed to continue a run, and the result of a finished one."""

    run_id: str
    task: str
    status: RunStatus = "running"
    stop_reason: str | None = None
    answer: str | None = None
    error: str | None = None
    messages: list[Message] = Field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)
    elapsed_seconds: float = 0.0


class CheckpointStore:
    """Persists the latest ``RunState`` of each run in SQLite."""

    def __init__(self, path: str | Path = ":memory:"):
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "run_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
            "state TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        self._db.commit()

    def save(self, state: RunState) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO runs (run_id, status, state, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "status = excluded.status, state = excluded.state, "
                "updated_at = excluded.updated_at",
                (state.run_id, state.status, state.model_dump_json(), time.time()),
            )

    def load(self, run_id: str) -> RunState | None:
        row = self._db.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunState.model_validate_json(row[0]) if row else None

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> CheckpointStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
