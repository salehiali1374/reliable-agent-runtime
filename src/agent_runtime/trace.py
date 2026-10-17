"""Structured per-step event log for runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    run_id: str
    seq: int
    ts: float
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Tracer(Protocol):
    def emit(self, run_id: str, type: str, **data: Any) -> None: ...


class NullTracer:
    def emit(self, run_id: str, type: str, **data: Any) -> None:
        return None


class MemoryTracer:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, run_id: str, type: str, **data: Any) -> None:
        seq = sum(1 for e in self.events if e.run_id == run_id) + 1
        self.events.append(TraceEvent(run_id=run_id, seq=seq, ts=time.time(), type=type, data=data))

    def for_run(self, run_id: str) -> list[TraceEvent]:
        return [e for e in self.events if e.run_id == run_id]


class SQLiteTracer:
    """Appends events to a SQLite table; safe to reuse across process restarts."""

    def __init__(self, path: str | Path = ":memory:"):
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, "
            "type TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (run_id, seq))"
        )
        self._db.commit()

    def emit(self, run_id: str, type: str, **data: Any) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO events (run_id, seq, ts, type, data) VALUES (?, "
                "(SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?), ?, ?, ?)",
                (run_id, run_id, time.time(), type, json.dumps(data, default=str)),
            )

    def for_run(self, run_id: str) -> list[TraceEvent]:
        rows = self._db.execute(
            "SELECT run_id, seq, ts, type, data FROM events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [
            TraceEvent(run_id=r, seq=s, ts=t, type=ty, data=json.loads(d))
            for r, s, t, ty, d in rows
        ]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> SQLiteTracer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
