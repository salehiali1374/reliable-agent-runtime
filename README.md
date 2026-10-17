# reliable-agent-runtime

A small, tested runtime for LLM agents that call tools: bounded execution, explicit state,
recovery from failures, per-step tracing and a reproducible evaluation harness. The core has
no agent-framework dependency; the only runtime dependency is Pydantic.

Status: milestones 1-3 done (core runtime, reliability, observability and evaluation). See
[docs/design.md](docs/design.md) for scope, architecture and the roadmap.

## Quick look

```python
from pydantic import BaseModel
from agent_runtime import Agent, Budget, ToolRegistry, tool
from agent_runtime.openai_compat import OpenAICompatClient


class ReadArgs(BaseModel):
    path: str


@tool(ReadArgs)
def read_file(args: ReadArgs) -> str:
    """Read a file from the workspace."""
    return open(args.path).read()


agent = Agent(
    OpenAICompatClient("http://localhost:11434/v1", "my-model"),
    ToolRegistry([read_file], default_timeout=10),
    budget=Budget(max_steps=10, max_tokens=50_000, max_seconds=120),
)
result = agent.run("Why does test_add fail?")
print(result.status, result.stop_reason, result.answer)
```

Every run returns a `RunState`: status (`completed`, `budget_exceeded`, `failed`), the full
message history, step and tool-call counts, token usage and elapsed time.

## What the runtime guarantees

- Tool failures (unknown tool, invalid arguments, exception, timeout) become error results the
  model can react to; they never abort the run.
- Transient errors are retried with capped exponential backoff; permanent ones are not.
- Steps, tool calls, tokens, cost, wall-clock time and identical repeated calls are bounded.
- The run state is checkpointed to SQLite after every step and can be resumed.
- Every model call and tool call is recorded as a structured event (`MemoryTracer`,
  `SQLiteTracer`): chosen tool and arguments, the model's accompanying text, output, latency,
  attempts, error kind and token usage.

## Evaluation

```bash
python -m agent_runtime.eval benchmarks/triage.json --mode simulated
```

The benchmark runs 16 tasks on a fixture project with seeded bugs and a failing test report:
answer tasks, recovery tasks (invalid arguments, unknown tool, injected transient faults, a
tool outage) and boundary tasks (step, tool-call and token limits, loop detection, path
escape). Success criteria are declarative and fixed in the task file.

Reported metrics: task success rate, recovery rate, boundary pass rate, tool-call validity,
tool-selection accuracy, and steps, tokens, cost and latency per task.

Simulated and live results are kept apart. A simulated run replays scripted model behaviour,
including deliberately wrong answers, to verify the runtime and the scoring; it says nothing
about any real model. The current simulated report is in
[results/sim_triage.md](results/sim_triage.md).

Live run against an OpenAI-compatible server with tool calling:

```bash
AGENT_LLM_BASE_URL=http://127.0.0.1:11434/v1 AGENT_LLM_MODEL=<model> \
  python -m agent_runtime.eval benchmarks/triage.json --mode live
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -m "not live"
```
