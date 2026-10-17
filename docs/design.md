# Design

## Goal

An agent receives a multi-step task, chooses tools, inspects their output and decides the next
step. The runtime keeps that loop bounded (steps, cost, time), recoverable (retries,
checkpoints) and observable (a trace for every step), and ships with a fixed benchmark so
changes can be compared by numbers.

Version 1 runs only inside a restricted sandbox workspace. The agent has no access to the host
file system outside it, and destructive changes are out of scope.

## Architecture

```
Task (goal, constraints, success criteria)
        |
   Agent Runtime  --- decision loop, state, budget, stop conditions
     |      |
LLM Adapter  Tool Registry --- schemas, input validation, execution
     |
State & Checkpoints (SQLite)    Tracing & Evaluation
```

| Component | Responsibility |
| --- | --- |
| Agent Runtime | Controlled loop: ask the model, validate the tool call, execute, record, decide to continue or stop |
| LLM Adapter | Provider-independent interface; a scripted fake for tests, an OpenAI-compatible HTTP client for live runs |
| Tool Registry | Tool registration, JSON schema, Pydantic input validation, sandboxed execution |
| State Management | Run state and checkpoints in SQLite, resumable after a crash |
| Error Handling | Per-call timeout, bounded retry for transient errors, step and cost limits |
| Observability | Event log per step: chosen tool, rationale, output, latency, token usage, errors |
| Evaluation | Fixed task set with programmatic success checks |
| MCP Client | Connects to MCP servers over stdio and exposes their tools through the same `Tool` interface as local ones |
| Skill Loader | Discovers `SKILL.md` bundles, advertises name and description, loads the full instructions on demand |

Dependencies are kept minimal: Python 3.11+, Pydantic, pytest, SQLite from the standard
library. No agent framework. The MCP client is a small JSON-RPC implementation over stdio; the
official `mcp` SDK is used only for the demo server and is an optional extra.

## Tools, MCP and skills

Three extension mechanisms, each with a distinct job:

- **Tool** - a function the runtime can call: name, description, JSON schema for arguments,
  an executor. Local Python tools and MCP tools implement the same interface, so the loop does
  not know where a tool lives.
- **MCP server** - a separate process that publishes tools (and optionally resources and
  prompts) over the Model Context Protocol (JSON-RPC 2.0; stdio transport first). It lets any
  MCP-capable client use an application's capabilities without custom glue.
- **Skill** - a directory with a `SKILL.md` (YAML front matter: `name`, `description`;
  Markdown body with instructions) and optional files. Only the name and description sit in
  the prompt; the body is loaded through a `load_skill` tool when the model decides it applies.
  Skills carry procedure and domain knowledge, tools carry actions.

## Milestones

1. **Core runtime and tools** - agent loop, `Tool` abstraction, registry, schema generation
   from Pydantic models, input validation, request/response handling. Exit: a multi-step task
   completes against fake tools, covered by automated tests.
2. **Reliability and recovery** - timeouts, transient-error retry, step and cost caps, state
   save and resume. Exit: tests show a failing or runaway run terminates in a controlled way.
3. **Observability and evaluation** - per-step traces, benchmark with passing and failing
   tasks. Exit: a repeatable report of success rate, steps and cost.
4. **MCP** - stdio MCP client (initialize, `tools/list`, `tools/call`, error mapping) that
   registers remote tools in the registry; an MCP server exposing the sandbox workspace tools
   (read file, search code, run tests). Exit: the agent completes a task using only MCP tools,
   and the same server works from a third-party MCP client.
5. **Skills** - skill discovery and validation, index injected into the system prompt,
   `load_skill` tool, optional bundled reference files. Exit: benchmark comparison of the same
   tasks with and without a triage skill.
6. **Codebase triage demo** - a sandbox project with seeded bugs and failing tests; the agent
   gathers evidence via MCP tools, follows a skill, and writes a diagnosis report without being
   allowed to modify code. Exit: runnable demo, sample benchmark and a technical write-up.

## Reliability semantics

- **Errors are results.** A tool failure (unknown tool, invalid arguments, exception, timeout)
  is returned to the model as an error message; it never aborts the run.
- **Retry.** Only `TransientToolError`, `ToolTimeoutError` and `TransientLLMError` are retried:
  exponential backoff, capped delay, bounded attempts. Everything else fails on the first try.
- **Timeouts.** Tool calls run in a worker thread and the runtime stops waiting after the
  limit. A Python thread cannot be killed, so tools that need hard cancellation must use a
  subprocess.
- **Budget.** Steps, tool calls, tokens, cost, wall-clock time and identical repeated calls
  are checked before every model call. A run that hits a limit ends with
  `status="budget_exceeded"` and a `stop_reason`; the history stays valid (every tool call has
  a result).
- **Checkpoints.** The full `RunState` is written to SQLite after every completed step.
  `Agent.resume(run_id)` continues from the last checkpoint and counts work done before the
  interruption against the budget. A step interrupted mid-way is executed again, so tools
  must be idempotent; the sandbox tools are read-only.
- **Failure model.** A model that stays unavailable ends the run as `failed`. An unexpected
  exception from an adapter is a bug and propagates; the last checkpoint remains resumable.

## Observability and evaluation

- **Trace.** Four event types per run: `run_started`, `llm_call` (step, latency, attempts,
  tokens, the model's text, requested calls), `tool_call` (name, arguments, output, latency,
  attempts, `error_kind`) and `run_finished`. Events are append-only with a per-run sequence
  number, so a resumed run continues its own trace.
- **Benchmark.** A task has a prompt, declarative success criteria (status, stop reason,
  required and forbidden answer text, required tool output), expected tool calls, optional
  fault injection and a budget override. Tasks run against a fixture workspace through
  read-only tools that refuse paths outside it.
- **Categories.** `task` needs a correct answer, `recovery` needs one after a failure that
  actually occurred (a run that never failed does not count as recovered), `boundary` needs a
  limit to stop the run.
- **Simulated vs live.** Simulated runs use scripted models and a frozen clock, so the report
  is byte-for-byte reproducible and reports latency as n/a. Live runs use a real model and
  are written to separate files that name the model.
- **Adapter.** `OpenAICompatClient` speaks the chat-completions API with tool calls over the
  standard library. Timeouts, connection errors and HTTP 408/425/429/5xx are transient;
  other failures are permanent; unparsable tool-call arguments are passed on as an invalid
  call the model can correct.

## Metrics

- Task success rate against criteria fixed before the run.
- Tool-call accuracy: right tool with valid arguments.
- Recovery rate after recoverable failures.
- Cost and latency per task.
- Reliability: step, budget and access limits hold in boundary tests.

Values are published only after the benchmark runs. Results from simulated (scripted) models
and from real model runs are reported separately.
