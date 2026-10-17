# Benchmark `triage` (simulated)

Simulated run: the model is a fixed script, so these numbers verify the runtime's handling of tools, errors and limits. They say nothing about the quality of any real model.

Model: scripted. Tasks: 16 (6 task, 5 recovery, 5 boundary).

| Metric | Value |
| --- | --- |
| Task success rate | 73% |
| Recovery rate | 80% |
| Boundary pass rate | 100% |
| Tool-call validity | 94% |
| Tool-selection accuracy | 80% |
| Steps per task | mean 2.8, median 3.0 |
| Tool calls per task | mean 2.1, median 2.0 |
| Tokens per task | mean 412, median 450 |
| Latency per task (s) | n/a |
| Cost (USD, total) | n/a (no pricing) |

## Tasks

| Task | Category | Result | Status | Steps | Calls | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| T01-add-bug | task | pass | completed | 3 | 2 |  |
| T02-mean-cascade | task | pass | completed | 4 | 3 |  |
| T03-empty-mean | task | pass | completed | 3 | 2 |  |
| T04-full-report | task | pass | completed | 4 | 4 |  |
| T05-wrong-diagnosis | task | FAIL | completed | 3 | 2 | answer lacks 'calc/ops.py'; answer lacks 'a - b' |
| T06-ungrounded-answer | task | FAIL | completed | 1 | 0 | answer lacks 'calc/ops.py'; answer lacks 'a - b' |
| R01-invalid-arguments | recovery | pass | completed | 3 | 2 |  |
| R02-unknown-tool | recovery | pass | completed | 3 | 2 |  |
| R03-transient-fault | recovery | pass | completed | 2 | 1 |  |
| R04-fallback-after-outage | recovery | pass | completed | 3 | 2 |  |
| R05-gives-up | recovery | FAIL | completed | 2 | 1 | answer lacks 'calc/ops.py'; answer lacks 'a - b' |
| B01-step-limit | boundary | pass | budget_exceeded (max_steps) | 4 | 4 |  |
| B02-identical-call-loop | boundary | pass | budget_exceeded (repeated_call) | 3 | 3 |  |
| B03-tool-call-cap | boundary | pass | budget_exceeded (max_tool_calls) | 1 | 2 |  |
| B04-token-cap | boundary | pass | budget_exceeded (max_tokens) | 2 | 2 |  |
| B05-path-escape | boundary | pass | completed | 3 | 2 |  |
