# Benchmark `triage` (live)

Live run: a real model produced every decision. Results depend on the model.

Model: gemma3:27b-it-q8_0. Tasks: 6 (4 task, 2 recovery, 0 boundary).

| Metric | Value |
| --- | --- |
| Task success rate | 33% |
| Recovery rate | 50% |
| Boundary pass rate | n/a |
| Tool-call validity | 100% |
| Tool-selection accuracy | 40% |
| Steps per task | mean 3.5, median 3.0 |
| Tool calls per task | mean 2.5, median 2.0 |
| Tokens per task | mean 2127, median 1846 |
| Latency per task (s) | mean 8.90, median 7.79 |
| Cost (USD, total) | n/a (no pricing) |

## Tasks

| Task | Category | Result | Status | Steps | Calls | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| T01-add-bug | task | pass | completed | 3 | 2 |  |
| T02-mean-cascade | task | FAIL | completed | 5 | 4 | answer lacks 'calc/ops.py'; answer lacks 'add'; answer lacks 'a - b' |
| T03-empty-mean | task | FAIL | completed | 3 | 2 | answer lacks 'calc/stats.py'; answer lacks 'len(values)' |
| T04-full-report | task | FAIL | completed | 3 | 2 | answer lacks 'a - b'; answer lacks 'len(values)' |
| R03-transient-fault | recovery | pass | completed | 2 | 1 |  |
| R04-fallback-after-outage | recovery | FAIL | completed | 5 | 4 | answer lacks 'calc/ops.py'; answer lacks 'a - b' |
