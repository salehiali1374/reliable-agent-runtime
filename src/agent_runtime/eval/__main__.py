"""Command line: ``python -m agent_runtime.eval <benchmark.json> --mode simulated|live``."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from agent_runtime import SQLiteTracer
from agent_runtime.eval.metrics import compute_metrics
from agent_runtime.eval.report import render_markdown
from agent_runtime.eval.runner import run_benchmark, scripted_llm
from agent_runtime.eval.tasks import load_benchmark
from agent_runtime.openai_compat import OpenAICompatClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_runtime.eval")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--mode", choices=["simulated", "live"], default="simulated")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--trace-db", type=Path, help="also write every event to this SQLite file")
    args = parser.parse_args(argv)

    benchmark = load_benchmark(args.benchmark)
    model = None
    if args.mode == "simulated":
        llm_for = scripted_llm
    else:
        base_url = os.environ.get("AGENT_LLM_BASE_URL")
        model = os.environ.get("AGENT_LLM_MODEL")
        if not base_url or not model:
            print("live mode needs AGENT_LLM_BASE_URL and AGENT_LLM_MODEL", file=sys.stderr)
            return 2
        client = OpenAICompatClient(base_url, model, api_key=os.environ.get("AGENT_LLM_API_KEY"))

        def llm_for(task):
            return client

    tracer = SQLiteTracer(args.trace_db) if args.trace_db else None
    outcomes = run_benchmark(benchmark, llm_for, tracer=tracer, measure_time=args.mode == "live")
    if tracer:
        tracer.close()
    metrics = compute_metrics(outcomes)

    prefix = "sim" if args.mode == "simulated" else "live"
    suffix = f"_{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}" if model else ""
    stem = f"{prefix}_{benchmark.name}{suffix}"
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": benchmark.name,
        "mode": args.mode,
        "model": model,
        "metrics": metrics,
        "outcomes": [o.to_dict() for o in outcomes],
    }
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown = render_markdown(benchmark.name, args.mode, model, metrics, outcomes)
    (args.out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
