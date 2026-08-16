"""Command-line entry point for the OpenAI-compatible benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from benchmarks.output import dumps_json, require_within, write_json
from benchmarks.runner import BenchmarkConfig, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible Laguna endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--stream-max-tokens", type=int, default=512)
    parser.add_argument("--context-window-tokens", type=int, default=262_144)
    parser.add_argument("--prompt-tokens", type=int, default=260_000)
    parser.add_argument("--chars-per-token", type=int, default=4)
    parser.add_argument("--needle-depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--long-max-tokens", type=int, default=128)
    parser.add_argument("--skip-long-context", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = BenchmarkConfig(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            stream_max_tokens=args.stream_max_tokens,
            context_window_tokens=args.context_window_tokens,
            prompt_tokens=args.prompt_tokens,
            chars_per_token=args.chars_per_token,
            needle_depth=args.needle_depth,
            seed=args.seed,
            long_max_tokens=args.long_max_tokens,
            run_long_context=not args.skip_long_context,
        )
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        report = run_benchmark(config, api_key=api_key)
        exit_code = 0
    except (OSError, RuntimeError, ValueError) as error:
        report = {
            "schema_version": 1,
            "success": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 1
    if args.output:
        project_root = Path(__file__).resolve().parents[1]
        try:
            output = require_within(args.output, project_root)
            write_json(output, report)
        except (OSError, ValueError) as error:
            sys.stderr.write(f"無法寫入 benchmark JSON：{error}\n")
            return 1
    else:
        sys.stdout.write(dumps_json(report))
    return exit_code
