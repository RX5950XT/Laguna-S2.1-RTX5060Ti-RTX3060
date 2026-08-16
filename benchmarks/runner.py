"""Benchmark orchestration and report schema."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from benchmarks.api import OpenAIClient
from benchmarks.prompts import generate_needle_prompt


T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str = "http://127.0.0.1:8000"
    model: str | None = None
    timeout: float = 600.0
    stream_max_tokens: int = 512
    context_window_tokens: int = 262_144
    prompt_tokens: int = 260_000
    chars_per_token: int = 4
    needle_depth: float = 0.5
    seed: int = 20260813
    long_max_tokens: int = 128
    run_long_context: bool = True

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ValueError("model 不得是空字串")
        if self.timeout <= 0:
            raise ValueError("timeout 必須大於 0")
        if self.stream_max_tokens <= 0 or self.long_max_tokens <= 0:
            raise ValueError("max_tokens 必須大於 0")
        if self.context_window_tokens < 262_144:
            raise ValueError("context_window_tokens 不得小於 262144")
        if self.prompt_tokens <= 0 or self.chars_per_token <= 0:
            raise ValueError("prompt_tokens/chars_per_token 必須大於 0")
        if self.prompt_tokens + self.long_max_tokens >= self.context_window_tokens:
            raise ValueError("prompt_tokens 必須為 template 與 generation 預留 context")
        if not 0.0 <= self.needle_depth <= 1.0:
            raise ValueError("needle_depth 必須介於 0 與 1")


def run_benchmark(
    config: BenchmarkConfig, api_key: str | None = None
) -> dict[str, Any]:
    """Run model discovery, chat, streaming, and optional needle retrieval."""
    client = OpenAIClient(config.base_url, api_key=api_key, timeout=config.timeout)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    models, models_duration = _timed(client.list_models)
    model = config.model or _first_model_id(models)
    chat, chat_duration = _timed(
        lambda: client.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=16,
        )
    )
    stream = client.stream_chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Produce a deterministic short response for throughput "
                    "measurement."
                ),
            }
        ],
        max_tokens=config.stream_max_tokens,
    )
    long_context = _run_long_context(client, model, config)
    completed_utc = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "success": True,
        "started_at_utc": started_utc,
        "completed_at_utc": completed_utc,
        "duration_seconds": time.perf_counter() - started,
        "configuration": _configuration(config, model),
        "models": {"duration_seconds": models_duration, "response": models},
        "chat": {"duration_seconds": chat_duration, "response": chat},
        "streaming": stream.to_dict(),
        "long_context": long_context,
    }


def _run_long_context(
    client: OpenAIClient, model: str, config: BenchmarkConfig
) -> dict[str, Any]:
    if not config.run_long_context:
        return {"skipped": True}
    prompt = generate_needle_prompt(
        config.prompt_tokens,
        depth=config.needle_depth,
        seed=config.seed,
        chars_per_token=config.chars_per_token,
    )
    result = client.stream_chat(
        model=model,
        messages=[{"role": "user", "content": prompt.prompt}],
        max_tokens=config.long_max_tokens,
    )
    needle_found = prompt.answer.casefold() in result.content.casefold()
    return {
        "skipped": False,
        "needle_found": needle_found,
        "expected_answer": prompt.answer,
        "prompt": prompt.metadata(),
        "response": result.to_dict(),
    }


def _first_model_id(models: dict[str, Any]) -> str:
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("/v1/models 未提供可用 model")
    first = data[0]
    if not isinstance(first, dict) or not isinstance(first.get("id"), str):
        raise ValueError("/v1/models 第一筆缺少字串 id")
    if not first["id"].strip():
        raise ValueError("/v1/models model id 不得為空")
    return first["id"]


def _timed(operation: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _configuration(config: BenchmarkConfig, model: str) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "model": model,
        "timeout_seconds": config.timeout,
        "stream_max_tokens": config.stream_max_tokens,
        "context_window_tokens": config.context_window_tokens,
        "prompt_tokens": config.prompt_tokens,
        "chars_per_token": config.chars_per_token,
        "needle_depth": config.needle_depth,
        "seed": config.seed,
        "long_max_tokens": config.long_max_tokens,
        "run_long_context": config.run_long_context,
    }
