"""OpenAI-compatible inference benchmarks for Laguna S 2.1."""

from benchmarks.api import ApiError, OpenAIClient, StreamResult
from benchmarks.prompts import NeedlePrompt, generate_needle_prompt
from benchmarks.runner import BenchmarkConfig, run_benchmark

__all__ = [
    "ApiError",
    "BenchmarkConfig",
    "NeedlePrompt",
    "OpenAIClient",
    "StreamResult",
    "generate_needle_prompt",
    "run_benchmark",
]
