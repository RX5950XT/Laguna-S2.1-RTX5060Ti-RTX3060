"""Deterministic long-context needle prompt generation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NeedlePrompt:
    """Generated prompt plus enough metadata to reproduce and score it."""

    prompt: str
    answer: str
    needle: str
    target_tokens: int
    estimated_tokens: int
    chars_per_token: int
    target_characters: int
    actual_characters: int
    requested_depth: float
    actual_depth: float
    seed: int
    estimate_method: str = "character_budget"

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("prompt")
        return data


def generate_needle_prompt(
    target_tokens: int,
    *,
    depth: float = 0.5,
    seed: int = 20260813,
    chars_per_token: int = 4,
) -> NeedlePrompt:
    """Build a deterministic prompt with one answer-bearing record."""
    if target_tokens <= 0:
        raise ValueError("target_tokens 必須大於 0")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token 必須大於 0")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth 必須介於 0 與 1")
    target_characters = target_tokens * chars_per_token
    answer = f"TIDAL-{hashlib.sha256(str(seed).encode()).hexdigest()[:16].upper()}"
    needle = (
        "\n[CRITICAL RECORD]\n"
        f"Exact answer: {answer}\n"
        "[/CRITICAL RECORD]\n"
    )
    prefix = (
        "Read the entire telemetry archive. Ignore routine records. "
        "Return only the exact answer from the single CRITICAL RECORD.\n"
        "[ARCHIVE START]\n"
    )
    suffix = "\n[ARCHIVE END]\nQuestion: What is the exact answer?\nAnswer:"
    available = target_characters - len(prefix) - len(needle) - len(suffix)
    if available < 32:
        minimum = len(prefix) + len(needle) + len(suffix) + 32
        raise ValueError(f"target 太小；至少需要 {minimum} characters")
    before_length = round(available * depth)
    after_length = available - before_length
    before = _haystack(before_length, seed)
    after = _haystack(after_length, seed + 1)
    prompt = prefix + before + needle + after + suffix
    needle_position = prompt.index(needle)
    return NeedlePrompt(
        prompt=prompt,
        answer=answer,
        needle=needle.strip(),
        target_tokens=target_tokens,
        estimated_tokens=round(len(prompt) / chars_per_token),
        chars_per_token=chars_per_token,
        target_characters=target_characters,
        actual_characters=len(prompt),
        requested_depth=depth,
        actual_depth=needle_position / len(prompt),
        seed=seed,
    )


def _haystack(length: int, seed: int) -> str:
    vocabulary = [
        "amber", "bridge", "cobalt", "delta", "ember", "forest",
        "granite", "harbor", "island", "juniper", "kernel", "lantern",
        "meadow", "north", "orbit", "prairie", "quartz", "river",
        "signal", "timber", "uplink", "valley", "willow", "zenith",
    ]
    randomizer = random.Random(seed)
    parts: list[str] = []
    size = 0
    record = 0
    while size < length:
        words = randomizer.sample(vocabulary, 8)
        line = f"record-{record:06d}: {' '.join(words)}; status routine.\n"
        parts.append(line)
        size += len(line)
        record += 1
    return "".join(parts)[:length]
