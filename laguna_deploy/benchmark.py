from __future__ import annotations

from statistics import median
from typing import Mapping, Sequence


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def decode_statistics(measurements: Sequence[float]) -> dict[str, float]:
    if not measurements:
        raise ValueError("至少需要一筆吞吐量測")
    values = [float(value) for value in measurements]
    if any(value <= 0 for value in values):
        raise ValueError("吞吐量必須為正值")
    return {
        "minimum_tps": min(values),
        "p5_tps": _percentile(values, 0.05),
        "median_tps": float(median(values)),
        "maximum_tps": max(values),
    }


def validate_acceptance(
    *,
    context_size: int,
    decode_tps: Sequence[float],
    quality_losses: Mapping[str, float],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if context_size < 262_144:
        reasons.append(f"context {context_size} 小於 262144")

    stats = decode_statistics(decode_tps)
    if stats["p5_tps"] < 20.0:
        reasons.append(f"p5 decode {stats['p5_tps']:.3f} tok/s 小於 20")

    for name, loss in quality_losses.items():
        if not 0.0 <= float(loss) < 0.05:
            reasons.append(f"{name} 品質損失 {float(loss):.3%} 未嚴格小於 5%")

    return not reasons, reasons

