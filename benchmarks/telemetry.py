"""Parsers for llama-bench JSON and llama.cpp MoE-cache evidence logs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


class TelemetryError(ValueError):
    """A machine-readable telemetry parse failure."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": dict(self.details),
            },
        }


@dataclass(frozen=True)
class LlamaBenchRecord:
    model_filename: str
    n_ctx: int
    n_gen: int
    avg_ts: float
    samples_ts: tuple[float, ...]
    p5_ts: float


@dataclass(frozen=True)
class DeviceCacheStats:
    hits: int
    probes: int
    hit_rate_percent: float
    fill_failures: int
    dispatch_failures: int
    collect_failures: int

    @property
    def total_failures(self) -> int:
        return self.fill_failures + self.dispatch_failures + self.collect_failures


@dataclass(frozen=True)
class MoeCacheTelemetry:
    n_ctx: int | None
    kv_mib_by_device: dict[str, float]
    granted_mib_by_device: dict[str, float]
    devices: dict[str, DeviceCacheStats]
    enabled: bool

    @property
    def kv_total_mib(self) -> float:
        return sum(self.kv_mib_by_device.values())

    @property
    def total_failures(self) -> int:
        return sum(device.total_failures for device in self.devices.values())


_N_CTX_PATTERN = re.compile(r"\bn_ctx\s*=\s*(\d+)")
_NON_SWA_CTX_PATTERN = re.compile(
    r"creating\s+non-SWA\s+KV\s+cache,\s*size\s*=\s*(\d+)\s*cells",
    re.IGNORECASE,
)
_KV_PATTERN = re.compile(
    r"\b(CUDA\d+|CPU)\s+KV\s+buffer\s+size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.IGNORECASE,
)
_GRANT_PATTERN = re.compile(
    r"\[moe-cache\]\s+(CUDA\d+)\s+capacity:.*?\bgranted=([0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.IGNORECASE,
)
_STATS_PATTERN = re.compile(
    r"\[moe-cache\]\s+(CUDA\d+)\s+hits=(\d+)/(\d+)\s+\(([0-9]+(?:\.[0-9]+)?)%\)(.*)",
    re.IGNORECASE,
)


def parse_llama_bench_json(text: str) -> tuple[LlamaBenchRecord, ...]:
    """Parse llama-bench JSON array, object, or JSONL output."""
    if not isinstance(text, str) or not text.strip():
        raise TelemetryError("invalid_input", "llama-bench 輸入必須是非空文字")
    raw_records = _decode_json_records(text)
    records = tuple(_parse_bench_record(record, index) for index, record in enumerate(raw_records))
    if not records:
        raise TelemetryError("invalid_schema", "llama-bench 結果不得為空")
    return records


def parse_moe_cache_log(text: str, *, require_complete: bool = True) -> MoeCacheTelemetry:
    """Extract 256K context, KV, cache grants, residency and failures."""
    if not isinstance(text, str) or not text.strip():
        raise TelemetryError("invalid_input", "MoE-cache log 必須是非空文字")
    if not isinstance(require_complete, bool):
        raise TelemetryError("invalid_input", "require_complete 必須是 bool")

    n_ctx = _parse_context_size(text)
    kv_mib = _parse_kv_buffers(text)
    grants = {device.upper(): float(amount) for device, amount in _GRANT_PATTERN.findall(text)}
    devices = _parse_device_stats(text)
    enabled = "[moe-cache] enabled" in text.lower()
    result = MoeCacheTelemetry(n_ctx, kv_mib, grants, devices, enabled)
    if require_complete:
        _require_complete_log(result)
    return result


def _parse_context_size(text: str) -> int | None:
    values = {
        int(value)
        for pattern in (_N_CTX_PATTERN, _NON_SWA_CTX_PATTERN)
        for value in pattern.findall(text)
    }
    if len(values) > 1:
        raise TelemetryError(
            "inconsistent_log",
            "log 包含衝突的 context 大小",
            {"n_ctx_values": sorted(values)},
        )
    return next(iter(values), None)


def _parse_kv_buffers(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for device, amount in _KV_PATTERN.findall(text):
        name = device.upper()
        totals[name] = totals.get(name, 0.0) + float(amount)
    return totals


def _parse_device_stats(text: str) -> dict[str, DeviceCacheStats]:
    devices: dict[str, DeviceCacheStats] = {}
    for match in _STATS_PATTERN.finditer(text):
        device, hits_text, probes_text, rate_text, suffix = match.groups()
        hits, probes = int(hits_text), int(probes_text)
        rate = float(rate_text)
        if probes <= 0 or hits > probes or not 0.0 <= rate <= 100.0:
            raise TelemetryError(
                "invalid_log_value",
                "cache hit 統計超出有效範圍",
                {"device": device.upper()},
            )
        devices[device.upper()] = DeviceCacheStats(
            hits=hits,
            probes=probes,
            hit_rate_percent=rate,
            fill_failures=_parse_counter(suffix, "fill-fail", device),
            dispatch_failures=_parse_counter(suffix, "dispatch-fail", device),
            collect_failures=_parse_counter(suffix, "collect-fail", device),
        )
    return devices


def _require_complete_log(result: MoeCacheTelemetry) -> None:
    evidence = {
        "n_ctx": result.n_ctx is not None,
        "kv_buffers": bool(result.kv_mib_by_device),
        "cache_enabled": result.enabled,
        "capacity_grants": bool(result.granted_mib_by_device),
        "cache_stats": bool(result.devices),
    }
    missing = [name for name, present in evidence.items() if not present]
    if missing:
        raise TelemetryError(
            "incomplete_log",
            "log 缺少必要驗收證據",
            {"missing": missing},
        )


def _decode_json_records(text: str) -> list[object]:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded_lines = []
        try:
            for line in text.splitlines():
                if line.strip():
                    decoded_lines.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise TelemetryError(
                "invalid_json",
                "llama-bench 輸出不是有效 JSON 或 JSONL",
                {"line": error.lineno, "column": error.colno},
            ) from error
        decoded = decoded_lines
    if isinstance(decoded, dict):
        return [decoded]
    if isinstance(decoded, list):
        return decoded
    raise TelemetryError("invalid_schema", "llama-bench 根節點必須是物件或陣列")


def _parse_bench_record(raw: object, index: int) -> LlamaBenchRecord:
    if not isinstance(raw, dict):
        raise TelemetryError("invalid_schema", "llama-bench record 必須是物件", {"index": index})
    n_ctx = _positive_int(raw.get("n_ctx"), "n_ctx", index)
    n_gen = _nonnegative_int(raw.get("n_gen"), "n_gen", index)
    samples_raw = raw.get("samples_ts")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise TelemetryError("invalid_schema", "samples_ts 必須是非空陣列", {"index": index})
    samples = tuple(_positive_finite(value, "samples_ts", index) for value in samples_raw)
    avg_raw = raw.get("avg_ts")
    avg_ts = sum(samples) / len(samples) if avg_raw is None else _positive_finite(avg_raw, "avg_ts", index)
    model = raw.get("model_filename", "")
    if not isinstance(model, str):
        raise TelemetryError("invalid_schema", "model_filename 必須是字串", {"index": index})
    return LlamaBenchRecord(
        model_filename=model,
        n_ctx=n_ctx,
        n_gen=n_gen,
        avg_ts=avg_ts,
        samples_ts=samples,
        p5_ts=_linear_percentile(samples, 0.05),
    )


def _positive_int(value: object, field: str, index: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TelemetryError("invalid_schema", f"{field} 必須是正整數", {"index": index})
    return value


def _nonnegative_int(value: object, field: str, index: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TelemetryError("invalid_schema", f"{field} 必須是非負整數", {"index": index})
    return value


def _positive_finite(value: object, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TelemetryError("invalid_schema", f"{field} 必須是數值", {"index": index})
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise TelemetryError("invalid_schema", f"{field} 必須有限且大於零", {"index": index})
    return normalized


def _parse_counter(suffix: str, name: str, device: str) -> int:
    match = re.search(rf"\b{re.escape(name)}=(\d+)", suffix)
    if not match:
        raise TelemetryError(
            "incomplete_log",
            f"cache stats 缺少 {name}",
            {"device": device.upper()},
        )
    return int(match.group(1))


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
