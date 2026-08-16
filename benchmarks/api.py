"""Small OpenAI-compatible HTTP client with SSE timing instrumentation."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
from urllib.parse import urlparse


class ApiError(RuntimeError):
    """An HTTP, transport, or response-contract failure."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class StreamEvent:
    """One received SSE event and its client-side receive timestamp."""

    index: int
    timestamp_utc: str
    elapsed_seconds: float
    event_type: str | None
    data: dict[str, Any] | None
    content_delta: str = ""
    reasoning_delta: str = ""
    finish_reason: str | None = None
    done: bool = False


@dataclass(frozen=True)
class StreamStatistics:
    """Latency and throughput derived from a streaming completion."""

    request_duration_seconds: float
    time_to_first_event_seconds: float | None
    time_to_first_output_seconds: float | None
    output_event_count: int
    completion_tokens: int
    token_count_source: str
    decode_duration_seconds: float | None
    decode_tokens_per_second: float | None
    end_to_end_tokens_per_second: float | None
    inter_output_event_seconds_p50: float | None
    inter_output_event_seconds_p95: float | None


@dataclass(frozen=True)
class StreamResult:
    """Combined streamed text, raw event timeline, usage, and statistics."""

    response_id: str | None
    model: str | None
    content: str
    reasoning_content: str
    finish_reason: str | None
    usage: dict[str, Any]
    events: tuple[StreamEvent, ...]
    statistics: StreamStatistics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAIClient:
    """Standard-library client for the OpenAI-compatible endpoints under test."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url 必須是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url 不得包含認證資訊")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout 必須是有限正數")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def list_models(self) -> dict[str, Any]:
        response = self.request_json("GET", "/v1/models")
        _validate_models_response(response)
        return response

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 128,
        temperature: float = 0.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = _chat_payload(
            model, messages, max_tokens, temperature, False, extra_body
        )
        response = self.request_json("POST", "/v1/chat/completions", payload)
        _validate_chat_response(response)
        return response

    def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> StreamResult:
        payload = _chat_payload(
            model, messages, max_tokens, temperature, True, extra_body
        )
        request = self._build_request("POST", "/v1/chat/completions", payload)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return _consume_stream(response, started)
        except urllib.error.HTTPError as error:
            raise _http_error(error, self._api_key) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ApiError(f"API transport error: {error}") from error

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = self._build_request(method, path, payload)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            raise _http_error(error, self._api_key) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ApiError(f"API transport error: {error}") from error
        return _decode_json_object(body, f"{method.upper()} {path}")

    def _build_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
    ) -> urllib.request.Request:
        if not path.startswith("/"):
            raise ValueError("API path 必須以 / 開頭")
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            f"{self._base_url}{path}", data=data, headers=headers, method=method
        )


def _chat_payload(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model 不得為空")
    if not messages:
        raise ValueError("messages 不得為空")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens 必須是整數")
    if max_tokens <= 0:
        raise ValueError("max_tokens 必須大於 0")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("temperature 必須是有限數值")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature 必須介於 0 與 2")
    protected = {"model", "messages", "stream", "stream_options"}
    if extra_body and protected.intersection(extra_body):
        raise ValueError("extra_body 不得覆寫 model/messages/stream/stream_options")
    payload: dict[str, Any] = {
        "model": model,
        "messages": _validated_messages(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if extra_body:
        payload.update(extra_body)
    try:
        json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("chat request body 必須可序列化為 JSON") from error
    return payload


def _validated_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] 必須是物件")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"messages[{index}].role 必須是非空字串")
        copied = dict(message)
        try:
            json.dumps(copied, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"messages[{index}] 必須可序列化為 JSON") from error
        validated.append(copied)
    return validated


def _consume_stream(stream: BinaryIO, started: float) -> StreamResult:
    events: list[StreamEvent] = []
    contents: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    response_id = model = finish_reason = None
    saw_done = False
    for event_type, raw_data in _iter_sse(stream):
        received = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        if raw_data == "[DONE]":
            saw_done = True
            done = _done_event(
                len(events), timestamp, received - started, event_type
            )
            events.append(done)
            break
        data = _decode_json_object(raw_data.encode("utf-8"), "SSE data")
        extracted = _extract_stream_fields(data)
        response_id = extracted[0] or response_id
        model = extracted[1] or model
        content, reasoning, event_finish = extracted[2:]
        finish_reason = event_finish or finish_reason
        contents.append(content)
        reasoning_parts.append(reasoning)
        if isinstance(data.get("usage"), dict):
            usage = dict(data["usage"])
        events.append(
            StreamEvent(
                index=len(events),
                timestamp_utc=timestamp,
                elapsed_seconds=received - started,
                event_type=event_type,
                data=data,
                content_delta=content,
                reasoning_delta=reasoning,
                finish_reason=event_finish,
            )
        )
    completed = time.perf_counter()
    if not events:
        raise ApiError("streaming response 沒有 SSE events")
    if not saw_done and finish_reason is None:
        raise ApiError("streaming response 在 [DONE] 或 finish_reason 前中斷")
    stats = _stream_statistics(events, usage, completed - started)
    return StreamResult(
        response_id=response_id,
        model=model,
        content="".join(contents),
        reasoning_content="".join(reasoning_parts),
        finish_reason=finish_reason,
        usage=usage,
        events=tuple(events),
        statistics=stats,
    )


def _iter_sse(stream: BinaryIO) -> Iterator[tuple[str | None, str]]:
    data_lines: list[str] = []
    event_type: str | None = None
    for raw_line in stream:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ApiError("SSE response 不是有效 UTF-8") from error
        if not line:
            if data_lines:
                yield event_type, "\n".join(data_lines)
            data_lines = []
            event_type = None
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data_lines.append(value)
        elif field == "event":
            event_type = value
    if data_lines:
        yield event_type, "\n".join(data_lines)


def _extract_stream_fields(
    data: Mapping[str, Any],
) -> tuple[str | None, str | None, str, str, str | None]:
    choices = data.get("choices", [])
    if not isinstance(choices, list):
        raise ApiError("SSE choices 必須是陣列")
    content = reasoning = ""
    finish_reason = None
    if choices:
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ApiError("SSE choice 必須是物件")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise ApiError("SSE delta 必須是物件")
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ApiError("SSE content/reasoning delta 必須是字串")
        candidate = choice.get("finish_reason")
        finish_reason = candidate if isinstance(candidate, str) else None
    return (
        _optional_string(data.get("id")),
        _optional_string(data.get("model")),
        content,
        reasoning,
        finish_reason,
    )


def _stream_statistics(
    events: Sequence[StreamEvent], usage: Mapping[str, Any], duration: float
) -> StreamStatistics:
    output_times = [
        event.elapsed_seconds
        for event in events
        if event.content_delta or event.reasoning_delta
    ]
    usage_tokens = usage.get("completion_tokens")
    if isinstance(usage_tokens, int) and usage_tokens >= 0:
        completion_tokens = usage_tokens
        source = "usage"
    else:
        completion_tokens = len(output_times)
        source = "output_event_count_fallback"
    decode_duration = None
    decode_tps = None
    if len(output_times) >= 2 and completion_tokens >= 2:
        decode_duration = output_times[-1] - output_times[0]
        if decode_duration > 0:
            decode_tps = (completion_tokens - 1) / decode_duration
    intervals = [right - left for left, right in zip(output_times, output_times[1:])]
    return StreamStatistics(
        request_duration_seconds=duration,
        time_to_first_event_seconds=events[0].elapsed_seconds if events else None,
        time_to_first_output_seconds=output_times[0] if output_times else None,
        output_event_count=len(output_times),
        completion_tokens=completion_tokens,
        token_count_source=source,
        decode_duration_seconds=decode_duration,
        decode_tokens_per_second=decode_tps,
        end_to_end_tokens_per_second=(
            completion_tokens / duration if duration > 0 else None
        ),
        inter_output_event_seconds_p50=_percentile(intervals, 0.50),
        inter_output_event_seconds_p95=_percentile(intervals, 0.95),
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _validate_chat_response(response: Mapping[str, Any]) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ApiError("chat completion 缺少 choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise ApiError("chat completion 缺少 choices[0].message")


def _validate_models_response(response: Mapping[str, Any]) -> None:
    models = response.get("data")
    if not isinstance(models, list):
        raise ApiError("/v1/models 缺少 data array")
    for index, model in enumerate(models):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise ApiError(f"/v1/models data[{index}] 缺少字串 id")


def _decode_json_object(body: bytes, source: str) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiError(f"{source} 回傳無效 JSON") from error
    if not isinstance(value, dict):
        raise ApiError(f"{source} JSON root 必須是物件")
    return value


def _http_error(error: urllib.error.HTTPError, api_key: str | None) -> ApiError:
    try:
        detail = error.read(2048).decode("utf-8", errors="replace")
    except OSError:
        detail = ""
    detail = " ".join(detail.split())[:1000]
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    suffix = f": {detail}" if detail else ""
    return ApiError(f"API HTTP {error.code}{suffix}", status_code=error.code)


def _done_event(
    index: int, timestamp: str, elapsed: float, event_type: str | None
) -> StreamEvent:
    return StreamEvent(
        index=index,
        timestamp_utc=timestamp,
        elapsed_seconds=elapsed,
        event_type=event_type,
        data=None,
        done=True,
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
