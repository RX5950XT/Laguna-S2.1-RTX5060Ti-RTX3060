from __future__ import annotations

import json
import io
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.api import ApiError, OpenAIClient, _consume_stream
from benchmarks.output import require_within, write_json
from benchmarks.prompts import generate_needle_prompt
from benchmarks.runner import BenchmarkConfig, run_benchmark


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/echo-secret":
            self._json_response(400, {"error": self.headers.get("Authorization")})
            return
        if self.path != "/v1/models":
            self._json_response(404, {"error": "not found"})
            return
        self._json_response(
            200,
            {
                "object": "list",
                "data": [{"id": "laguna-test", "object": "model"}],
            },
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)  # type: ignore[attr-defined]
        if self.path != "/v1/chat/completions":
            self._json_response(404, {"error": "not found"})
            return
        if payload.get("stream"):
            self._stream_response(payload)
            return
        self._json_response(
            200,
            {
                "id": "chat-1",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    def _stream_response(self, payload: dict[str, object]) -> None:
        messages = payload["messages"]
        prompt = str(messages[-1]["content"])  # type: ignore[index]
        if "Exact answer:" in prompt:
            answer = prompt.split("Exact answer:", 1)[1].splitlines()[0].strip()
            pieces = [answer[: len(answer) // 2], answer[len(answer) // 2 :]]
        else:
            pieces = ["alpha", " beta", " gamma"]
        events = [
            {
                "id": "chat-stream",
                "model": payload["model"],
                "choices": [{"delta": {"content": piece}, "finish_reason": None}],
            }
            for piece in pieces
        ]
        events.append(
            {
                "id": "chat-stream",
                "model": payload["model"],
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }
        )
        events.append(
            {
                "id": "chat-stream",
                "model": payload["model"],
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": len(pieces),
                },
            }
        )
        body = b"".join(
            b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"
            for event in events
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        midpoint = len(body) // 2
        self.wfile.write(body[:midpoint])
        self.wfile.flush()
        time.sleep(0.005)
        self.wfile.write(body[midpoint:])
        self.wfile.flush()

    def _json_response(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
        cls.server.requests = []  # type: ignore[attr-defined]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        self.server.requests.clear()  # type: ignore[attr-defined]
        self.client = OpenAIClient(self.base_url, api_key="secret", timeout=2)

    def test_models_and_non_streaming_chat(self) -> None:
        models = self.client.list_models()
        chat = self.client.chat(
            model="laguna-test",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
        )
        self.assertEqual(models["data"][0]["id"], "laguna-test")
        self.assertEqual(chat["choices"][0]["message"]["content"], "pong")

    def test_stream_records_each_event_and_token_statistics(self) -> None:
        result = self.client.stream_chat(
            model="laguna-test",
            messages=[{"role": "user", "content": "count"}],
            max_tokens=8,
        )
        self.assertEqual(result.content, "alpha beta gamma")
        self.assertEqual(result.statistics.completion_tokens, 3)
        self.assertEqual(result.statistics.token_count_source, "usage")
        self.assertGreater(result.statistics.end_to_end_tokens_per_second, 0)
        self.assertGreater(result.statistics.decode_tokens_per_second, 0)
        self.assertGreaterEqual(len(result.events), 5)
        elapsed = [event.elapsed_seconds for event in result.events]
        self.assertEqual(elapsed, sorted(elapsed))
        self.assertTrue(
            all(event.timestamp_utc.endswith("+00:00") for event in result.events)
        )
        request = self.server.requests[-1]  # type: ignore[attr-defined]
        self.assertEqual(request["stream_options"], {"include_usage": True})

    def test_stream_rejects_truncated_response_without_done(self) -> None:
        body = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        )
        with self.assertRaisesRegex(ApiError, "DONE"):
            _consume_stream(body, time.perf_counter())

    def test_full_benchmark_includes_needle_check(self) -> None:
        config = BenchmarkConfig(
            base_url=self.base_url,
            model=None,
            timeout=2,
            stream_max_tokens=8,
            context_window_tokens=262_144,
            prompt_tokens=256,
            chars_per_token=4,
            needle_depth=0.6,
            seed=7,
            long_max_tokens=16,
        )
        report = run_benchmark(config, api_key="secret")
        self.assertTrue(report["success"])
        self.assertEqual(report["configuration"]["model"], "laguna-test")
        self.assertTrue(report["long_context"]["needle_found"])
        self.assertEqual(report["long_context"]["prompt"]["estimated_tokens"], 256)
        self.assertNotIn("prompt_text", report["long_context"]["prompt"])
        self.assertIn("events", report["streaming"])

    def test_http_errors_are_structured(self) -> None:
        with self.assertRaises(ApiError) as caught:
            self.client.request_json("GET", "/missing")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertNotIn("secret", str(caught.exception))

    def test_http_error_redacts_api_key_echoed_by_server(self) -> None:
        with self.assertRaises(ApiError) as caught:
            self.client.request_json("GET", "/echo-secret")
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_invalid_message_is_rejected_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "role"):
            self.client.chat("laguna-test", [{"content": "missing role"}])
        self.assertEqual(self.server.requests, [])  # type: ignore[attr-defined]


class NeedlePromptTests(unittest.TestCase):
    def test_prompt_is_reproducible_and_exactly_sized(self) -> None:
        first = generate_needle_prompt(512, depth=0.75, seed=42, chars_per_token=4)
        second = generate_needle_prompt(512, depth=0.75, seed=42, chars_per_token=4)
        self.assertEqual(first, second)
        self.assertEqual(len(first.prompt), 2048)
        self.assertEqual(first.estimated_tokens, 512)
        self.assertIn(first.answer, first.prompt)
        self.assertGreater(first.actual_depth, 0.65)
        self.assertLess(first.actual_depth, 0.85)

    def test_invalid_prompt_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_needle_prompt(1, depth=0.5, seed=1, chars_per_token=4)
        with self.assertRaises(ValueError):
            generate_needle_prompt(100, depth=1.1, seed=1, chars_per_token=4)


class JsonOutputTests(unittest.TestCase):
    def test_write_json_creates_valid_utf8_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json(path, {"success": True, "text": "臺灣"})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["text"], "臺灣")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_output_boundary_rejects_external_path(self) -> None:
        with tempfile.TemporaryDirectory() as project_directory:
            root = Path(project_directory) / "project"
            root.mkdir()
            self.assertEqual(require_within(root / "result.json", root).parent, root)
            with self.assertRaises(ValueError):
                require_within(root.parent / "outside.json", root)


if __name__ == "__main__":
    unittest.main()
