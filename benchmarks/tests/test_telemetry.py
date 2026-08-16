from __future__ import annotations

import unittest

from benchmarks.telemetry import TelemetryError, parse_llama_bench_json, parse_moe_cache_log


LLAMA_BENCH_JSON = """
[
  {
    "model_filename": "Laguna.gguf",
    "n_ctx": 262144,
    "n_gen": 256,
    "avg_ts": 12.21305,
    "samples_ts": [12.0748, 12.3399, 12.2245]
  }
]
"""

MOE_CACHE_LOG = """
llama_context: n_ctx = 262144
llama_kv_cache: CUDA0 KV buffer size = 3456.00 MiB
llama_kv_cache: CUDA0 KV buffer size = 40.50 MiB
[moe-cache] configured: mode=on devices=2 budget=12288 MiB cap
[moe-cache] CUDA0 capacity: cap=12288 MiB granted=4926 MiB free=7998 MiB reserve=3072 MiB
[moe-cache] CUDA1 capacity: cap=12288 MiB granted=8173 MiB free=11245 MiB reserve=3072 MiB
[moe-cache] enabled: first pool allocated on CUDA1
[moe-cache] CUDA0 hits=429684/551890 (77.9%) used=3619/3619 enqueued=14715 filled=14715 fill-fail=0 evictions=11096 skips=42 admission=107449 queue=0 jobs/0 MiB dispatch-fail=0 collect-fail=0
[moe-cache] CUDA1 hits=703873/889150 (79.2%) used=5945/5945 enqueued=22188 filled=22188 fill-fail=1 evictions=16243 skips=48 admission=163041 queue=0 jobs/0 MiB dispatch-fail=2 collect-fail=0
"""


class LlamaBenchParserTests(unittest.TestCase):
    def test_parses_valid_array_and_computes_p5(self) -> None:
        records = parse_llama_bench_json(LLAMA_BENCH_JSON)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].n_ctx, 262144)
        self.assertEqual(records[0].n_gen, 256)
        self.assertAlmostEqual(records[0].p5_ts, 12.08977, places=5)
        self.assertEqual(records[0].samples_ts, (12.0748, 12.3399, 12.2245))

    def test_accepts_jsonl_output(self) -> None:
        text = (
            '{"model_filename":"a","n_ctx":262144,"n_gen":128,"samples_ts":[10.0]}\n'
            '{"model_filename":"b","n_ctx":262144,"n_gen":128,"samples_ts":[11.0]}\n'
        )
        records = parse_llama_bench_json(text)
        self.assertEqual([record.model_filename for record in records], ["a", "b"])

    def test_rejects_bad_json_and_missing_context_with_structured_errors(self) -> None:
        for text, code in (("{", "invalid_json"), ('[{"samples_ts":[1]}]', "invalid_schema")):
            with self.subTest(code=code):
                with self.assertRaises(TelemetryError) as caught:
                    parse_llama_bench_json(text)
                self.assertEqual(caught.exception.to_dict()["error"]["code"], code)

    def test_rejects_nonpositive_or_nonfinite_throughput(self) -> None:
        for value in (0, -1, "NaN", "Infinity"):
            text = f'{{"n_ctx":262144,"n_gen":1,"samples_ts":[{value}]}}'
            with self.subTest(value=value):
                with self.assertRaises(TelemetryError) as caught:
                    parse_llama_bench_json(text)
                self.assertEqual(caught.exception.code, "invalid_schema")


class MoeCacheLogParserTests(unittest.TestCase):
    def test_parses_context_kv_grants_hits_and_failures(self) -> None:
        telemetry = parse_moe_cache_log(MOE_CACHE_LOG)

        self.assertEqual(telemetry.n_ctx, 262144)
        self.assertAlmostEqual(telemetry.kv_mib_by_device["CUDA0"], 3496.5)
        self.assertAlmostEqual(telemetry.kv_total_mib, 3496.5)
        self.assertEqual(telemetry.granted_mib_by_device, {"CUDA0": 4926.0, "CUDA1": 8173.0})
        self.assertEqual(telemetry.devices["CUDA0"].hits, 429684)
        self.assertEqual(telemetry.devices["CUDA1"].probes, 889150)
        self.assertAlmostEqual(telemetry.devices["CUDA0"].hit_rate_percent, 77.9)
        self.assertEqual(telemetry.devices["CUDA1"].fill_failures, 1)
        self.assertEqual(telemetry.devices["CUDA1"].dispatch_failures, 2)
        self.assertEqual(telemetry.total_failures, 3)
        self.assertTrue(telemetry.enabled)

    def test_parses_non_swa_context_line(self) -> None:
        text = MOE_CACHE_LOG.replace(
            "llama_context: n_ctx = 262144",
            "llama_kv_cache_iswa: creating non-SWA KV cache, size = 262144 cells",
        )
        self.assertEqual(parse_moe_cache_log(text).n_ctx, 262144)

    def test_incomplete_log_lists_missing_evidence(self) -> None:
        with self.assertRaises(TelemetryError) as caught:
            parse_moe_cache_log("llama_context: n_ctx = 262144")

        self.assertEqual(caught.exception.code, "incomplete_log")
        self.assertEqual(
            set(caught.exception.details["missing"]),
            {"kv_buffers", "cache_enabled", "capacity_grants", "cache_stats"},
        )

    def test_non_strict_mode_allows_startup_only_log(self) -> None:
        result = parse_moe_cache_log(
            "llama_context: n_ctx = 262144\nllama_kv_cache: CUDA0 KV buffer size = 1 MiB",
            require_complete=False,
        )
        self.assertFalse(result.enabled)
        self.assertEqual(result.granted_mib_by_device, {})

    def test_conflicting_context_values_fail_closed(self) -> None:
        text = MOE_CACHE_LOG + "\nllama_context: n_ctx = 131072\n"
        with self.assertRaises(TelemetryError) as caught:
            parse_moe_cache_log(text)
        self.assertEqual(caught.exception.code, "inconsistent_log")


if __name__ == "__main__":
    unittest.main()
