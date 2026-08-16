import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MoeCacheRuntimeScriptTests(unittest.TestCase):
    def test_build_script_pins_source_and_cuda_architectures(self) -> None:
        script = (ROOT / "scripts" / "build-moe-cache.ps1").read_text(encoding="utf-8")

        self.assertIn("e3096b046bb809f7f80bc47801f6579aed1cbc60", script)
        self.assertIn("https://github.com/leloch/llama.cpp.git", script)
        self.assertIn("86-real;120a-real", script)
        self.assertIn("vswhere.exe", script)
        self.assertIn("vcvars64.bat", script)
        self.assertIn("llama-server", script)
        self.assertIn("test-moe-cache", script)
        self.assertIn("patches/moe-cache", script)
        self.assertIn("git -C $source am", script)

    def test_bench_patch_exposes_and_reports_real_context_size(self) -> None:
        patches = sorted((ROOT / "patches" / "moe-cache").glob("*.patch"))

        self.assertGreaterEqual(len(patches), 1)
        combined = "\n".join(path.read_text(encoding="utf-8") for path in patches)
        self.assertIn('--ctx-size', combined)
        self.assertIn('"n_ctx"', combined)
        self.assertIn("cparams.n_ctx", combined)

    def test_start_script_preserves_acceptance_boundaries(self) -> None:
        script = (ROOT / "scripts" / "start-moe-cache.ps1").read_text(encoding="utf-8")

        self.assertIn('"--ctx-size", "262144"', script)
        self.assertIn('"--host", "127.0.0.1"', script)
        self.assertIn('"--cors-origins", "localhost"', script)
        self.assertIn('"--no-webui"', script)
        self.assertIn('"--n-cpu-moe", "48"', script)
        self.assertIn('"--moe-cache"', script)
        self.assertIn("models/apex_compact", script)
        self.assertIn('[string]$KvType = "q4_0"', script)
        self.assertIn('[int]$MoeCacheMiB = 12288', script)
        self.assertIn('$env:GGML_CUDA_MOE_CACHE_RESERVE_MB = "2048"', script)
        self.assertIn('"--threads", "4"', script)
        self.assertIn('"--ubatch-size", "160"', script)

    def test_stop_script_covers_moe_cache_server(self) -> None:
        script = (ROOT / "scripts" / "stop.ps1").read_text(encoding="utf-8")

        self.assertIn("logs/moe-cache-server.pid", script)
        self.assertIn("build/llama.cpp-moe-cache-v2/bin/llama-server.exe", script)

    def test_tuner_is_project_local_resumable_and_restores_environment(self) -> None:
        script = (ROOT / "scripts" / "tune-moe-cache.ps1").read_text(encoding="utf-8")

        self.assertIn("config/moe-cache-search.json", script)
        self.assertIn("benchmarks/moe-cache-search", script)
        self.assertIn("--ctx-size", script)
        self.assertIn("262144", script)
        self.assertIn("Test-Path -LiteralPath $resultPath", script)
        self.assertIn("finally", script)
        self.assertIn("GGML_CUDA_MOE_CACHE_RESERVE_MB", script)
        self.assertIn("GGML_CUDA_MOE_CACHE_STATS", script)
        self.assertIn("WaitForExit", script)

    def test_search_config_preserves_acceptance_boundary_and_unique_arms(self) -> None:
        config = json.loads(
            (ROOT / "config" / "moe-cache-search.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["context_size"], 262144)
        self.assertEqual(config["minimum_decode_tps"], 20.0)
        self.assertEqual(config["model"], "models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf")
        names = [arm["name"] for arm in config["arms"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("ubatch-8", names)
        self.assertIn("reserve-2048", names)

    def test_tuner_plan_only_is_fixed_and_project_local(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        self.assertIsNotNone(powershell)
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-File",
                str(ROOT / "scripts" / "tune-moe-cache.ps1"),
                "-PlanOnly",
                "-Case",
                "ubatch-8",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
        )
        plan = json.loads(completed.stdout)

        self.assertEqual(len(plan["cases"]), 1)
        case = plan["cases"][0]
        self.assertEqual(case["context_size"], 262144)
        self.assertEqual(case["n_gen"], 256)
        self.assertIn("--ctx-size", case["arguments"])
        self.assertTrue(Path(case["model"]).is_relative_to(ROOT))
        self.assertTrue(Path(case["stdout"]).is_relative_to(ROOT))
        self.assertTrue(Path(case["stderr"]).is_relative_to(ROOT))

    def test_tuner_plan_supports_auto_cache_with_repacking(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-File",
                str(ROOT / "scripts" / "tune-moe-cache.ps1"),
                "-PlanOnly",
                "-Case",
                "auto-repack",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
        )
        case = json.loads(completed.stdout)["cases"][0]

        self.assertEqual(case["arguments"][case["arguments"].index("--moe-cache") + 1], "auto")
        self.assertIn("--repack", case["arguments"])
        self.assertIn("on", case["arguments"])
        self.assertNotIn("-nr", case["arguments"])

    def test_tuner_identity_changes_when_runtime_dll_changes(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        runtime_dir = ROOT / "tmp" / "runtime-hash-test"
        shutil.rmtree(runtime_dir, ignore_errors=True)
        runtime_dir.mkdir(parents=True)
        bench = runtime_dir / "llama-bench.exe"
        dll = runtime_dir / "ggml-cuda.dll"
        bench.write_bytes(b"bench")
        dll.write_bytes(b"cuda-v1")

        command = [
            powershell,
            "-NoProfile",
            "-File",
            str(ROOT / "scripts" / "tune-moe-cache.ps1"),
            "-PlanOnly",
            "-Case",
            "ubatch-8",
            "-RuntimePath",
            str(bench),
        ]
        try:
            first = json.loads(
                subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=True,
                ).stdout
            )["cases"][0]["binary_sha256"]
            dll.write_bytes(b"cuda-v2")
            second = json.loads(
                subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    check=True,
                ).stdout
            )["cases"][0]["binary_sha256"]
        finally:
            shutil.rmtree(runtime_dir, ignore_errors=True)

        self.assertNotEqual(first, second)

    def test_profile_script_uses_bounded_project_local_nsight_trace(self) -> None:
        script = (ROOT / "scripts" / "profile-moe-cache.ps1").read_text(encoding="utf-8")
        bench = (ROOT / "third_party" / "llama.cpp-moe-cache-v2" / "tools" / "llama-bench" / "llama-bench.cpp").read_text(encoding="utf-8")
        cmake = (ROOT / "third_party" / "llama.cpp-moe-cache-v2" / "tools" / "llama-bench" / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("Nsight Systems", script)
        self.assertIn('"profile"', script)
        self.assertIn('"--trace=cuda,nvtx"', script)
        self.assertNotIn("osrt", script)
        self.assertIn('"--sample=none"', script)
        self.assertIn('"--cuda-graph-trace=graph"', script)
        self.assertIn('"--capture-range=cudaProfilerApi"', script)
        self.assertIn('"--capture-range-end=stop-shutdown"', script)
        self.assertIn("LLAMA_BENCH_PROFILE_GENERATION", script)
        self.assertIn("cudaProfilerStart()", bench)
        self.assertIn("cudaProfilerStop()", bench)
        self.assertIn("CUDA::cudart", cmake)
        self.assertIn('"--ctx-size", "262144"', script)
        self.assertIn("GGML_CUDA_MOE_CACHE_STATS", script)
        self.assertIn("benchmarks/moe-cache-profiles", script)
        self.assertIn("finally", script)


if __name__ == "__main__":
    unittest.main()
