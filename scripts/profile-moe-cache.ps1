[CmdletBinding()]
param(
    [ValidateRange(64, 512)]
    [int]$GenerationTokens = 128,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bench = Join-Path $root "build/llama.cpp-moe-cache-v2/bin/llama-bench.exe"
$model = Join-Path $root "models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf"
$outputRoot = Join-Path $root "benchmarks/moe-cache-profiles"
$tempRoot = Join-Path $root "tmp/moe-cache-profile"
$traceBase = Join-Path $outputRoot "split-experts-generation"
$stdoutPath = "$traceBase.stdout.json"
$stderrPath = "$traceBase.stderr.log"
$summaryPath = "$traceBase.stats.txt"
$reportPath = "$traceBase.nsys-rep"

$nsysCandidates = @(
    "C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe"
)
$nsys = $nsysCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $nsys) { throw "找不到相容的 Nsight Systems CLI" }
if (-not (Test-Path -LiteralPath $bench)) { throw "缺少 llama-bench：$bench" }
if (-not (Test-Path -LiteralPath $model)) { throw "缺少模型：$model" }

$benchArguments = @(
    "-m", $model,
    "--ctx-size", "262144",
    "-p", "0", "-n", "$GenerationTokens",
    "--n-gen-warmup", "256",
    "-r", "1", "-o", "json",
    "-ctk", "q4_0", "-ctv", "q4_0",
    "-ngl", "99", "-fa", "on", "-nr",
    "-sm", "layer", "-ts", "1/0", "-mg", "0",
    "-t", "4", "--poll", "50",
    "-b", "2048", "-ub", "160",
    "-ncmoe", "48", "--moe-cache", "12288",
    "-lm", "mmap", "-v", "--progress"
)
$profileArguments = @(
    "profile",
    "--trace=cuda,nvtx",
    "--sample=none",
    "--backtrace=none",
    "--cuda-graph-trace=graph",
    "--capture-range=cudaProfilerApi",
    "--capture-range-end=stop-shutdown",
    "--wait=primary",
    "--force-overwrite=true",
    "--stats=false",
    "--output=$traceBase",
    $bench
) + $benchArguments

if ($PlanOnly) {
    [ordered]@{
        nsys=$nsys; output=$reportPath; stdout=$stdoutPath; stderr=$stderrPath
        environment=[ordered]@{
            GGML_CUDA_MOE_CACHE_RESERVE_MB="2048"
            GGML_CUDA_MOE_CACHE_STATS="64"
            GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS="1"
            LLAMA_BENCH_PROFILE_GENERATION="1"
        }
        arguments=$profileArguments
    } | ConvertTo-Json -Depth 8
    exit 0
}

if (Get-CimInstance Win32_Process -Filter "Name='llama-bench.exe' OR Name='llama-server.exe'") {
    throw "已有 llama runtime 執行中，拒絕產生受污染 profile"
}
New-Item -ItemType Directory -Force -Path $outputRoot, $tempRoot | Out-Null
$environmentNames = @("GGML_CUDA_MOE_CACHE_RESERVE_MB", "GGML_CUDA_MOE_CACHE_STATS", "GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS", "LLAMA_BENCH_PROFILE_GENERATION", "TMP", "TEMP")
$snapshot = @{}
foreach ($name in $environmentNames) {
    $snapshot[$name] = [ordered]@{
        exists = Test-Path -LiteralPath "Env:$name"
        value = [Environment]::GetEnvironmentVariable($name, "Process")
    }
}

try {
    $env:GGML_CUDA_MOE_CACHE_RESERVE_MB = "2048"
    $env:GGML_CUDA_MOE_CACHE_STATS = "64"
    $env:GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS = "1"
    $env:LLAMA_BENCH_PROFILE_GENERATION = "1"
    $env:TMP = $tempRoot
    $env:TEMP = $tempRoot
    & $nsys @profileArguments 1> $stdoutPath 2> $stderrPath
    if ($LASTEXITCODE -ne 0) { throw "Nsight Systems profile 失敗：exit $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $reportPath)) { throw "缺少 Nsight report：$reportPath" }
    & $nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format column $reportPath |
        Set-Content -LiteralPath $summaryPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Nsight Systems stats 失敗：exit $LASTEXITCODE" }
    Write-Output "Nsight profile 已完成：$reportPath"
} finally {
    foreach ($name in $environmentNames) {
        if ($snapshot[$name].exists) {
            [Environment]::SetEnvironmentVariable($name, $snapshot[$name].value, "Process")
        } else {
            [Environment]::SetEnvironmentVariable($name, $null, "Process")
        }
    }
}
