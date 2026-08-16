[CmdletBinding()]
param(
    [ValidateSet("q8_0", "q4_0")]
    [string]$KvType = "q4_0",
    [ValidateRange(1024, 16384)]
    [int]$MoeCacheMiB = 12288
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$server = Join-Path $root "build/llama.cpp-moe-cache-v2/bin/llama-server.exe"
$model = Join-Path $root "models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf"
$logs = Join-Path $root "logs"
$tmp = Join-Path $root "tmp"
$pidPath = Join-Path $logs "moe-cache-server.pid"

if (-not (Test-Path -LiteralPath $server)) {
    & (Join-Path $PSScriptRoot "build-moe-cache.ps1")
}
if (-not (Test-Path -LiteralPath $server)) { throw "缺少 MoE-cache server：$server" }
if (-not (Test-Path -LiteralPath $model)) { throw "缺少模型：$model" }
New-Item -ItemType Directory -Force -Path $logs, $tmp | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -Raw -LiteralPath $pidPath)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        throw "MoE-cache server 已在執行，PID $oldPid"
    }
    Remove-Item -LiteralPath $pidPath -Force
}
$running = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object {
    $_.ExecutablePath -eq $server -and $_.CommandLine -like "*$root*"
}
if ($running) { throw "偵測到專案 MoE-cache server 已在執行，PID $(($running.ProcessId) -join ', ')" }

$env:HF_HOME = Join-Path $root ".cache/huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"
$env:TMP = $tmp
$env:TEMP = $tmp
$env:GGML_CUDA_MOE_CACHE_RESERVE_MB = "2048"
$env:GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS = "1"

$arguments = @(
    "--model", $model,
    "--ctx-size", "262144",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--cors-origins", "localhost",
    "--no-webui",
    "--parallel", "1",
    "--jinja",
    "--flash-attn", "on",
    "--cache-type-k", $KvType,
    "--cache-type-v", $KvType,
    "--gpu-layers", "99",
    "--n-cpu-moe", "48",
    "--moe-cache", $MoeCacheMiB.ToString(),
    "--no-repack",
    "--split-mode", "layer",
    "--tensor-split", "1,0",
    "--main-gpu", "0",
    "--threads", "4",
    "--threads-batch", "16",
    "--batch-size", "2048",
    "--ubatch-size", "160"
)

$process = Start-Process -FilePath $server -ArgumentList $arguments -WorkingDirectory $root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs "moe-cache-server.stdout.log") `
    -RedirectStandardError (Join-Path $logs "moe-cache-server.stderr.log")
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii

$deadline = (Get-Date).AddMinutes(3)
$modelsResponse = $null
while ((Get-Date) -lt $deadline) {
    $process.Refresh()
    if ($process.HasExited) {
        $tail = (Get-Content (Join-Path $logs "moe-cache-server.stderr.log") -Tail 30) -join [Environment]::NewLine
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        throw "MoE-cache server 載入期間退出（code $($process.ExitCode)）：`n$tail"
    }
    try {
        $modelsResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/models" -TimeoutSec 2
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $modelsResponse) {
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    throw "MoE-cache server 未在 3 分鐘內啟動"
}
Write-Output "MoE-cache server 已啟動，PID $($process.Id)；context 固定 262144，KV=$KvType。"
