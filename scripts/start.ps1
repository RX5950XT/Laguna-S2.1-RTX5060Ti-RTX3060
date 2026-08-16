[CmdletBinding()]
param(
    [ValidateSet("apex_compact", "apex_subcompact", "unsloth_iq4_xs")]
    [string]$Variant = "apex_compact",
    [ValidateRange(0, 48)]
    [int]$CpuMoeLayers = 48,
    [ValidatePattern("^\d+(\.\d+)?,\d+(\.\d+)?$")]
    [string]$TensorSplit = "57,43",
    [switch]$AutoTensorSplit,
    [ValidateSet("layer", "row", "tensor")]
    [string]$SplitMode = "layer",
    [ValidateSet("f16", "bf16", "q8_0", "q4_0")]
    [string]$KvType = "q8_0",
    [ValidatePattern("^(auto|all|\d+)$")]
    [string]$GpuLayers = "auto",
    [ValidateRange(1, 16)]
    [int]$Threads = 8,
    [switch]$NoMmap,
    [switch]$DFlash
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$server = Join-Path $root "build/llama-cpp-cuda/bin/Release/llama-server.exe"
$logs = Join-Path $root "logs"
$tmp = Join-Path $root "tmp"
$models = [ordered]@{
    apex_compact = Join-Path $root "models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf"
    apex_subcompact = Join-Path $root "models/apex_subcompact/Laguna-S-2.1-APEX-i-subcompact.gguf"
    unsloth_iq4_xs = Join-Path $root "models/unsloth_iq4_xs/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf"
}
$model = $models[$Variant]
$drafts = [ordered]@{
    apex_compact = Join-Path $root "models/apex_compact/laguna-s-2.1-DFlash-Q4_K.gguf"
    apex_subcompact = Join-Path $root "models/apex_subcompact/laguna-s-2.1-DFlash-Q4_K.gguf"
    unsloth_iq4_xs = Join-Path $root "models/poolside_dflash/laguna-s-2.1-DFlash-BF16.gguf"
}

if (-not (Test-Path -LiteralPath $server)) { throw "缺少 llama-server：$server" }
if (-not (Test-Path -LiteralPath $model)) { throw "缺少模型：$model" }
New-Item -ItemType Directory -Force -Path $logs, $tmp | Out-Null
$pidPath = Join-Path $logs "server.pid"
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -Raw -LiteralPath $pidPath)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { throw "server 已在執行，PID $oldPid" }
    Remove-Item -LiteralPath $pidPath -Force
}
$running = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object {
    $_.ExecutablePath -eq $server -and $_.CommandLine -like "*$root*"
}
if ($running) {
    $runningPid = [int]($running | Select-Object -First 1).ProcessId
    $runningPid | Set-Content -LiteralPath $pidPath -Encoding ascii
    throw "偵測到專案 llama-server 已在執行，PID $runningPid"
}
$env:HF_HOME = Join-Path $root ".cache/huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"
$env:TMP = $tmp
$env:TEMP = $tmp

$arguments = @(
    "--model", $model,
    "--ctx-size", "262144",
    "--host", "127.0.0.1",
    "--port", "8000",
    "--parallel", "1",
    "--jinja",
    "--flash-attn", "on",
    "--cache-type-k", $KvType,
    "--cache-type-v", $KvType,
    "--gpu-layers", $GpuLayers,
    "--split-mode", $SplitMode,
    "--n-cpu-moe", $CpuMoeLayers,
    "--threads", $Threads,
    "--threads-batch", "16",
    "--batch-size", "2048",
    "--ubatch-size", "512"
)
if (-not $AutoTensorSplit) { $arguments += @("--tensor-split", $TensorSplit) }
if ($NoMmap) { $arguments += "--no-mmap" }
if ($DFlash) {
    $draft = $drafts[$Variant]
    if (-not (Test-Path -LiteralPath $draft)) { throw "缺少 DFlash：$draft" }
    $arguments += @(
        "--model-draft", $draft,
        "--spec-type", "draft-dflash",
        "--spec-draft-n-max", "7",
        "--spec-draft-override-tensor", ".*=CUDA0"
    )
}

$process = Start-Process -FilePath $server -ArgumentList $arguments -WorkingDirectory $root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs "server.stdout.log") `
    -RedirectStandardError (Join-Path $logs "server.stderr.log")
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$deadline = (Get-Date).AddMinutes(3)
$health = $null
while ((Get-Date) -lt $deadline) {
    $process.Refresh()
    if ($process.HasExited) {
        $tail = if (Test-Path (Join-Path $logs "server.stderr.log")) {
            (Get-Content (Join-Path $logs "server.stderr.log") -Tail 20) -join [Environment]::NewLine
        } else { "無 stderr log" }
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        throw "llama-server 載入期間退出（code $($process.ExitCode)）：`n$tail"
    }
    try {
        $health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($health.StatusCode -eq 200) { break }
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $health -or $health.StatusCode -ne 200) {
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    throw "llama-server 未在 3 分鐘內通過 health check"
}
Write-Output "llama-server 已啟動，PID $($process.Id)；context 固定 262144。"
