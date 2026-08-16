[CmdletBinding()]
param(
    [ValidateSet("apex_subcompact", "apex_compact")]
    [string]$Variant = "apex_subcompact",
    [ValidateSet("q8_0", "q4_0", "fp8", "fp16", "f32")]
    [string]$KvType = "q8_0",
    [ValidateRange(1, 48)]
    [int]$HostCacheGiB = 48,
    [ValidateRange(1, 8)]
    [int]$DeviceCacheGiB = 2,
    [ValidateRange(1, 256)]
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$server = Join-Path $root "build/pulsar/release/pulsar-serve.exe"
$models = [ordered]@{
    apex_subcompact = Join-Path $root "models/apex_subcompact/Laguna-S-2.1-APEX-i-subcompact.gguf"
    apex_compact = Join-Path $root "models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf"
}
$model = $models[$Variant]
$logs = Join-Path $root "logs"
$tmp = Join-Path $root "tmp"
$pidPath = Join-Path $logs "pulsar-server.pid"

if (-not (Test-Path -LiteralPath $server)) {
    & (Join-Path $PSScriptRoot "build-pulsar.ps1")
}
if (-not (Test-Path -LiteralPath $server)) { throw "缺少 Pulsar server：$server" }
if (-not (Test-Path -LiteralPath $model)) { throw "缺少模型：$model" }
New-Item -ItemType Directory -Force -Path $logs, $tmp | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -Raw -LiteralPath $pidPath)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) {
        throw "Pulsar server 已在執行，PID $oldPid"
    }
    Remove-Item -LiteralPath $pidPath -Force
}
$running = Get-CimInstance Win32_Process -Filter "Name='pulsar-serve.exe'" | Where-Object {
    $_.ExecutablePath -eq $server -and $_.CommandLine -like "*$root*"
}
if ($running) { throw "偵測到專案 Pulsar server 已在執行，PID $(($running.ProcessId) -join ', ')" }

$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:PULSAR_GPU = "0"
$env:PULSAR_ATTN_GPU = "off"
$env:PULSAR_SPLIT = "off"
$env:PULSAR_CPU = "1"
$env:PULSAR_CPU_CAP = "1"
$env:PULSAR_CPU_STEAL = "0"
$env:PULSAR_CPU_B = "0"
$env:PULSAR_CACHE_GB = $HostCacheGiB.ToString()
$env:PULSAR_DEV_CACHE_GB = $DeviceCacheGiB.ToString()
$env:PULSAR_BATCH = $BatchSize.ToString()
$env:PULSAR_KV = $KvType
$env:PULSAR_OFFLINE = "1"
$env:PULSAR_TEMPLATE_CACHE = Join-Path $root ".cache/pulsar-template"
$env:TMP = $tmp
$env:TEMP = $tmp
New-Item -ItemType Directory -Force -Path $env:PULSAR_TEMPLATE_CACHE | Out-Null

$process = Start-Process -FilePath $server -ArgumentList @(
    "-m", $model,
    "--host", "127.0.0.1",
    "--port", "11435",
    "--ctx", "262144",
    "--jinja-chat"
) -WorkingDirectory $root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logs "pulsar-server.stdout.log") `
    -RedirectStandardError (Join-Path $logs "pulsar-server.stderr.log")
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii

$deadline = (Get-Date).AddMinutes(3)
$modelsResponse = $null
while ((Get-Date) -lt $deadline) {
    $process.Refresh()
    if ($process.HasExited) {
        $tail = (Get-Content (Join-Path $logs "pulsar-server.stderr.log") -Tail 30) -join [Environment]::NewLine
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        throw "Pulsar server 載入期間退出（code $($process.ExitCode)）：`n$tail"
    }
    try {
        $modelsResponse = Invoke-RestMethod -Uri "http://127.0.0.1:11435/v1/models" -TimeoutSec 2
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $modelsResponse) {
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    throw "Pulsar server 未在 3 分鐘內啟動"
}
$reportedContext = $modelsResponse.data[0].max_model_len
if ($reportedContext -ne 262144) {
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    throw "Pulsar 回報 context=$reportedContext，拒絕降級"
}
Write-Output "Pulsar server 已啟動，PID $($process.Id)；context=262144，模型=$Variant。"
