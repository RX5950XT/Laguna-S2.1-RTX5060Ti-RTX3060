$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pidPath = Join-Path $root "logs/server.pid"
$server = Join-Path $root "build/llama-cpp-cuda/bin/Release/llama-server.exe"
$moeCachePidPath = Join-Path $root "logs/moe-cache-server.pid"
$moeCacheServer = Join-Path $root "build/llama.cpp-moe-cache-v2/bin/llama-server.exe"
$pulsarPidPath = Join-Path $root "logs/pulsar-server.pid"
$pulsarServer = Join-Path $root "build/pulsar/release/pulsar-serve.exe"
$serverPids = @()
if (Test-Path -LiteralPath $pidPath) {
    $serverPids += [int](Get-Content -Raw -LiteralPath $pidPath)
}
if (Test-Path -LiteralPath $moeCachePidPath) {
    $serverPids += [int](Get-Content -Raw -LiteralPath $moeCachePidPath)
}
$serverPids += @(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object {
    $_.ExecutablePath -in @($server, $moeCacheServer) -and $_.CommandLine -like "*$root*"
} | ForEach-Object { [int]$_.ProcessId })
$serverPids = @($serverPids | Sort-Object -Unique)
if ($serverPids.Count -eq 0) {
    Write-Output "沒有執行中的專案 llama-server。"
} else {
    $stoppedPids = @()
    foreach ($serverPid in $serverPids) {
        $process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $serverPid
            $null = $process.WaitForExit(30000)
            $stoppedPids += $serverPid
        }
    }
    if ($stoppedPids.Count -gt 0) {
        Write-Output "llama-server 已停止：$($stoppedPids -join ', ')。"
    } else {
        Write-Output "已清除過期 PID 紀錄，沒有執行中的專案 llama-server。"
    }
}
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $moeCachePidPath -Force -ErrorAction SilentlyContinue

$pulsarPids = @()
if (Test-Path -LiteralPath $pulsarPidPath) {
    $pulsarPids += [int](Get-Content -Raw -LiteralPath $pulsarPidPath)
}
$pulsarPids += @(Get-CimInstance Win32_Process -Filter "Name='pulsar-serve.exe'" | Where-Object {
    $_.ExecutablePath -eq $pulsarServer -and $_.CommandLine -like "*$root*"
} | ForEach-Object { [int]$_.ProcessId })
$pulsarPids = @($pulsarPids | Sort-Object -Unique)
$stoppedPulsar = @()
foreach ($pulsarPid in $pulsarPids) {
    $process = Get-Process -Id $pulsarPid -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $pulsarPid
        $null = $process.WaitForExit(30000)
        $stoppedPulsar += $pulsarPid
    }
}
Remove-Item -LiteralPath $pulsarPidPath -Force -ErrorAction SilentlyContinue
if ($stoppedPulsar.Count -gt 0) {
    Write-Output "Pulsar server 已停止：$($stoppedPulsar -join ', ')。"
}
