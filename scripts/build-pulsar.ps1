[CmdletBinding()]
param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $root "third_party/pulsar-runtime"
$patchDir = Join-Path $root "patches/pulsar"
$cargoHome = Join-Path $root ".cache/cargo-pulsar"
$target = Join-Path $root "build/pulsar"
$baseRevision = "c7245425dbc46cd8b310370a71d05fa30a9d9b34"
$repo = "https://github.com/giannisanni/pulsar.git"
$lockSha256 = "D92C6F39E04B495DABED0C5027278909835D3884FD5E429C9D463BB511B97884"

if (-not (Test-Path -LiteralPath $patchDir)) {
    throw "缺少 Pulsar patches：$patchDir"
}
if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $source) | Out-Null
    git clone --filter=blob:none $repo $source
    if ($LASTEXITCODE -ne 0) { throw "Pulsar clone 失敗" }
    git -C $source checkout --detach $baseRevision
    if ($LASTEXITCODE -ne 0) { throw "Pulsar checkout 失敗" }
    $patches = Get-ChildItem -LiteralPath $patchDir -Filter "*.patch" | Sort-Object Name
    if ($patches.Count -eq 0) { throw "Pulsar patch 清單為空" }
    git -C $source am ($patches.FullName)
    if ($LASTEXITCODE -ne 0) { throw "Pulsar patch 套用失敗；請檢查 $source" }
}

$base = git -C $source merge-base HEAD $baseRevision
if ($LASTEXITCODE -ne 0 -or $base.Trim() -ne $baseRevision) {
    throw "Pulsar source 不是固定 revision 的衍生版本：$source"
}
if (git -C $source status --porcelain) {
    throw "Pulsar source 有未提交修改：$source"
}

$env:CARGO_HOME = $cargoHome
$env:CARGO_TARGET_DIR = $target
$env:PULSAR_CUDA_ARCH = "86,120a"
$env:TMP = Join-Path $root "tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $cargoHome, $target, $env:TMP | Out-Null

$lockPath = Join-Path $source "Cargo.lock"
if (-not (Test-Path -LiteralPath $lockPath)) {
    cargo generate-lockfile --manifest-path (Join-Path $source "Cargo.toml")
    if ($LASTEXITCODE -ne 0) { throw "Pulsar dependency lock 產生失敗" }
}
$actualLockSha256 = (Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash
if ($actualLockSha256 -ne $lockSha256) {
    throw "Pulsar dependency lock 與已驗證版本不符：$actualLockSha256"
}

if (-not $SkipBuild) {
    cargo build --manifest-path (Join-Path $source "Cargo.toml") --locked --release `
        -p engine --bin pulsar-cli -p serve --bin pulsar-serve
    if ($LASTEXITCODE -ne 0) { throw "Pulsar Release build 失敗" }
}

Write-Output "Pulsar source 已固定於 $baseRevision；CUDA 架構為 sm_86 + sm_120a。"
