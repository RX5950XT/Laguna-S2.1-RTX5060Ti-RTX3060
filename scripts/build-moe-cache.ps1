[CmdletBinding()]
param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = Join-Path $root "third_party/llama.cpp-moe-cache-v2"
$build = Join-Path $root "build/llama.cpp-moe-cache-v2"
$patchDir = Join-Path $root "patches/moe-cache"
$revision = "e3096b046bb809f7f80bc47801f6579aed1cbc60"
$repo = "https://github.com/leloch/llama.cpp.git"

function Import-MsvcEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { throw "找不到 vswhere.exe" }
    $installation = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
    if (-not $installation) { throw "找不到 Visual Studio C++ toolchain" }
    $vcvars = Join-Path $installation.Trim() "VC/Auxiliary/Build/vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) { throw "找不到 vcvars64.bat" }

    $environment = & $env:ComSpec /d /s /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw "載入 MSVC build environment 失敗" }
    foreach ($line in $environment) {
        $separator = $line.IndexOf("=")
        if ($separator -le 0) { continue }
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

if (-not (Test-Path -LiteralPath $patchDir)) { throw "缺少 MoE-cache patches：$patchDir" }
$patches = @(Get-ChildItem -LiteralPath $patchDir -Filter "*.patch" | Sort-Object Name)
if ($patches.Count -eq 0) { throw "MoE-cache patch 清單為空" }

if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $source) | Out-Null
    git clone --filter=blob:none --branch moe-cache-v2-pr $repo $source
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp MoE-cache clone 失敗" }
    git -C $source checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp MoE-cache checkout 失敗" }
}

if (git -C $source status --porcelain) {
    throw "llama.cpp MoE-cache source 有未追蹤檔案：$source"
}
$base = (git -C $source merge-base HEAD $revision).Trim()
if ($LASTEXITCODE -ne 0 -or $base -ne $revision) {
    throw "llama.cpp MoE-cache source 不是固定 revision 的衍生版本：$source"
}
$head = (git -C $source rev-parse HEAD).Trim()
if ($head -eq $revision) {
    git -C $source am ($patches.FullName)
    if ($LASTEXITCODE -ne 0) { throw "MoE-cache patch 套用失敗；請檢查 $source" }
}
$patchCount = [int](git -C $source rev-list --count "$revision..HEAD")
if ($LASTEXITCODE -ne 0 -or $patchCount -ne $patches.Count) {
    throw "MoE-cache patch 數量不符：source=$patchCount manifest=$($patches.Count)"
}

$env:TMP = Join-Path $root "tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $build, $env:TMP | Out-Null
Import-MsvcEnvironment

cmake -S $source -B $build -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    "-DCMAKE_CUDA_ARCHITECTURES=86-real;120a-real" `
    -DGGML_CUDA_FA=ON `
    -DGGML_CUDA_GRAPHS=ON `
    -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_BUILD_TESTS=ON `
    -DLLAMA_BUILD_TOOLS=ON `
    -DBUILD_SHARED_LIBS=ON
if ($LASTEXITCODE -ne 0) { throw "llama.cpp MoE-cache configure 失敗" }

if (-not $SkipBuild) {
    cmake --build $build --config Release --target llama-server llama-bench test-moe-cache --parallel 12
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp MoE-cache build 失敗" }
}

Write-Output "llama.cpp MoE-cache 已固定於 $revision；CUDA 架構為 sm_86 + sm_120a。"
