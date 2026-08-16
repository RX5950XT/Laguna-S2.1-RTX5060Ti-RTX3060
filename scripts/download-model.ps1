[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet("apex_compact", "apex_subcompact", "poolside_baseline", "poolside_dflash", "unsloth_iq4_xs", "unsloth_iq3_xxs", "unsloth_iq2_m")]
    [string]$Variant = "apex_compact",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $projectRoot "config/models.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json -AsHashtable
$entry = $manifest[$Variant]
if (-not $entry) { throw "未知模型候選：$Variant" }

$cache = Join-Path $projectRoot ".cache/huggingface"
$target = Join-Path $projectRoot ("models/" + $Variant)
$temporary = Join-Path $projectRoot "tmp"
$env:HF_HOME = $cache
$env:HF_HUB_CACHE = Join-Path $cache "hub"
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:HF_XET_CACHE = Join-Path $cache "xet"
$env:TMP = $temporary
$env:TEMP = $temporary

$arguments = @(
    "download", $entry.repo_id,
    "--revision", $entry.revision,
    "--local-dir", $target,
    "--max-workers", "4"
)
foreach ($file in $entry.files) { $arguments += @("--include", $file.path) }
if ($DryRun) { $arguments += "--dry-run" }

if ($PSCmdlet.ShouldProcess($target, "下載固定 revision 的 $Variant")) {
    New-Item -ItemType Directory -Force -Path $cache, $target, $temporary | Out-Null
    & hf @arguments
    if ($LASTEXITCODE -ne 0) { throw "hf download 失敗：$LASTEXITCODE" }
}

if (-not $DryRun) {
    foreach ($file in $entry.files) {
        $path = Join-Path $target $file.path
        if (-not (Test-Path -LiteralPath $path)) { throw "缺少模型檔：$path" }
        $item = Get-Item -LiteralPath $path
        if ($item.Length -ne [int64]$file.size) { throw "檔案大小不符：$path" }
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actualHash -ne $file.sha256) { throw "SHA-256 不符：$path" }
    }
    Write-Output "已完成 $Variant 固定 revision 與 SHA-256 驗證。"
}
