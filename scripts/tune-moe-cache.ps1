[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$PlanOnly,
    [ValidateSet("screen", "confirm", "all")]
    [string]$Stage = "screen",
    [ValidateRange(30, 3600)]
    [int]$CaseTimeoutSeconds = 900,
    [string[]]$Case = @(),
    [string]$OutputRoot = "",
    [string]$RuntimePath = ""
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = Join-Path $root "config/moe-cache-search.json"
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ($config.schema_version -ne 1) { throw "不支援的搜尋設定版本：$($config.schema_version)" }
if ($config.context_size -ne 262144) { throw "context 必須固定為 262144" }
if ([double]$config.minimum_decode_tps -lt 20.0) { throw "吞吐門檻不得低於 20 tok/s" }

function Resolve-ProjectPath([string]$Path) {
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $root $Path }
    $resolved = [IO.Path]::GetFullPath($candidate)
    $prefix = $root.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "路徑必須位於專案內：$resolved"
    }
    return $resolved
}

$bench = Resolve-ProjectPath $(if ($RuntimePath) { $RuntimePath } else { $config.runtime })
$model = Resolve-ProjectPath $config.model
$defaultOutputRoot = Resolve-ProjectPath "benchmarks/moe-cache-search/tuning-v1"
$resultRoot = Resolve-ProjectPath $(if ($OutputRoot) { $OutputRoot } else { $defaultOutputRoot })
$tempRoot = Resolve-ProjectPath "tmp/moe-cache-search"

function Merge-Arm([object]$Arm) {
    $values = [ordered]@{}
    foreach ($property in $config.base.PSObject.Properties) { $values[$property.Name] = $property.Value }
    foreach ($property in $Arm.PSObject.Properties) {
        if ($property.Name -notin @("name", "stage", "environment")) { $values[$property.Name] = $property.Value }
    }
    return $values
}

function Get-Arguments([System.Collections.IDictionary]$Values, [int]$Repetitions) {
    $arguments = @(
        "-m", $model,
        "--ctx-size", "$($config.context_size)",
        "-p", "0", "-n", "$($config.n_gen)",
        "--n-gen-warmup", "$($config.n_gen_warmup)",
        "-r", "$Repetitions", "-o", "json",
        "-ctk", "q4_0", "-ctv", "q4_0",
        "-ngl", "99", "-fa", "on",
        "-sm", "layer", "-ts", "$($Values.tensor_split)", "-mg", "$($Values.main_gpu)",
        "-t", "$($Values.threads)", "--poll", "$($Values.poll)",
        "-b", "$($Values.batch)", "-ub", "$($Values.ubatch)",
        "-ncmoe", "$($Values.n_cpu_moe)", "--moe-cache", "$($Values.moe_cache)",
        "-lm", "$($Values.load_mode)", "--progress"
    )
    if ($Values.repack -eq "on") {
        $arguments += @("--repack", "on")
    } elseif ($Values.repack -eq "off") {
        $arguments += "-nr"
    } else {
        throw "repack 必須是 on 或 off"
    }
    return $arguments
}

$selected = @($config.arms | Where-Object {
    ($Stage -eq "all" -or $_.stage -eq $Stage) -and
    ($Case.Count -eq 0 -or $_.name -in $Case)
})
if ($selected.Count -eq 0) { throw "沒有符合條件的搜尋案例" }

function Get-RuntimeIdentityHash([string]$Runtime) {
    if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) { return "missing" }
    $runtimeDirectory = Split-Path -Parent $Runtime
    $files = @(
        Get-Item -LiteralPath $Runtime
        Get-ChildItem -LiteralPath $runtimeDirectory -Filter "*.dll" -File |
            Sort-Object Name
    )
    $manifest = @($files | ForEach-Object {
        $name = $_.Name.ToLowerInvariant()
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$name`:$hash"
    }) -join "`n"
    $bytes = [Text.Encoding]::UTF8.GetBytes($manifest)
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}

$binaryHash = Get-RuntimeIdentityHash $bench
$cases = foreach ($arm in $selected) {
    $values = Merge-Arm $arm
    $repetitions = if ($arm.stage -eq "confirm") { [int]$config.confirm_repetitions } else { [int]$config.screen_repetitions }
    $arguments = Get-Arguments $values $repetitions
    $environment = [ordered]@{}
    if ($arm.environment) {
        foreach ($property in $arm.environment.PSObject.Properties) { $environment[$property.Name] = "$($property.Value)" }
    }
    $identity = [ordered]@{ name=$arm.name; arguments=$arguments; environment=$environment; binary_sha256=$binaryHash }
    $identityJson = $identity | ConvertTo-Json -Depth 8 -Compress
    $identityBytes = [Text.Encoding]::UTF8.GetBytes($identityJson)
    $caseId = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($identityBytes)).ToLowerInvariant()
    [ordered]@{
        name = $arm.name
        case_id = $caseId
        stage = $arm.stage
        context_size = [int]$config.context_size
        n_gen = [int]$config.n_gen
        n_gen_warmup = [int]$config.n_gen_warmup
        repetitions = $repetitions
        model = $model
        binary = $bench
        binary_sha256 = $binaryHash
        arguments = $arguments
        environment = $environment
        stdout = Join-Path $resultRoot "$($arm.name).json"
        stderr = Join-Path $resultRoot "$($arm.name).log"
        state = Join-Path $resultRoot "$($arm.name).state.json"
    }
}

if ($PlanOnly) {
    [ordered]@{ schema_version=1; output_root=$resultRoot; cases=@($cases) } |
        ConvertTo-Json -Depth 10
    exit 0
}
if (-not (Test-Path -LiteralPath $bench)) { throw "缺少 llama-bench：$bench" }
if (-not (Test-Path -LiteralPath $model)) { throw "缺少模型：$model" }
New-Item -ItemType Directory -Force -Path $resultRoot, $tempRoot | Out-Null

function Test-CaseComplete([System.Collections.IDictionary]$Item) {
    if (-not $Resume -or -not (Test-Path -LiteralPath $Item.state)) { return $false }
    try {
        $resultPath = $Item.stdout
        $state = Get-Content -Raw -LiteralPath $Item.state | ConvertFrom-Json
        if (-not $state.success -or $state.case_id -ne $Item.case_id) { return $false }
        if (-not (Test-Path -LiteralPath $resultPath)) { return $false }
        $data = @(Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json)
        $log = Get-Content -Raw -LiteralPath $Item.stderr
        return $data.Count -ge 1 -and $data[0].n_ctx -eq 262144 -and
            $log -match 'llama_context:\s+n_ctx\s+=\s+262144' -and
            $log -match '\[moe-cache\] enabled'
    } catch { return $false }
}

$environmentNames = @(
    "GGML_CUDA_MOE_CACHE_RESERVE_MB", "GGML_CUDA_MOE_CACHE_NDEV",
    "GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "GGML_CUDA_MOE_CACHE_ADMIT_AFTER",
    "GGML_CUDA_MOE_CACHE_THROTTLE", "GGML_CUDA_MOE_CACHE_DEDICATED_MMV",
    "GGML_CUDA_MOE_CACHE_INSERTS", "GGML_CUDA_MOE_CACHE_SERIAL_FILL",
    "GGML_CUDA_MOE_CACHE_STATS", "GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS",
    "TMP", "TEMP"
)

foreach ($item in $cases) {
    if (Test-CaseComplete $item) {
        Write-Output "SKIP $($item.name)：已完成且證據有效"
        continue
    }
    $snapshot = @{}
    foreach ($name in $environmentNames) {
        $snapshot[$name] = [ordered]@{
            exists = Test-Path -LiteralPath "Env:$name"
            value = [Environment]::GetEnvironmentVariable($name, "Process")
        }
    }
    $process = $null
    $timedOut = $false
    $started = [DateTimeOffset]::UtcNow
    try {
        $env:TMP = $tempRoot
        $env:TEMP = $tempRoot
        foreach ($name in $item.environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $item.environment[$name], "Process")
        }
        $process = Start-Process -FilePath $bench -ArgumentList $item.arguments -WorkingDirectory $root `
            -PassThru -WindowStyle Hidden -RedirectStandardOutput $item.stdout -RedirectStandardError $item.stderr
        $activePid = Join-Path $resultRoot "active.pid"
        $process.Id | Set-Content -LiteralPath $activePid -Encoding ascii
        if (-not $process.WaitForExit($CaseTimeoutSeconds * 1000)) {
            $timedOut = $true
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            $process.WaitForExit()
        }
        if ($timedOut) { throw "案例 $($item.name) 超時" }
        if ($process.ExitCode -ne 0) { throw "案例 $($item.name) 退出碼 $($process.ExitCode)" }

        $data = @(Get-Content -Raw -LiteralPath $item.stdout | ConvertFrom-Json)
        $log = Get-Content -Raw -LiteralPath $item.stderr
        if ($data.Count -ne 1 -or $data[0].n_ctx -ne 262144) { throw "JSON 未證明實際 256K context" }
        if ($log -notmatch 'llama_context:\s+n_ctx\s+=\s+262144') { throw "runtime log 未證明實際 256K context" }
        if ($log -notmatch '\[moe-cache\] enabled') { throw "MoE-cache 未啟用" }
        if ($log -match '(?:fill|dispatch|collect)-fail=[1-9][0-9]*') { throw "MoE-cache 發生 failure" }

        $state = [ordered]@{
            schema_version=1; success=$true; case_id=$item.case_id; name=$item.name
            started_utc=$started.ToString("O"); finished_utc=[DateTimeOffset]::UtcNow.ToString("O")
            exit_code=$process.ExitCode; timed_out=$false; binary_sha256=$item.binary_sha256
            arguments=$item.arguments; environment=$item.environment; result=$data[0]
            stdout=$item.stdout; stderr=$item.stderr
        }
        $stateTemp = "$($item.state).tmp"
        $state | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $stateTemp -Encoding utf8
        Move-Item -LiteralPath $stateTemp -Destination $item.state -Force
        Write-Output ("PASS {0}: {1:N3} tok/s" -f $item.name, [double]$data[0].avg_ts)
    } finally {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            $process.WaitForExit()
        }
        Remove-Item -LiteralPath (Join-Path $resultRoot "active.pid") -Force -ErrorAction SilentlyContinue
        foreach ($name in $environmentNames) {
            if ($snapshot[$name].exists) {
                [Environment]::SetEnvironmentVariable($name, $snapshot[$name].value, "Process")
            } else {
                [Environment]::SetEnvironmentVariable($name, $null, "Process")
            }
        }
    }
}
