[CmdletBinding()]
param(
    [string]$OutputPath = "config/inventory.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$absoluteOutput = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputPath))
if (-not $absoluteOutput.StartsWith($projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "清冊輸出路徑必須位於專案內"
}

$gpuRows = nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pcie.link.gen.gpucurrent,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit --format=csv,noheader,nounits
$gpus = foreach ($row in $gpuRows) {
    $fields = $row -split ",\s*"
    [ordered]@{
        index = [int]$fields[0]
        name = $fields[1]
        memory_total_mib = [int]$fields[2]
        memory_free_mib = [int]$fields[3]
        compute_capability = $fields[4]
        pcie_generation_current = [int]$fields[5]
        pcie_generation_max = [int]$fields[6]
        pcie_width_current = [int]$fields[7]
        pcie_width_max = [int]$fields[8]
        power_limit_watts = [double]$fields[9]
    }
}

$computer = Get-CimInstance Win32_ComputerSystem
$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$memory = Get-CimInstance Win32_PhysicalMemory
$drive = Get-Volume -DriveLetter D
$inventory = [ordered]@{
    captured_at = (Get-Date).ToString("o")
    project_root = $projectRoot
    constraints = [ordered]@{
        memory_speed_mts = 2133
        memory_speed_is_stability_limit = $true
        context_tokens = 262144
        minimum_decode_tps = 20.0
        maximum_quality_loss_exclusive = 0.05
    }
    cpu = [ordered]@{
        name = $processor.Name
        cores = $processor.NumberOfCores
        logical_processors = $processor.NumberOfLogicalProcessors
    }
    ram = [ordered]@{
        total_bytes = [int64]$computer.TotalPhysicalMemory
        configured_speed_mts = @($memory | ForEach-Object { [int]$_.ConfiguredClockSpeed })
    }
    gpus = @($gpus)
    storage = [ordered]@{
        drive = "D:"
        total_bytes = [int64]$drive.Size
        free_bytes = [int64]$drive.SizeRemaining
    }
}

$parent = Split-Path -Parent $absoluteOutput
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$inventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $absoluteOutput -Encoding utf8NoBOM
Write-Output $absoluteOutput
