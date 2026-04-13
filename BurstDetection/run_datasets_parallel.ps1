# Launch one main.py process per job in a separate console (parallel runs).
#
# Mode A — indices (0-based rows in config.py SpikeTime_Mat_File):
#   .\run_datasets_parallel.ps1 -CondaEnv bursts -Indices 0,1,2
#
# Mode B — substring bundles (one window per bundle; survives reordering config):
#   .\run_datasets_parallel.ps1 -CondaEnv bursts -Kind kilosortset -ContainsJobs @(
#       'm361,imec0',
#       'm361,imec1',
#       'm361,imec2'
#   )
# Each ContainsJobs string is comma-separated fragments (AND), passed as multiple
#   --only-dataset-contains flags. Use -Kind kilosort or kilosortset so short tokens
#   like imec0 do not match .mat lines or the wrong Kilosort mode.
#
# Omit -CondaEnv if python is already on PATH.

param(
    [string]$CondaEnv = "",
    [int[]]$Indices = @(),
    [string[]]$ContainsJobs = @(),
    [ValidateSet("", "kilosort", "kilosortset")]
    [string]$Kind = ""
)

$burstDir = $PSScriptRoot

if ($Indices.Count -gt 0 -and $ContainsJobs.Count -gt 0) {
    Write-Error "Use either -Indices or -ContainsJobs, not both."
    exit 1
}
if ($Indices.Count -eq 0 -and $ContainsJobs.Count -eq 0) {
    Write-Error "Provide -Indices (e.g. 0,1,2) or -ContainsJobs (e.g. @('m361,imec0','m361,imec1'))."
    exit 1
}

if ($Indices.Count -gt 0) {
    foreach ($i in $Indices) {
        $parts = @("main.py", "--only-datasets", "$i")
        if ($Kind) {
            $parts = @("main.py", "--only-dataset-kind", $Kind, "--only-datasets", "$i")
        }
        $argLine = ($parts | ForEach-Object {
                if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '`"') + '"' } else { $_ }
            }) -join " "
        if ($CondaEnv) {
            $inner = "Set-Location -LiteralPath '$burstDir'; conda run -n '$CondaEnv' python $argLine"
        }
        else {
            $inner = "Set-Location -LiteralPath '$burstDir'; python $argLine"
        }
        Start-Process powershell -WorkingDirectory $burstDir -ArgumentList @("-NoExit", "-Command", $inner)
    }
    Write-Host "Started $($Indices.Count) window(s) (--only-datasets per index)."
}
else {
    foreach ($job in $ContainsJobs) {
        $n++
        $frags = $job -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        if ($frags.Count -eq 0) {
            Write-Warning "Skipping empty ContainsJobs entry: $job"
            continue
        }
        $parts = @("main.py")
        if ($Kind) {
            $parts += "--only-dataset-kind"
            $parts += $Kind
        }
        foreach ($f in $frags) {
            $parts += "--only-dataset-contains"
            $parts += $f
        }
        $argLine = ($parts | ForEach-Object {
                if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '`"') + '"' } else { $_ }
            }) -join " "
        if ($CondaEnv) {
            $inner = "Set-Location -LiteralPath '$burstDir'; conda run -n '$CondaEnv' python $argLine"
        }
        else {
            $inner = "Set-Location -LiteralPath '$burstDir'; python $argLine"
        }
        Start-Process powershell -WorkingDirectory $burstDir -ArgumentList @("-NoExit", "-Command", $inner)
    }
    Write-Host "Started $($ContainsJobs.Count) window(s) (--only-dataset-contains bundles)."
}
