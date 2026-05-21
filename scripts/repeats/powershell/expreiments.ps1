$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $scriptDir "../../..")
$thisScript = $MyInvocation.MyCommand.Name

$scripts = Get-ChildItem -Path $scriptDir -Filter "*.ps1" -File |
    Where-Object { $_.Name -ne $thisScript } |
    Sort-Object Name

Push-Location $repoRoot
try {
    foreach ($script in $scripts) {
        Write-Host "Running $($script.Name)"
        Write-Host "-----------------------------------"

        & $script.FullName

        if ($LASTEXITCODE -ne 0) {
            throw "$($script.Name) failed with exit code $LASTEXITCODE"
        }

        Write-Host "Finished $($script.Name)"
        Write-Host "==================================="
    }
}
finally {
    Pop-Location
}
