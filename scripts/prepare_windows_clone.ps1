param(
    [string]$PythonExe = "",
    [string]$RawDataRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

if ($RawDataRoot) {
    $env:GRENET_RAWDATA_ROOT = $RawDataRoot
}

$PythonCommand = @()
$VenvPython = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"

if ($PythonExe) {
    $PythonCommand = @($PythonExe)
} elseif (Test-Path $VenvPython) {
    $PythonCommand = @($VenvPython)
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = @("py", "-3.12")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = @("python")
} else {
    throw "Python executable not found. Create .venv312 or pass -PythonExe explicitly."
}

function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments = @()
    )

    $LauncherArgs = @()
    if ($PythonCommand.Length -gt 1) {
        $LauncherArgs = $PythonCommand[1..($PythonCommand.Length - 1)]
    }

    & $PythonCommand[0] @LauncherArgs $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python script failed with exit code $LASTEXITCODE: $ScriptPath"
    }
}

$GenerateFoldsScript = Join-Path $PSScriptRoot "generate_folds.py"
$PatchNotebooksScript = Join-Path $PSScriptRoot "patch_notebooks_for_local.py"

Invoke-PythonScript -ScriptPath $GenerateFoldsScript -Arguments @("--setup-layout")
Invoke-PythonScript -ScriptPath $PatchNotebooksScript

Write-Host "Repository prepared for a cloned Windows environment."
Write-Host "  Project root : $ProjectRoot"
if ($env:GRENET_RAWDATA_ROOT) {
    Write-Host "  Raw data root: $env:GRENET_RAWDATA_ROOT"
} else {
    Write-Host "  Raw data root: $ProjectRoot\data\raw"
}
Write-Host "  Folds file   : $ProjectRoot\data\processed\folds.pkl"
Write-Host "  Notebooks    : patched to use repo-relative paths"
