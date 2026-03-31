param(
    [string]$PythonExe = "",
    [string]$RawDataRoot = ""
)

$ScriptPath = Join-Path $PSScriptRoot "prepare_windows_clone.ps1"
& $ScriptPath @PSBoundParameters
