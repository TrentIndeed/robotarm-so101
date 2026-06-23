# Launch the 3D pick-and-place practice simulator.
#   .\scripts\practice.ps1               # drive with the Xbox controller
#   .\scripts\practice.ps1 -Keyboard     # no controller? drive with the keyboard
#   .\scripts\practice.ps1 -Seed 0       # repeatable block positions
param([switch]$Keyboard, [int]$Seed = -1)

$ErrorActionPreference = "Stop"
$cmd = @("-m", "so101.sim.practice")
if ($Keyboard)   { $cmd += "--keyboard" }
if ($Seed -ge 0) { $cmd += @("--seed", "$Seed") }

# Prefer the project venv if present, so it works without manual activation.
$py = if (Test-Path "$PSScriptRoot\..\.venv\Scripts\python.exe") {
    "$PSScriptRoot\..\.venv\Scripts\python.exe"
} else { "python" }

& $py @cmd
