# Run a trained policy on the SO-101. Defaults to the sim; add -Real for hardware.
#   .\scripts\run_policy.ps1 -Checkpoint outputs\train\my_run\checkpoints\last\pretrained_model `
#                            -Dataset local/so101_pick_place_sim
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Dataset,
    [switch]$Real
)

$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$flags = @("-m", "so101.run_policy", "--checkpoint", $Checkpoint, "--dataset", $Dataset)
if (-not $Real) { $flags += "--sim" }
& $py @flags
