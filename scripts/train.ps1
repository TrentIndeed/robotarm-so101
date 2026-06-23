# Train a policy on a recorded dataset with LeRobot (PyTorch).
# Defaults to an ACT policy on the sim dataset. CPU training is slow — a CUDA GPU
# is strongly recommended for real runs; this machine's torch is CPU-only.
#   .\scripts\train.ps1 -RepoId local/so101_pick_place_sim -Steps 20000
param(
    [string]$RepoId = "local/so101_pick_place_sim",
    [string]$Policy = "act",
    [int]$Steps = 20000,
    [int]$BatchSize = 8,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$root = "$PSScriptRoot\..\data\" + ($RepoId -replace "/", "__")
if ($OutputDir -eq "") { $OutputDir = "$PSScriptRoot\..\outputs\train\$Policy" }
$device = "cpu"   # change to "cuda" if you have an NVIDIA GPU + CUDA torch

& "$PSScriptRoot\..\.venv\Scripts\lerobot-train.exe" `
    --dataset.repo_id=$RepoId `
    --dataset.root=$root `
    --policy.type=$Policy `
    --policy.device=$device `
    --output_dir=$OutputDir `
    --batch_size=$BatchSize `
    --steps=$Steps
