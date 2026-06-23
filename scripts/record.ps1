# Record pick-and-place episodes with the Xbox controller.
#   .\scripts\record.ps1                 # 10 episodes, default task
#   .\scripts\record.ps1 -NumEpisodes 30
param(
    [int]$NumEpisodes = 10,
    [string]$Task = "Pick up the small object and place it at the target.",
    [string]$RepoId = "local/so101_pick_place"
)

$ErrorActionPreference = "Stop"
python -m so101.record `
    --num-episodes $NumEpisodes `
    --task "$Task" `
    --repo-id "$RepoId"
