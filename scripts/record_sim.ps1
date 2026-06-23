# Record pick-and-place episodes in the MuJoCo sim (Xbox controller, no hardware).
# Same dataset format as the real arm — drop --sim later to record for real.
#   .\scripts\record_sim.ps1 -NumEpisodes 20
param(
    [int]$NumEpisodes = 10,
    [string]$Task = "Pick up the small object and place it at the target.",
    [string]$RepoId = "local/so101_pick_place_sim"
)

$ErrorActionPreference = "Stop"
$py = "$PSScriptRoot\..\.venv\Scripts\python.exe"
& $py -m so101.record --sim --num-episodes $NumEpisodes --task "$Task" --repo-id "$RepoId"
