# One-time calibration of the SO-101 follower.
# Reads the port + id from config/robot.yaml and runs LeRobot's calibration,
# which records each joint's range of motion. Re-run only if you re-cable the arm.
#
# Follow the on-screen prompts: move each joint through its full range, then center.

$ErrorActionPreference = "Stop"
$cfg = Get-Content "$PSScriptRoot\..\config\robot.yaml" -Raw
$port = ([regex]::Match($cfg, "(?m)^port:\s*(\S+)")).Groups[1].Value
$id   = ([regex]::Match($cfg, "(?m)^id:\s*(\S+)")).Groups[1].Value
$lerobot = "$PSScriptRoot\..\.venv\Scripts\lerobot-calibrate.exe"

Write-Host "Calibrating SO-101 follower on $port (id: $id)..."
& $lerobot --robot.type=so101_follower --robot.port=$port --robot.id=$id
