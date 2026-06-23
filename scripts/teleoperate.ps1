# Drive the SO-101 follower with the Xbox controller (no recording).
# Pass -Debug to instead print live controller axis/button numbers.
param([switch]$Debug, [switch]$NoCameras)

$ErrorActionPreference = "Stop"
$args = @()
if ($Debug)      { $args += "--debug" }
if ($NoCameras)  { $args += "--no-cameras" }

python -m so101.xbox_teleop @args
