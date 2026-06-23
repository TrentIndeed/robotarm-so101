# Launch the 3D pick-and-place practice simulator (Xbox controller required).
# Pass -Seed for repeatable block positions:  .\scripts\practice.ps1 -Seed 0
param([int]$Seed = -1)

$ErrorActionPreference = "Stop"
if ($Seed -ge 0) {
    python -m so101.sim.practice --seed $Seed
} else {
    python -m so101.sim.practice
}
