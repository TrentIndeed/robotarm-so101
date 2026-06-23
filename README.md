# SO-101 — Xbox-Teleoperated Pick & Place

A [LeRobot](https://github.com/huggingface/lerobot) project for the **SO-101 follower arm**,
teleoperated with an **Xbox controller** instead of a leader arm.

The goal is to collect demonstrations of a single manipulation task — **pick up a small
object and place it somewhere** — and (later) train a policy on them.

## Hardware

| Item                | Detail                                                        |
| ------------------- | ------------------------------------------------------------- |
| Arm                 | SO-101 **follower** (6-DOF, Feetech STS3215 servos)           |
| Teleop input        | **Xbox controller** (XInput, read via `pygame`)               |
| Camera 1 — *gripper*| Wrist/gripper-mounted camera for close-up grasping            |
| Camera 2 — *desk*   | Fixed overhead/side camera for the whole workspace            |

> There is **no leader arm**. All motion is commanded from the Xbox controller, which
> drives joint targets on the follower in real time.

## Task: pick & place

1. A small object sits somewhere on the desk (in view of the *desk* camera).
2. Operator drives the arm over the object using the *gripper* camera for the final approach.
3. Close the gripper, lift, move to the target location, release.

Each successful run is recorded as one **episode** (synced video from both cameras + joint
states + actions) into a LeRobot dataset for later policy training.

## One app, every mode

Everything is reachable from a single interactive menu — no flags to remember:

```powershell
.\scripts\so101.ps1        # or:  python -m so101
```

```
  1) Practice in the simulator        5) Train a policy
  2) Record episodes (sim or real)    6) Cameras: list / preview
  3) Teleoperate the real arm         7) Calibrate the arm
  4) Run a trained policy (sim/real)  8) Find the arm's serial port
```

Each mode prompts for its options (backend, episode count, …) with sensible defaults.
The individual `python -m so101.*` commands and `scripts/*.ps1` helpers still exist for
scripting; the app just front-ends them.

## Practice in simulation first

Before wiring up any motors or cameras, you can rehearse the whole task in a 3D
simulator driven by the **same** Xbox controller and the **same** control code:

```powershell
pip install -r requirements.txt
pip install -e .
.\scripts\so101.ps1        # → 1) Practice in the simulator
```

A MuJoCo window opens with the arm on a desk, a red block, and a green target pad.
Pick up the block and drop it on the pad — each success scores a point (shown on the
on-screen HUD) and respawns the block. Because the sim exposes the identical
`get_observation` / `send_action` interface as the real arm, your muscle memory (and
your `config/teleop.yaml` tuning) transfers straight to the hardware.

No controller yet? Drive with the keyboard instead:

```powershell
.\scripts\practice.ps1 -Keyboard
# A/D W/S I/K J/L T/G move joints, F toggles the gripper, R respawns the block
```

## Repo layout

```
so101/
├── config/
│   ├── robot.yaml      # follower serial port + arm id
│   ├── cameras.yaml    # gripper + desk camera indices / resolution / fps
│   └── teleop.yaml     # Xbox axis→joint mapping, speeds, deadzone
├── src/so101/
│   ├── app.py          # `python -m so101` — interactive menu for every mode
│   ├── controller.py   # Xbox controller → normalized joint commands (shared)
│   ├── robot.py        # make_robot(sim=…) — THE sim/real swap point
│   ├── xbox_teleop.py  # drive the real follower
│   ├── cameras.py      # build LeRobot camera configs from cameras.yaml
│   ├── record.py       # record episodes into a LeRobot dataset (real or --sim)
│   ├── run_policy.py   # run a trained policy on either backend (real or --sim)
│   └── sim/            # MuJoCo backend (so101.xml + cameras, sim_robot, practice)
├── scripts/            # PowerShell helpers (practice, calibrate, record, train, run_policy)
├── data/               # recorded datasets (git-ignored)
└── docs/SETUP.md       # step-by-step first-time setup
```

## Sim ↔ real: one flag

The MuJoCo sim and the real arm expose the **same** LeRobot interface
(`get_observation` / `send_action`, same normalized joints, same `gripper`/`desk`
camera keys). Everything goes through `make_robot(sim=…)`, so the whole pipeline is
backend-agnostic — `--sim` is the only thing that changes:

```powershell
# Collect demos in sim (no hardware), then train, then evaluate — all in sim:
.\scripts\record_sim.ps1 -NumEpisodes 30
.\scripts\train.ps1      -RepoId local/so101_pick_place_sim
.\scripts\run_policy.ps1 -Checkpoint outputs\train\act\checkpoints\last\pretrained_model `
                         -Dataset local/so101_pick_place_sim

# Later, on the physical arm: same commands, drop -Sim / add -Real.
```

Sim datasets are byte-for-byte the same format as real ones (parquet joint
states/actions + encoded MP4 videos for both cameras). The only thing the sim can't
replicate is the physics/appearance of reality (the sim-to-real gap) — a sim-trained
policy is a starting point, not a finished real-world policy.

## Quick start

See [docs/SETUP.md](docs/SETUP.md) for the full walkthrough. The short version:

```powershell
# 1. Create the environment and install deps (Python 3.12 recommended)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1           # Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

# 2. Find the arm's serial port, then calibrate it (one time)
.\.venv\Scripts\lerobot-find-port      # note the COM port → put it in config/robot.yaml
.\scripts\calibrate.ps1

# 3. Identify your cameras, fill in config/cameras.yaml
python -m so101.cameras --list

# 4. Drive the arm with the Xbox controller (no recording)
.\scripts\teleoperate.ps1

# 5. Record pick-and-place episodes
.\scripts\record.ps1
```

## Status

- [x] Repo scaffold
- [x] Practice simulator (MuJoCo)
- [x] Sim wired to LeRobot: cameras, dataset recording, policy-eval — all sim/real swappable
- [ ] Controller mapping rehearsed in sim
- [ ] First sim dataset recorded + policy trained
- [ ] Robot port + calibration confirmed
- [ ] Cameras configured and previewed
- [ ] Xbox teleop tuned (speeds / deadzone / mapping)
- [ ] First pick-and-place dataset recorded
- [ ] Policy trained
