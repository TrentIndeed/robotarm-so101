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

## Repo layout

```
so101/
├── config/
│   ├── robot.yaml      # follower serial port + arm id
│   ├── cameras.yaml    # gripper + desk camera indices / resolution / fps
│   └── teleop.yaml     # Xbox axis→joint mapping, speeds, deadzone
├── src/so101/
│   ├── xbox_teleop.py  # read Xbox controller → command follower joints
│   ├── cameras.py      # build LeRobot camera configs from cameras.yaml
│   └── record.py       # record pick-and-place episodes into a dataset
├── scripts/            # Windows PowerShell helpers (find port, calibrate, run)
├── data/               # recorded datasets (git-ignored)
└── docs/SETUP.md       # step-by-step first-time setup
```

## Quick start

See [docs/SETUP.md](docs/SETUP.md) for the full walkthrough. The short version:

```powershell
# 1. Create the environment and install deps
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Find the arm's serial port, then calibrate it (one time)
python -m lerobot.find_port            # note the COM port → put it in config/robot.yaml
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
- [ ] Robot port + calibration confirmed
- [ ] Cameras configured and previewed
- [ ] Xbox teleop tuned (speeds / deadzone / mapping)
- [ ] First pick-and-place dataset recorded
- [ ] Policy trained
