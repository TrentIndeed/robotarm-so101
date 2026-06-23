# First-time setup

Step-by-step for getting the Xbox-teleoperated SO-101 follower recording episodes.

## 0. Prerequisites

- Windows 11, Python 3.10+ (`py --version`).
- SO-101 **follower** arm assembled and powered, USB serial adapter plugged in.
- Xbox controller connected (USB or the Xbox Wireless Adapter / Bluetooth). Confirm
  Windows sees it in *Settings → Bluetooth & devices*.
- Two USB cameras: one on the wrist/gripper, one fixed over the desk.

## 1. Environment

```powershell
cd C:\Users\Trenton\CodeProjects\so101
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .          # makes `python -m so101.*` importable
```

If PowerShell blocks the activate script:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. Find the arm's serial port

```powershell
python -m lerobot.find_port
```

Unplug the arm when prompted, plug it back in, and note the `COM` port it reports.
Put that value in [config/robot.yaml](../config/robot.yaml) under `port:`.

## 3. Calibrate the follower (one time)

```powershell
.\scripts\calibrate.ps1
```

Move each joint through its full range when prompted. Calibration is stored under the
`id` in `config/robot.yaml`; you only redo this if you re-cable or swap a servo.

## 4. Set up the cameras

```powershell
python -m so101.cameras --list                 # see which indices are live
python -m so101.cameras --preview gripper      # confirm the gripper view
python -m so101.cameras --preview desk         # confirm the desk view
```

Edit [config/cameras.yaml](../config/cameras.yaml) so `gripper` and `desk` point at the
right indices. Tip: cover one camera with your hand to tell them apart.

## 5. Map the Xbox controller

Controller axis/button numbers vary, so check yours:

```powershell
.\scripts\teleoperate.ps1 -Debug
```

Move each stick and press buttons; the live `axes=[...]` / `buttons_down=[...]` readout
tells you the index of each. Reconcile against [config/teleop.yaml](../config/teleop.yaml)
and fix any axis that's off. Defaults assume a standard XInput layout:

| Control          | Joint                         |
| ---------------- | ----------------------------- |
| Left stick X/Y   | shoulder_pan / shoulder_lift  |
| Right stick X/Y  | wrist_roll / elbow_flex       |
| Triggers (LT/RT) | wrist_flex down / up          |
| A / B            | gripper open / close          |
| Back/View        | emergency hold                |
| Start / X        | (recording) save / re-record  |

## 6. Drive it

```powershell
.\scripts\teleoperate.ps1
```

The arm seeds its targets from the current pose, so it won't jump. Tune `global_speed`,
per-joint `speed`, and `deadzone` in `config/teleop.yaml` until it feels controllable.
**Hold the Back/View button** any time as an emergency hold.

## 7. Record pick-and-place episodes

```powershell
.\scripts\record.ps1 -NumEpisodes 20
```

For each episode: drive the arm to pick up the object and place it, then press **Start**
to save and advance, or **X** to discard and retry. Data lands in
`data/local__so101_pick_place` (git-ignored).

Aim for variety — different object positions, lighting, and approach angles make a more
robust dataset.

## 8. Next: train a policy

Once you have a few dozen clean episodes, train a policy (e.g. ACT or a diffusion policy)
on the dataset with LeRobot's training entrypoint. That's the next milestone in the
[README](../README.md#status) checklist.
