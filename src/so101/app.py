"""Interactive launcher for the SO-101 project.

One entry point for everything — no flags to remember. Run it and pick a mode:

    python -m so101

Each mode prompts for its options (backend, episode count, etc.) with sensible
defaults, then hands off to the same functions the individual `python -m so101.*`
commands use. Ctrl+C from any mode drops you back to this menu.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import REPO_ROOT, load_config

# ---------------------------------------------------------------- prompts ----


def _prompt(text: str, default: str | None = None) -> str | None:
    """Ask for a line of input. Returns None on EOF (e.g. piped/non-interactive)."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        s = input(f"{text}{suffix}: ").strip()
    except EOFError:
        return None
    return s or (default or "")


def _choice(title: str, options: list[str]) -> int | None:
    """Numbered menu. Returns the 0-based index, or None on EOF."""
    print(f"\n{title}")
    for i, label in enumerate(options, 1):
        print(f"  {i}) {label}")
    while True:
        s = _prompt("Select", "1")
        if s is None:
            return None
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s) - 1
        print("  Please enter a number from the list.")


def _yesno(text: str, default: bool = True) -> bool:
    s = _prompt(f"{text} ({'Y/n' if default else 'y/N'})", "")
    return default if not s else s.lower().startswith("y")


def _backend_is_sim() -> bool:
    return _choice("Backend:", ["Simulator (no hardware)", "Real arm"]) == 0


def _lerobot(name: str, *args: str) -> None:
    """Run a lerobot console entry point (prefer the venv's, fall back to PATH)."""
    exe = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    cmd = [str(exe if exe.exists() else name), *args]
    subprocess.run(cmd)


# ------------------------------------------------------------------ modes ----


def practice_mode() -> None:
    method = _choice("Input device:", ["Xbox controller", "Keyboard"])
    if method is None:
        return
    from .sim import practice
    practice.run(keyboard=(method == 1))


def record_mode() -> None:
    sim = _backend_is_sim()
    n = int(_prompt("Number of episodes", "10") or 10)
    default_repo = "local/so101_pick_place_sim" if sim else "local/so101_pick_place"
    repo = _prompt("Dataset id", default_repo)
    from .record import DEFAULT_TASK, record
    task = _prompt("Task description", DEFAULT_TASK)
    record(n, task or DEFAULT_TASK, repo, None, sim, display=True)


def teleop_mode() -> None:
    action = _choice("Real-arm teleop:",
                     ["Drive the arm", "Show controller axes/buttons (debug)"])
    if action is None:
        return
    from . import xbox_teleop
    if action == 1:
        xbox_teleop.debug_loop()
    else:
        xbox_teleop.teleop_loop(with_cameras=_yesno("Open cameras?", default=False))


def policy_mode() -> None:
    sim = _backend_is_sim()
    ckpt = _prompt("Checkpoint dir (…/pretrained_model)", "")
    if not ckpt:
        print("  A trained checkpoint is required — train one first.")
        return
    ds = _prompt("Training dataset id (for normalization stats)", "local/so101_pick_place_sim")
    from .run_policy import DEFAULT_TASK, run
    run(ckpt, ds, sim, DEFAULT_TASK, 30.0, display=True)


def train_mode() -> None:
    repo = _prompt("Dataset id", "local/so101_pick_place_sim")
    policy = _prompt("Policy type", "act")
    steps = _prompt("Training steps", "20000")
    device = _prompt("Device (cpu/cuda)", "cpu")
    root = REPO_ROOT / "data" / repo.replace("/", "__")
    out = REPO_ROOT / "outputs" / "train" / policy
    if device == "cpu":
        print("  Note: CPU training is slow; a CUDA GPU is strongly recommended.")
    _lerobot("lerobot-train",
             f"--dataset.repo_id={repo}", f"--dataset.root={root}",
             f"--policy.type={policy}", f"--policy.device={device}",
             f"--output_dir={out}", f"--steps={steps}")


def camera_mode() -> None:
    from . import cameras
    pick = _choice("Cameras:", ["List available indices", "Preview gripper", "Preview desk"])
    if pick == 0:
        cameras.list_cameras()
    elif pick == 1:
        cameras.preview("gripper")
    elif pick == 2:
        cameras.preview("desk")


def setup_motors_mode() -> None:
    cfg = load_config("robot")
    print(f"Assigning motor IDs on {cfg['port']}.")
    print("Connect ONE motor at a time to the controller board when prompted.")
    _lerobot("lerobot-setup-motors", "--robot.type=so101_follower",
             f"--robot.port={cfg['port']}", f"--robot.id={cfg['id']}")


def calibrate_mode() -> None:
    cfg = load_config("robot")
    print(f"Calibrating SO-101 follower on {cfg['port']} (id: {cfg['id']}).")
    _lerobot("lerobot-calibrate", "--robot.type=so101_follower",
             f"--robot.port={cfg['port']}", f"--robot.id={cfg['id']}")


def findport_mode() -> None:
    _lerobot("lerobot-find-port")


# ------------------------------------------------------------------- main ----

MODES = [
    ("Practice in the simulator", practice_mode),
    ("Record episodes (sim or real)", record_mode),
    ("Teleoperate the real arm", teleop_mode),
    ("Run a trained policy (sim or real)", policy_mode),
    ("Train a policy", train_mode),
    ("Cameras: list / preview", camera_mode),
    ("Find the arm's serial port", findport_mode),
    ("Assign motor IDs (first-time setup)", setup_motors_mode),
    ("Calibrate the arm", calibrate_mode),
]


def main() -> None:
    print("=" * 56)
    print("  SO-101 :: Xbox-teleoperated pick & place")
    print("=" * 56)
    labels = [m[0] for m in MODES] + ["Quit"]
    while True:
        idx = _choice("Choose a mode:", labels)
        if idx is None or idx == len(MODES):
            print("Bye.")
            return
        try:
            MODES[idx][1]()
        except KeyboardInterrupt:
            print("\n(interrupted — back to menu)")
        except Exception as exc:  # keep the menu alive on any mode error
            print(f"\n[error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
