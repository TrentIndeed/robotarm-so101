"""Graphical launcher for the SO-101 project (no terminal questions).

A Tkinter window with a button per mode and inline dropdowns / fields for each
mode's options. Clicking a Launch button spawns that mode in its own window
(MuJoCo viewer, camera previews) plus a console for its logs — the launcher stays
open so you can start another mode.

    python -m so101            # opens this GUI

Tkinter ships with the standard python.org build, so no extra dependencies.
The text-menu version is still available as ``python -m so101.app``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from . import CONFIG_DIR, REPO_ROOT, load_config

# Kept local so the GUI starts instantly (importing record would pull in mujoco).
DEFAULT_TASK = "Pick up the small object and place it at the target."

# Persisted GUI field values (machine-local, git-ignored).
SETTINGS_PATH = REPO_ROOT / ".app_settings.json"

PAD = {"padx": 8, "pady": 4}


def _console_kwargs() -> dict:
    # Give each launched mode its own console for logs / interactive prompts.
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_CONSOLE}
    return {}


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._saved = self._load_settings()   # values from last session
        self._vars: dict[str, tk.Variable] = {}
        root.title("SO-101 launcher")
        root.minsize(560, 0)

        ttk.Label(root, text="SO-101  ::  Xbox-teleoperated pick & place",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        ttk.Label(root, text="Pick a mode and click Launch. Each opens in its own window.",
                  foreground="#555").pack(anchor="w", padx=12, pady=(0, 8))

        self._practice_section()
        self._record_section()
        self._policy_section()
        self._train_section()
        self._hardware_section()
        self._camera_section()

        self.status = tk.StringVar(value="Ready.")
        ttk.Separator(root, orient="horizontal").pack(fill="x", pady=(8, 0))
        ttk.Label(root, textvariable=self.status, foreground="#0a6").pack(anchor="w", padx=12, pady=8)

    # -- settings persistence ------------------------------------------------
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        try:
            data = {k: v.get() for k, v in self._vars.items()}
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _var(self, key: str, default, cls=tk.StringVar):
        """A tk variable that restores its last value and autosaves on every change."""
        var = cls(value=self._saved.get(key, default))
        var.trace_add("write", lambda *_: self._save_settings())
        self._vars[key] = var
        return var

    # -- launching -----------------------------------------------------------
    def _launch_module(self, module: str, args: list[str], label: str) -> None:
        subprocess.Popen([sys.executable, "-m", module, *args], **_console_kwargs())
        self.status.set(f"Launched: {label}.  A new window opened.")

    def _launch_exe(self, name: str, args: list[str], label: str) -> None:
        exe = Path(sys.executable).parent / (f"{name}.exe" if os.name == "nt" else name)
        prog = str(exe) if exe.exists() else name
        subprocess.Popen([prog, *args], **_console_kwargs())
        self.status.set(f"Launched: {label}.  A new console opened.")

    # -- sections ------------------------------------------------------------
    def _group(self, title: str) -> ttk.LabelFrame:
        f = ttk.LabelFrame(self.root, text=title)
        f.pack(fill="x", padx=12, pady=5)
        return f

    def _practice_section(self) -> None:
        f = self._group("Practice in the simulator")
        self.practice_input = self._var("practice_input", "xbox")
        ttk.Label(f, text="Input:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Radiobutton(f, text="Xbox controller", variable=self.practice_input,
                        value="xbox").grid(row=0, column=1, sticky="w", **PAD)
        ttk.Radiobutton(f, text="Keyboard", variable=self.practice_input,
                        value="keyboard").grid(row=0, column=2, sticky="w", **PAD)
        ttk.Button(f, text="Launch practice", command=self._do_practice).grid(
            row=0, column=3, sticky="e", **PAD)

    def _do_practice(self) -> None:
        args = ["--keyboard"] if self.practice_input.get() == "keyboard" else []
        self._launch_module("so101.sim.practice", args, "practice sim")

    def _record_section(self) -> None:
        f = self._group("Record episodes")
        self.rec_backend = self._var("rec_backend", "sim")
        self.rec_eps = self._var("rec_eps", "10")
        self.rec_repo = self._var("rec_repo", "local/so101_pick_place_sim")
        self.rec_task = self._var("rec_task", DEFAULT_TASK)

        ttk.Label(f, text="Backend:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Radiobutton(f, text="Sim", variable=self.rec_backend, value="sim",
                        command=self._sync_rec_repo).grid(row=0, column=1, sticky="w", **PAD)
        ttk.Radiobutton(f, text="Real arm", variable=self.rec_backend, value="real",
                        command=self._sync_rec_repo).grid(row=0, column=2, sticky="w", **PAD)
        ttk.Label(f, text="Episodes:").grid(row=0, column=3, sticky="e", **PAD)
        ttk.Spinbox(f, from_=1, to=999, width=5, textvariable=self.rec_eps).grid(
            row=0, column=4, sticky="w", **PAD)

        ttk.Label(f, text="Dataset id:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.rec_repo, width=34).grid(
            row=1, column=1, columnspan=3, sticky="we", **PAD)
        ttk.Label(f, text="Task:").grid(row=2, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.rec_task, width=44).grid(
            row=2, column=1, columnspan=4, sticky="we", **PAD)
        ttk.Button(f, text="Launch record", command=self._do_record).grid(
            row=0, column=5, sticky="e", **PAD)

    def _sync_rec_repo(self) -> None:
        # Swap the default dataset id to match the backend (only if unchanged).
        known = {"local/so101_pick_place_sim", "local/so101_pick_place"}
        if self.rec_repo.get() in known:
            self.rec_repo.set("local/so101_pick_place_sim" if self.rec_backend.get() == "sim"
                              else "local/so101_pick_place")

    def _do_record(self) -> None:
        # The on-screen recording UI (start/stop buttons, counter, watch demos).
        args = ["--repo-id", self.rec_repo.get(), "--task", self.rec_task.get()]
        if self.rec_backend.get() == "sim":
            args.append("--sim")
        self._launch_module("so101.record_ui", args, "recording UI")

    def _policy_section(self) -> None:
        f = self._group("Run a trained policy")
        self.pol_backend = self._var("pol_backend", "sim")
        self.pol_ckpt = self._var("pol_ckpt", "")
        self.pol_ds = self._var("pol_ds", "local/so101_pick_place_sim")

        ttk.Label(f, text="Backend:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Radiobutton(f, text="Sim", variable=self.pol_backend, value="sim").grid(
            row=0, column=1, sticky="w", **PAD)
        ttk.Radiobutton(f, text="Real arm", variable=self.pol_backend, value="real").grid(
            row=0, column=2, sticky="w", **PAD)

        ttk.Label(f, text="Checkpoint:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.pol_ckpt, width=34).grid(
            row=1, column=1, columnspan=2, sticky="we", **PAD)
        ttk.Button(f, text="Browse…", command=self._browse_ckpt).grid(row=1, column=3, **PAD)

        ttk.Label(f, text="Dataset id:").grid(row=2, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.pol_ds, width=34).grid(
            row=2, column=1, columnspan=2, sticky="we", **PAD)
        ttk.Button(f, text="Launch policy", command=self._do_policy).grid(
            row=0, column=3, sticky="e", **PAD)

    def _browse_ckpt(self) -> None:
        d = filedialog.askdirectory(title="Select the pretrained_model directory",
                                    initialdir=str(REPO_ROOT / "outputs"))
        if d:
            self.pol_ckpt.set(d)

    def _do_policy(self) -> None:
        if not self.pol_ckpt.get():
            self.status.set("Pick a trained checkpoint first (Browse…).")
            return
        args = ["--checkpoint", self.pol_ckpt.get(), "--dataset", self.pol_ds.get()]
        if self.pol_backend.get() == "sim":
            args.append("--sim")
        self._launch_module("so101.run_policy", args, "run policy")

    def _train_section(self) -> None:
        f = self._group("Train a policy")
        self.tr_repo = self._var("tr_repo", "local/so101_pick_place_sim")
        self.tr_policy = self._var("tr_policy", "act")
        self.tr_steps = self._var("tr_steps", "20000")
        self.tr_device = self._var("tr_device", "cpu")

        ttk.Label(f, text="Dataset id:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.tr_repo, width=28).grid(row=0, column=1, sticky="we", **PAD)
        ttk.Label(f, text="Policy:").grid(row=0, column=2, sticky="e", **PAD)
        ttk.Combobox(f, textvariable=self.tr_policy, width=8,
                     values=["act", "diffusion", "smolvla"]).grid(row=0, column=3, **PAD)
        ttk.Label(f, text="Steps:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.tr_steps, width=10).grid(row=1, column=1, sticky="w", **PAD)
        ttk.Label(f, text="Device:").grid(row=1, column=2, sticky="e", **PAD)
        ttk.Combobox(f, textvariable=self.tr_device, width=8,
                     values=["cpu", "cuda"]).grid(row=1, column=3, **PAD)
        ttk.Button(f, text="Launch training", command=self._do_train).grid(
            row=0, column=4, rowspan=2, sticky="e", **PAD)

    def _do_train(self) -> None:
        repo = self.tr_repo.get()
        root = REPO_ROOT / "data" / repo.replace("/", "__")
        out = REPO_ROOT / "outputs" / "train" / self.tr_policy.get()
        self._launch_exe("lerobot-train", [
            f"--dataset.repo_id={repo}", f"--dataset.root={root}",
            f"--policy.type={self.tr_policy.get()}", f"--policy.device={self.tr_device.get()}",
            f"--output_dir={out}", f"--steps={self.tr_steps.get()}",
        ], "training")

    def _hardware_section(self) -> None:
        f = self._group("Real arm")
        self.hw_port = self._var("hw_port", load_config("robot")["port"])

        # First-time setup, left to right in the order you run them.
        ttk.Label(f, text="USB port:").grid(row=0, column=0, sticky="w", **PAD)
        ttk.Entry(f, textvariable=self.hw_port, width=10).grid(row=0, column=1, sticky="w", **PAD)
        ttk.Button(f, text="Find port",
                   command=lambda: self._launch_exe("lerobot-find-port", [], "find port")
                   ).grid(row=0, column=2, **PAD)
        ttk.Button(f, text="Save to config", command=self._save_port).grid(row=0, column=3, **PAD)

        ttk.Label(f, text="Setup:").grid(row=1, column=0, sticky="w", **PAD)
        ttk.Button(f, text="1. Assign motor IDs", command=self._do_setup_motors).grid(
            row=1, column=1, columnspan=2, sticky="we", **PAD)
        ttk.Button(f, text="2. Calibrate", command=self._do_calibrate).grid(row=1, column=3, **PAD)

        ttk.Separator(f, orient="horizontal").grid(row=2, column=0, columnspan=5, sticky="we", pady=4)

        self.teleop_cams = self._var("teleop_cams", False, tk.BooleanVar)
        ttk.Button(f, text="Teleoperate (pad)", command=self._do_teleop).grid(row=3, column=0, **PAD)
        ttk.Checkbutton(f, text="with cameras", variable=self.teleop_cams).grid(
            row=3, column=1, sticky="w", **PAD)
        ttk.Button(f, text="Controller debug",
                   command=lambda: self._launch_module("so101.xbox_teleop", ["--debug"], "controller debug")
                   ).grid(row=3, column=2, columnspan=2, **PAD)

        ttk.Button(f, text="Teleoperate + 3D view (cameras + sim)",
                   command=lambda: self._launch_module("so101.xbox_teleop", ["--mirror"], "teleop + 3D view")
                   ).grid(row=4, column=0, columnspan=4, sticky="we", **PAD)

    def _save_port(self) -> None:
        # Rewrite just the `port:` line in config/robot.yaml (preserves comments).
        path = CONFIG_DIR / "robot.yaml"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^(port:\s*).*$", lambda m: m.group(1) + self.hw_port.get(), text)
        path.write_text(text, encoding="utf-8")
        self.status.set(f"Saved port {self.hw_port.get()} to config/robot.yaml.")

    def _do_setup_motors(self) -> None:
        cfg = load_config("robot")
        self._launch_exe("lerobot-setup-motors", [
            "--robot.type=so101_follower", f"--robot.port={self.hw_port.get()}", f"--robot.id={cfg['id']}",
        ], "motor ID setup (follow the console prompts)")

    def _do_teleop(self) -> None:
        args = ["--cameras"] if self.teleop_cams.get() else []
        self._launch_module("so101.xbox_teleop", args, "teleop")

    def _do_calibrate(self) -> None:
        cfg = load_config("robot")
        self._launch_exe("lerobot-calibrate", [
            "--robot.type=so101_follower", f"--robot.port={self.hw_port.get()}", f"--robot.id={cfg['id']}",
        ], "calibration")

    def _camera_section(self) -> None:
        f = self._group("Cameras")
        ttk.Button(f, text="List indices",
                   command=lambda: self._launch_module("so101.cameras", ["--list"], "camera list")
                   ).grid(row=0, column=0, **PAD)
        ttk.Button(f, text="Preview gripper",
                   command=lambda: self._launch_module("so101.cameras", ["--preview", "gripper"], "gripper preview")
                   ).grid(row=0, column=1, **PAD)
        ttk.Button(f, text="Preview desk",
                   command=lambda: self._launch_module("so101.cameras", ["--preview", "desk"], "desk preview")
                   ).grid(row=0, column=2, **PAD)


def main() -> None:
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
