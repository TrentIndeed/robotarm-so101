"""On-screen recording UI for pick-and-place demos.

A single window with: the live gripper + desk camera feeds, big Start/Stop and
Discard buttons, a live "saved episodes" counter, and an episode list you can
double-click to watch a recorded demo. The Xbox controller's Start/X buttons do the
same actions AND update the on-screen buttons, so the two stay in sync.

    python -m so101.record_ui                 # real arm
    python -m so101.record_ui --sim           # MuJoCo sim, no hardware

Everything (robot serial I/O, controller, Tkinter) runs on one thread via Tk's
after() loop, so it's safe with the single-threaded Feetech bus. Drive with the
controller exactly as in teleop; press Start to begin a take, Start again (or the
on-screen button) to stop + save, X to discard.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from . import REPO_ROOT
from .controller import XboxTeleopController
from .record import DEFAULT_TASK, _reset_sim_block
from .robot import make_robot

CAM_W, CAM_H = 400, 300


class RecorderUI:
    def __init__(self, root, task, repo_id, sim):
        from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR

        self.root = root
        self.task = task
        self.sim = sim
        self._build_frame = build_dataset_frame
        self._OBS, self._ACTION = OBS_STR, ACTION

        self.robot = make_robot(sim=sim, use_cameras=True)
        self.ctrl = XboxTeleopController()
        self.btn = self.ctrl.cfg["buttons"]
        self.fps = self.ctrl.cfg["control_hz"]
        self.cam_names = [k for k, v in self.robot.observation_features.items() if isinstance(v, tuple)]
        self.robot.connect()
        self.ctrl.connect()

        features = {
            **hw_to_dataset_features(self.robot.observation_features, OBS_STR, use_video=True),
            **hw_to_dataset_features(self.robot.action_features, ACTION),
        }
        self.root_dir = REPO_ROOT / "data" / repo_id.replace("/", "__")
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=self.fps, features=features, root=self.root_dir,
            robot_type=self.robot.name, use_videos=True)

        self.recording = False
        self.episodes = 0
        self.rng = random.Random()
        self.n_steps = max(1, round((1.0 / self.fps) / self.robot.model.opt.timestep)) if sim else 0
        self._prev_btn: dict[int, bool] = {}
        self._imgs: dict[str, ImageTk.PhotoImage] = {}

        self._build_ui(repo_id)
        if sim:
            _reset_sim_block(self.robot, self.rng)
        self.ctrl.seed_targets(self.robot.get_observation())
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(0, self._tick)

    # -- UI ------------------------------------------------------------------
    def _build_ui(self, repo_id):
        self.root.title(f"SO-101 — Record  ({'SIM' if self.sim else 'REAL'})")
        ttk.Label(self.root, text=f"Dataset: {repo_id}", foreground="#555").pack(anchor="w", padx=10, pady=(8, 0))
        ttk.Label(self.root, text=f"Task: {self.task}", foreground="#555").pack(anchor="w", padx=10)

        cams = ttk.Frame(self.root)
        cams.pack(padx=10, pady=8)
        self.cam_labels = {}
        for i, name in enumerate(self.cam_names):
            col = ttk.Frame(cams)
            col.grid(row=0, column=i, padx=6)
            ttk.Label(col, text=name).pack()
            lbl = ttk.Label(col)
            lbl.pack()
            self.cam_labels[name] = lbl

        self.status = tk.StringVar(value="idle")
        self.status_lbl = tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 16, "bold"),
                                   fg="#0a0")
        self.status_lbl.pack(pady=(0, 4))

        controls = ttk.Frame(self.root)
        controls.pack(pady=4)
        self.start_btn = tk.Button(controls, text="Start recording", width=16, height=2,
                                   bg="#1a7", fg="white", font=("Segoe UI", 11, "bold"),
                                   command=self._toggle_record)
        self.start_btn.grid(row=0, column=0, padx=6)
        self.discard_btn = tk.Button(controls, text="Discard take", width=14, height=2,
                                     command=self._discard, state="disabled")
        self.discard_btn.grid(row=0, column=1, padx=6)

        self.count = tk.StringVar(value="Saved episodes: 0")
        ttk.Label(self.root, textvariable=self.count, font=("Segoe UI", 12)).pack(pady=2)

        watch = ttk.LabelFrame(self.root, text="Watch demos")
        watch.pack(fill="both", expand=True, padx=10, pady=8)
        self.ep_list = tk.Listbox(watch, height=6)
        self.ep_list.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.ep_list.bind("<Double-Button-1>", lambda _e: self._watch_selected())
        side = ttk.Frame(watch)
        side.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(side, text="Watch selected", command=self._watch_selected).pack(fill="x", pady=2)
        ttk.Label(side, text="(opens the\nrerun viewer)", foreground="#777").pack()

        ttk.Label(self.root, text="Controller: Start = begin/stop+save, X = discard.",
                  foreground="#777").pack(anchor="w", padx=10, pady=(0, 8))

    # -- control loop --------------------------------------------------------
    def _pressed(self, b: int) -> bool:
        now = bool(self.ctrl.joystick.get_button(b))
        was = self._prev_btn.get(b, False)
        self._prev_btn[b] = now
        return now and not was

    def _tick(self):
        if not self.root.winfo_exists():
            return
        obs = self.robot.get_observation()
        action = self.ctrl.compute_action()
        self.robot.send_action(action)
        if self.sim:
            self.robot.step(self.n_steps)

        # Controller buttons mirror the on-screen buttons.
        if self._pressed(self.btn["episode_done"]):
            self._toggle_record()
        elif self.recording and self._pressed(self.btn["episode_cancel"]):
            self._discard()

        if self.recording:
            of = self._build_frame(self.dataset.features, obs, prefix=self._OBS)
            af = self._build_frame(self.dataset.features, action, prefix=self._ACTION)
            self.dataset.add_frame({**of, **af, "task": self.task})

        self._show_cameras(obs)
        self.root.after(int(1000 / self.fps), self._tick)

    def _show_cameras(self, obs):
        for name, lbl in self.cam_labels.items():
            if name in obs:
                im = Image.fromarray(obs[name]).resize((CAM_W, CAM_H))
                photo = ImageTk.PhotoImage(im)
                self._imgs[name] = photo          # keep a reference
                lbl.configure(image=photo)

    # -- actions (shared by buttons + controller) ----------------------------
    def _toggle_record(self):
        if not self.recording:
            if self.sim:
                _reset_sim_block(self.robot, self.rng)
            self.recording = True
            self.start_btn.configure(text="Stop & Save", bg="#c33")
            self.discard_btn.configure(state="normal")
            self.status.set("RECORDING")
            self.status_lbl.configure(fg="#c00")
        else:
            self.dataset.save_episode(parallel_encoding=False)
            self.episodes += 1
            self.recording = False
            self.start_btn.configure(text="Start recording", bg="#1a7")
            self.discard_btn.configure(state="disabled")
            self.status.set("idle")
            self.status_lbl.configure(fg="#0a0")
            self.count.set(f"Saved episodes: {self.episodes}")
            self.ep_list.insert("end", f"Episode {self.episodes - 1}")

    def _discard(self):
        if not self.recording:
            return
        self.dataset.clear_episode_buffer()
        self.recording = False
        self.start_btn.configure(text="Start recording", bg="#1a7")
        self.discard_btn.configure(state="disabled")
        self.status.set("idle (discarded)")
        self.status_lbl.configure(fg="#0a0")

    def _watch_selected(self):
        sel = self.ep_list.curselection()
        if not sel:
            return
        idx = sel[0]
        exe = "lerobot-dataset-viz" + (".exe" if os.name == "nt" else "")
        prog = os.path.join(os.path.dirname(sys.executable), exe)
        kwargs = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if os.name == "nt" else {}
        subprocess.Popen([prog, "--repo-id", self.dataset.repo_id,
                          "--root", str(self.root_dir), "--episode-index", str(idx)], **kwargs)

    def _on_close(self):
        try:
            if self.recording:
                self.dataset.clear_episode_buffer()
            self.dataset.finalize()
        finally:
            self.ctrl.disconnect()
            self.robot.disconnect()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="On-screen recording UI for SO-101 demos")
    parser.add_argument("--sim", action="store_true", help="record from the MuJoCo sim")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--repo-id", default="local/so101_pick_place")
    args = parser.parse_args()

    root = tk.Tk()
    RecorderUI(root, args.task, args.repo_id, args.sim)
    root.mainloop()


if __name__ == "__main__":
    main()
