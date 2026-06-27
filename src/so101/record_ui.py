"""The SO-101 app — one window: live cameras, demo recording, settings in the menu.

Opens with your saved settings (shared with the launcher's .app_settings.json), connects
to the chosen backend, shows the gripper + desk feeds, and lets you record / watch demos.
Everything day-to-day lives here; the menu bar holds settings (backend, dataset, task) and
tools. Drive with the Xbox controller; Start = begin a take, Start again (or the on-screen
button) = stop + save, X = discard. The on-screen buttons stay in sync with the controller.

    python -m so101            # this app (via ./run)
    python -m so101.record_ui  # same, directly

Robot serial I/O + controller + Tk all run on one thread via Tk's after() loop, safe with
the single-threaded Feetech bus. Heavy/extra tools (3D mirror, training, dataset replay)
necessarily open their own windows — Tk can't host a MuJoCo/rerun/console view.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from . import REPO_ROOT, load_config
from .controller import XboxTeleopController
from .record import DEFAULT_TASK, _reset_sim_block
from .robot import make_robot

SETTINGS_PATH = REPO_ROOT / ".app_settings.json"
CAM_W, CAM_H = 400, 300


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(updates: dict) -> None:
    try:
        data = _load_settings()
        data.update(updates)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        s = _load_settings()
        self.sim = s.get("rec_backend", "sim") == "sim"
        self.repo_id = s.get("rec_repo", "local/so101_pick_place")
        self.task = s.get("rec_task", DEFAULT_TASK)

        self.robot = self.ctrl = self.dataset = None
        self.connected = False
        self.recording = False
        self.episodes = 0
        self.rng = random.Random()
        self.btn: dict = {}
        self.fps = 30
        self._prev_btn: dict[int, bool] = {}
        self._imgs: dict[str, ImageTk.PhotoImage] = {}
        self._gen = 0          # tick-loop generation (so reconnects don't double-run)
        self._frame_n = 0

        root.title("SO-101")
        self._build_menu()
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_status("Connecting…", "#a60")
        root.after(200, self._connect)

    # -- menu bar (settings + tools) -----------------------------------------
    def _build_menu(self):
        bar = tk.Menu(self.root)
        self.root.config(menu=bar)

        filem = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="File", menu=filem)
        filem.add_command(label="Quit", command=self._on_close)

        setm = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="Settings", menu=setm)
        self._backend_var = tk.StringVar(value="sim" if self.sim else "real")
        setm.add_radiobutton(label="Backend: Simulator (no hardware)", variable=self._backend_var,
                             value="sim", command=self._change_backend)
        setm.add_radiobutton(label="Backend: Real arm", variable=self._backend_var,
                             value="real", command=self._change_backend)
        setm.add_separator()
        setm.add_command(label="Dataset id…", command=self._change_dataset)
        setm.add_command(label="Task description…", command=self._change_task)
        setm.add_separator()
        setm.add_command(label="Reconnect", command=self._reconnect)

        toolm = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="Tools", menu=toolm)
        toolm.add_command(label="3D mirror view (separate window)",
                          command=lambda: self._launch_module("so101.xbox_teleop", ["--mirror"]))
        toolm.add_command(label="Controller axis/button check",
                          command=lambda: self._launch_module("so101.xbox_teleop", ["--debug"]))
        toolm.add_separator()
        toolm.add_command(label="Find USB port", command=lambda: self._launch_exe("lerobot-find-port", []))
        toolm.add_command(label="Assign motor IDs", command=self._setup_motors)
        toolm.add_command(label="Calibrate arm", command=self._calibrate)
        toolm.add_separator()
        toolm.add_command(label="Train a policy on this dataset…", command=self._train)

        helpm = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="Help", menu=helpm)
        helpm.add_command(label="About", command=lambda: messagebox.showinfo(
            "SO-101", "SO-101 pick-and-place control.\nSettings + tools are in the menu bar."))

    # -- main view -----------------------------------------------------------
    def _build_ui(self):
        self.header = tk.StringVar()
        ttk.Label(self.root, textvariable=self.header, foreground="#555").pack(anchor="w", padx=10, pady=(8, 0))
        self._refresh_header()

        self.cam_box = ttk.Frame(self.root)
        self.cam_box.pack(padx=10, pady=8)
        self.cam_labels: dict = {}
        self._cam_msg = ttk.Label(self.cam_box, text="Connecting…", font=("Segoe UI", 12))
        self._cam_msg.pack(padx=80, pady=60)

        self.status = tk.StringVar(value="…")
        self.status_lbl = tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 16, "bold"))
        self.status_lbl.pack(pady=(0, 4))

        controls = ttk.Frame(self.root)
        controls.pack(pady=4)
        self.start_btn = tk.Button(controls, text="Start recording", width=16, height=2,
                                   bg="#1a7", fg="white", font=("Segoe UI", 11, "bold"),
                                   command=self._toggle_record, state="disabled")
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

        ttk.Label(self.root, text="Controller: Start = begin/stop+save a take, X = discard.",
                  foreground="#777").pack(anchor="w", padx=10, pady=(0, 8))

    def _refresh_header(self):
        self.header.set(f"Backend: {'SIM' if self.sim else 'REAL'}    Dataset: {self.repo_id}"
                        f"    Task: {self.task}")

    def _set_status(self, text, color):
        self.status.set(text)
        self.status_lbl.configure(fg=color)

    # -- connection lifecycle ------------------------------------------------
    def _connect(self):
        try:
            from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.utils.constants import ACTION, OBS_STR

            self._build_frame, self._OBS, self._ACTION = build_dataset_frame, OBS_STR, ACTION
            self.robot = make_robot(sim=self.sim, use_cameras=True)
            self.robot.connect()
            try:
                self.ctrl = XboxTeleopController()
                self.ctrl.connect()
                self.btn = self.ctrl.cfg["buttons"]
                self.fps = self.ctrl.cfg["control_hz"]
            except Exception as exc:
                self.ctrl = None
                self._set_status(f"Cameras only — no controller ({exc})", "#a60")

            self.cam_names = [k for k, v in self.robot.observation_features.items() if isinstance(v, tuple)]
            self.n_steps = max(1, round((1.0 / self.fps) / self.robot.model.opt.timestep)) if self.sim else 0
            self.root_dir = REPO_ROOT / "data" / self.repo_id.replace("/", "__")
            if self.root_dir.exists():
                self.dataset = LeRobotDataset.resume(repo_id=self.repo_id, root=self.root_dir)
                self.episodes = self.dataset.num_episodes
            else:
                features = {
                    **hw_to_dataset_features(self.robot.observation_features, OBS_STR, use_video=True),
                    **hw_to_dataset_features(self.robot.action_features, ACTION),
                }
                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id, fps=self.fps, features=features, root=self.root_dir,
                    robot_type=self.robot.name, use_videos=True)
                self.episodes = 0

            self._populate_cameras()
            self._refresh_watch()
            if self.sim:
                _reset_sim_block(self.robot, self.rng)
            if self.ctrl:
                self.ctrl.seed_targets(self.robot.get_observation())
                self._set_status("idle", "#0a0")
            self.connected = True
            self._update_controls()
            self._gen += 1
            self.root.after(0, lambda g=self._gen: self._tick(g))
        except Exception as exc:
            self.connected = False
            self._set_status(f"Connection failed: {exc}", "#c00")
            self._cam_msg.configure(text="Not connected. Fix the arm / pick Settings → Backend,\n"
                                         "then Settings → Reconnect.")

    def _disconnect(self):
        self.connected = False
        if self.dataset is not None:
            try:
                if self.recording:
                    self.dataset.clear_episode_buffer()
                self.dataset.finalize()
            except Exception:
                pass
        for obj in (self.ctrl, self.robot):
            try:
                if obj is not None:
                    obj.disconnect()
            except Exception:
                pass
        self.robot = self.ctrl = self.dataset = None
        self.recording = False

    def _reconnect(self):
        self._disconnect()
        self._refresh_header()
        self._set_status("Connecting…", "#a60")
        self.root.after(200, self._connect)

    def _populate_cameras(self):
        for w in self.cam_box.winfo_children():
            w.destroy()
        self.cam_labels = {}
        for i, name in enumerate(self.cam_names):
            col = ttk.Frame(self.cam_box)
            col.grid(row=0, column=i, padx=6)
            ttk.Label(col, text=name).pack()
            lbl = ttk.Label(col)
            lbl.pack()
            self.cam_labels[name] = lbl

    def _update_controls(self):
        ok = self.connected and self.ctrl is not None
        self.start_btn.configure(state="normal" if ok else "disabled")

    # -- control loop --------------------------------------------------------
    def _pressed(self, b: int) -> bool:
        now = bool(self.ctrl.joystick.get_button(b))
        was = self._prev_btn.get(b, False)
        self._prev_btn[b] = now
        return now and not was

    def _tick(self, gen):
        if gen != self._gen or not self.connected or not self.root.winfo_exists():
            return
        t0 = time.perf_counter()
        obs = self.robot.get_observation()
        if self.ctrl is not None:
            action = self.ctrl.compute_action()
            self.robot.send_action(action)
            if self.sim:
                self.robot.step(self.n_steps)
            if self._pressed(self.btn["episode_done"]):
                self._toggle_record()
            elif self.recording and self._pressed(self.btn["episode_cancel"]):
                self._discard()
            if self.recording:
                of = self._build_frame(self.dataset.features, obs, prefix=self._OBS)
                af = self._build_frame(self.dataset.features, action, prefix=self._ACTION)
                self.dataset.add_frame({**of, **af, "task": self.task})

        self._show_cameras(obs)
        # Schedule the next frame accounting for the work just done, so the display
        # actually runs near `fps` instead of fps minus the per-frame work time.
        delay = max(1, int(1000 / self.fps - (time.perf_counter() - t0) * 1000))
        self.root.after(delay, lambda: self._tick(gen))

    def _show_cameras(self, obs):
        for name, lbl in self.cam_labels.items():
            if name in obs:
                photo = ImageTk.PhotoImage(Image.fromarray(obs[name]).resize((CAM_W, CAM_H)))
                self._imgs[name] = photo
                lbl.configure(image=photo)

    # -- recording actions (shared by buttons + controller) ------------------
    def _toggle_record(self):
        if not self.connected or self.ctrl is None:
            return
        if not self.recording:
            if self.sim:
                _reset_sim_block(self.robot, self.rng)
            self.recording = True
            self.start_btn.configure(text="Stop & Save", bg="#c33")
            self.discard_btn.configure(state="normal")
            self._set_status("RECORDING", "#c00")
        else:
            self.dataset.save_episode(parallel_encoding=False)
            self.episodes += 1
            self.recording = False
            self.start_btn.configure(text="Start recording", bg="#1a7")
            self.discard_btn.configure(state="disabled")
            self._set_status("idle", "#0a0")
            self.count.set(f"Saved episodes: {self.episodes}")
            self.ep_list.insert("end", f"Episode {self.episodes - 1}")

    def _discard(self):
        if not self.recording:
            return
        self.dataset.clear_episode_buffer()
        self.recording = False
        self.start_btn.configure(text="Start recording", bg="#1a7")
        self.discard_btn.configure(state="disabled")
        self._set_status("idle (discarded)", "#0a0")

    def _refresh_watch(self):
        self.ep_list.delete(0, "end")
        for i in range(self.episodes):
            self.ep_list.insert("end", f"Episode {i}")
        self.count.set(f"Saved episodes: {self.episodes}")

    def _watch_selected(self):
        sel = self.ep_list.curselection()
        if not sel or self.dataset is None:
            return
        self._launch_exe("lerobot-dataset-viz", [
            "--repo-id", self.dataset.repo_id, "--root", str(self.root_dir),
            "--episode-index", str(sel[0])])

    # -- settings handlers ---------------------------------------------------
    def _change_backend(self):
        self.sim = self._backend_var.get() == "sim"
        _save_settings({"rec_backend": "sim" if self.sim else "real"})
        self._reconnect()

    def _change_dataset(self):
        new = simpledialog.askstring("Dataset id", "Dataset id (folder under data/):",
                                     initialvalue=self.repo_id, parent=self.root)
        if new and new != self.repo_id:
            self.repo_id = new
            _save_settings({"rec_repo": new})
            self._reconnect()

    def _change_task(self):
        new = simpledialog.askstring("Task", "Task description (saved with each demo):",
                                     initialvalue=self.task, parent=self.root)
        if new:
            self.task = new
            _save_settings({"rec_task": new})
            self._refresh_header()

    # -- tools (these necessarily open their own window/console) --------------
    def _launch_module(self, module, args):
        kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if os.name == "nt" else {}
        subprocess.Popen([sys.executable, "-m", module, *args], **kw)

    def _launch_exe(self, name, args):
        exe = os.path.join(os.path.dirname(sys.executable), name + (".exe" if os.name == "nt" else ""))
        prog = exe if os.path.exists(exe) else name
        kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if os.name == "nt" else {}
        subprocess.Popen([prog, *args], **kw)

    def _calibrate(self):
        cfg = load_config("robot")
        self._launch_exe("lerobot-calibrate", ["--robot.type=so101_follower",
                         f"--robot.port={cfg['port']}", f"--robot.id={cfg['id']}"])

    def _setup_motors(self):
        cfg = load_config("robot")
        self._launch_exe("lerobot-setup-motors", ["--robot.type=so101_follower",
                         f"--robot.port={cfg['port']}", f"--robot.id={cfg['id']}"])

    def _train(self):
        root = REPO_ROOT / "data" / self.repo_id.replace("/", "__")
        out = REPO_ROOT / "outputs" / "train" / "act"
        device = _load_settings().get("tr_device", "cpu")
        self._launch_exe("lerobot-train", [
            f"--dataset.repo_id={self.repo_id}", f"--dataset.root={root}",
            "--policy.type=act", f"--policy.device={device}",
            f"--output_dir={out}", "--steps=20000"])

    def _on_close(self):
        self._disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
