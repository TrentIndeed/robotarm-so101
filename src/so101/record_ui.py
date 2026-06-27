"""The SO-101 app — one window: live cameras, demo recording, settings in the menu.

The robot control loop (serial I/O, camera reads, recording, video encoding) runs in a
BACKGROUND thread; the Tk main thread only draws the latest frames and sends commands.
That keeps the window responsive (draggable) and means saving a take — which encodes
video — never freezes the UI or stalls the controller. Videos use h264 (fast: ~1 s a
take, vs ~30 s for the default AV1). "Watch" finalizes the dataset so the demo is
readable, opens the replay, then resumes so you can keep recording.

Opens with your saved settings (.app_settings.json). Drive with the Xbox controller;
Start = begin a take, Start again (or the on-screen button) = stop + save, X = discard.
The on-screen buttons reflect the controller. Settings + tools are in the menu bar.

    python -m so101            # this app (via ./run)
"""

from __future__ import annotations

import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
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
VCODEC = "h264"     # fast x264 encode (~1 s/take) vs the default libsvtav1 (~30 s)


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
        self.input_mode = s.get("input_mode", "xbox")   # "xbox" or "desktop"

        self.rng = random.Random()
        self.cam_labels: dict = {}
        self._imgs: dict = {}
        self._lock = threading.Lock()
        self._cmd: queue.Queue = queue.Queue()
        self._shared = self._fresh_shared()
        self._worker = None
        self._save_thread = None
        self._input = None        # input controller (xbox or desktop), created per session

        root.title("SO-101")
        self._build_menu()
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_worker()
        self.root.after(60, self._refresh)

    # -- shared state --------------------------------------------------------
    def _fresh_shared(self):
        return {"frames": {}, "status": ("Connecting…", "#a60"), "episodes": 0,
                "recording": False, "running": True, "connected": False, "cam_names": [],
                "saving": False}

    def _set(self, **kw):
        with self._lock:
            self._shared.update(kw)

    def _get(self, key):
        with self._lock:
            return self._shared[key]

    # -- menu bar ------------------------------------------------------------
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
        self._input_var = tk.StringVar(value=self.input_mode)
        setm.add_radiobutton(label="Input: Xbox controller", variable=self._input_var,
                             value="xbox", command=self._change_input)
        setm.add_radiobutton(label="Input: Keyboard + mouse", variable=self._input_var,
                             value="desktop", command=self._change_input)
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
        ttk.Label(self.cam_box, text="Connecting…", font=("Segoe UI", 12)).pack(padx=80, pady=60)

        self.status = tk.StringVar(value="…")
        self.status_lbl = tk.Label(self.root, textvariable=self.status, font=("Segoe UI", 16, "bold"))
        self.status_lbl.pack(pady=(0, 4))

        controls = ttk.Frame(self.root)
        controls.pack(pady=4)
        self.start_btn = tk.Button(controls, text="Start recording", width=16, height=2,
                                   bg="#1a7", fg="white", font=("Segoe UI", 11, "bold"),
                                   state="disabled", command=lambda: self._cmd.put(("toggle",)))
        self.start_btn.grid(row=0, column=0, padx=6)
        self.discard_btn = tk.Button(controls, text="Discard take", width=14, height=2,
                                     state="disabled", command=lambda: self._cmd.put(("discard",)))
        self.discard_btn.grid(row=0, column=1, padx=6)

        self.count = tk.StringVar(value="Saved episodes: 0")
        ttk.Label(self.root, textvariable=self.count, font=("Segoe UI", 12)).pack(pady=2)

        self.cmap = ttk.LabelFrame(self.root, text="Controls — what each input moves")
        self.cmap.pack(fill="x", padx=10, pady=(2, 6))
        self._refresh_legend()

        # Mouse pad (used only in Keyboard+Mouse mode): move = wrist, L/R click = grip.
        self.pad = tk.Canvas(self.root, height=70, bg="#202020", highlightthickness=1,
                             highlightbackground="#888")
        self.pad.pack(fill="x", padx=10, pady=(0, 6))
        self.pad.create_text(12, 35, anchor="w", fill="#aaa",
                             text="Keyboard+Mouse: move mouse over the camera views (or here) = wrist  ·  "
                                  "L-click = open  ·  R-click = close")
        self.pad.bind("<Motion>", lambda e: self._route("mouse", e))
        self.pad.bind("<Leave>", lambda e: self._route("leave", e))
        self.pad.bind("<ButtonPress-1>", lambda e: self._route("l", True))
        self.pad.bind("<ButtonRelease-1>", lambda e: self._route("l", False))
        self.pad.bind("<ButtonPress-3>", lambda e: self._route("r", True))
        self.pad.bind("<ButtonRelease-3>", lambda e: self._route("r", False))
        self.root.bind("<KeyPress>", lambda e: self._on_key(e, True))
        self.root.bind("<KeyRelease>", lambda e: self._on_key(e, False))

        watch = ttk.LabelFrame(self.root, text="Watch demos")
        watch.pack(fill="both", expand=True, padx=10, pady=8)
        self.ep_list = tk.Listbox(watch, height=6)
        self.ep_list.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.ep_list.bind("<Double-Button-1>", lambda _e: self._watch_selected())
        wbtns = ttk.Frame(watch)
        wbtns.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(wbtns, text="Watch selected", command=self._watch_selected).pack(fill="x", pady=2)
        ttk.Button(wbtns, text="Delete selected", command=self._delete_selected).pack(fill="x", pady=2)

        ttk.Label(self.root, text="Controller: Start = begin/stop+save a take, X = discard.",
                  foreground="#777").pack(anchor="w", padx=10, pady=(0, 8))

    def _refresh_header(self):
        self.header.set(f"Backend: {'SIM' if self.sim else 'REAL'}    Dataset: {self.repo_id}"
                        f"    Task: {self.task}")

    def _populate_cameras(self, cam_names):
        for w in self.cam_box.winfo_children():
            w.destroy()
        self.cam_labels = {}
        for i, name in enumerate(cam_names):
            col = ttk.Frame(self.cam_box)
            col.grid(row=0, column=i, padx=6)
            ttk.Label(col, text=name).pack()
            lbl = ttk.Label(col)
            lbl.pack()
            # The camera view is also a mouse-control surface (keyboard+mouse mode):
            # move over it = wrist, left/right click = open/close gripper.
            lbl.bind("<Motion>", lambda e: self._route("mouse", e))
            lbl.bind("<Leave>", lambda e: self._route("leave", e))
            lbl.bind("<ButtonPress-1>", lambda e: self._route("l", True))
            lbl.bind("<ButtonRelease-1>", lambda e: self._route("l", False))
            lbl.bind("<ButtonPress-3>", lambda e: self._route("r", True))
            lbl.bind("<ButtonRelease-3>", lambda e: self._route("r", False))
            self.cam_labels[name] = lbl

    def _refresh_legend(self):
        for w in self.cmap.winfo_children():
            w.destroy()
        if self.input_mode == "desktop":
            legend = [("A / D", "rotate base"), ("W / S", "raise / lower"),
                      ("Q / E", "reach in / out"), ("mouse on cams", "wrist roll / tilt"),
                      ("L / R click", "gripper open / close"), ("Enter / Backspace", "save / discard")]
        else:
            legend = [("Left stick L/R", "rotate base"), ("Left stick U/D", "raise / lower arm"),
                      ("Right stick L/R", "twist wrist"), ("Right stick U/D", "reach in / out"),
                      ("LT / RT", "wrist tilt down / up"), ("A / B", "open / close gripper"),
                      ("Back/View", "hold (e-stop)"), ("Start / X", "save take / discard")]
        for i, (ctrl, what) in enumerate(legend):
            row, col = i // 2, (i % 2) * 2
            ttk.Label(self.cmap, text=ctrl, font=("Consolas", 9, "bold")).grid(
                row=row, column=col, sticky="w", padx=(8, 4), pady=1)
            ttk.Label(self.cmap, text=what, foreground="#555").grid(
                row=row, column=col + 1, sticky="w", padx=(0, 16), pady=1)

    def _route(self, kind, arg):
        from .desktop_control import DesktopController
        if not isinstance(self._input, DesktopController):
            return
        if kind == "mouse":
            self._input.on_mouse(arg.x, arg.y)
        elif kind == "leave":
            self._input.on_mouse_leave()
        else:
            self._input.set_click(kind, arg)

    def _on_key(self, event, down):
        k = event.keysym.lower()
        if k == "return" and down:
            self._cmd.put(("toggle",))
        elif k == "backspace" and down:
            self._cmd.put(("discard",))
        else:
            from .desktop_control import DesktopController
            if isinstance(self._input, DesktopController):
                self._input.set_key(k, down)

    # -- UI refresh (main thread, light) -------------------------------------
    def _refresh(self):
        if not self.root.winfo_exists():
            return
        with self._lock:
            frames = dict(self._shared["frames"])
            status, episodes = self._shared["status"], self._shared["episodes"]
            recording, connected = self._shared["recording"], self._shared["connected"]
            cam_names = list(self._shared["cam_names"])
            saving = self._shared["saving"]

        self.status.set(status[0])
        self.status_lbl.configure(fg=status[1])
        self.count.set(f"Saved episodes: {episodes}")
        if cam_names and set(self.cam_labels) != set(cam_names):
            self._populate_cameras(cam_names)
        for cam, frame in frames.items():
            if cam in self.cam_labels:
                photo = ImageTk.PhotoImage(Image.fromarray(frame).resize((CAM_W, CAM_H)))
                self._imgs[cam] = photo
                self.cam_labels[cam].configure(image=photo)

        ready = connected and bool(cam_names)
        self.start_btn.configure(state="normal" if (ready and not saving) else "disabled",
                                 text="Stop & Save" if recording else "Start recording",
                                 bg="#c33" if recording else "#1a7")
        self.discard_btn.configure(state="normal" if (ready and recording) else "disabled")
        while self.ep_list.size() < episodes:
            self.ep_list.insert("end", f"Episode {self.ep_list.size()}")
        while self.ep_list.size() > episodes:
            self.ep_list.delete(self.ep_list.size() - 1)
        self.root.after(50, self._refresh)

    # -- worker thread (all robot/dataset I/O) -------------------------------
    def _start_worker(self):
        self._set(**self._fresh_shared())
        while not self._cmd.empty():
            self._cmd.get_nowait()
        if self.input_mode == "desktop":
            from .desktop_control import DesktopController
            self._input = DesktopController()
            self._input.connect()
        else:
            self._input = None              # worker creates the Xbox controller
        self._worker = threading.Thread(target=self._run_worker, args=(self.sim, self.repo_id), daemon=True)
        self._worker.start()

    def _run_worker(self, sim, repo_id):
        from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR

        robot = ctrl = dataset = None
        try:
            robot = make_robot(sim=sim, use_cameras=True)
            robot.connect()
            if self.input_mode == "desktop":
                ctrl = self._input          # DesktopController, created + bound on main thread
                btn, fps = {}, ctrl.cfg["control_hz"]
            else:
                try:
                    ctrl = XboxTeleopController()
                    ctrl.connect()
                    self._input = ctrl
                    btn, fps = ctrl.cfg["buttons"], ctrl.cfg["control_hz"]
                except Exception as exc:
                    ctrl, btn, fps = None, {}, 30
                    self._set(status=(f"Cameras only — no controller ({exc})", "#a60"))

            cam_names = [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]
            n_steps = max(1, round((1.0 / fps) / robot.model.opt.timestep)) if sim else 0
            root_dir = REPO_ROOT / "data" / repo_id.replace("/", "__")
            dataset, episodes = self._open_dataset(LeRobotDataset, hw_to_dataset_features,
                                                   robot, OBS_STR, ACTION, root_dir, repo_id, fps)

            self._set(cam_names=cam_names, episodes=episodes, connected=True,
                      status=("idle" if ctrl else "cameras only — no controller", "#0a0" if ctrl else "#a60"))
            if sim:
                _reset_sim_block(robot, self.rng)
            if ctrl:
                ctrl.seed_targets(robot.get_observation())

            recording = False
            prev: dict = {}
            dt = 1.0 / fps

            def pressed(b):
                now = bool(ctrl.joystick.get_button(b))
                was = prev.get(b, False)
                prev[b] = now
                return now and not was

            while self._get("running"):
                t0 = time.perf_counter()
                while True:
                    try:
                        cmd = self._cmd.get_nowait()
                    except queue.Empty:
                        break
                    if cmd[0] == "toggle":
                        recording = self._toggle(recording, dataset, sim, robot)
                    elif cmd[0] == "discard":
                        recording = self._discard(recording, dataset)
                    elif cmd[0] == "watch" and not recording and not self._get("saving"):
                        dataset = self._watch(dataset, LeRobotDataset, repo_id, root_dir, cmd[1])

                obs = robot.get_observation()
                if ctrl is not None:
                    action = ctrl.compute_action()
                    robot.send_action(action)
                    if sim:
                        robot.step(n_steps)
                    if hasattr(ctrl, "joystick"):    # Xbox hardware buttons for record control
                        if pressed(btn["episode_done"]):
                            recording = self._toggle(recording, dataset, sim, robot)
                        elif recording and pressed(btn["episode_cancel"]):
                            recording = self._discard(recording, dataset)
                    if recording:
                        of = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                        af = build_dataset_frame(dataset.features, action, prefix=ACTION)
                        dataset.add_frame({**of, **af, "task": self.task})

                self._set(frames={c: obs[c] for c in cam_names if c in obs})
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        except Exception as exc:
            self._set(connected=False, status=(f"Connection failed: {exc}", "#c00"))
        finally:
            if self._save_thread is not None and self._save_thread.is_alive():
                self._save_thread.join(timeout=15)   # let an in-flight (streaming) save finish
            try:
                if dataset is not None:
                    dataset.finalize()
            except Exception:
                pass
            for o in (ctrl, robot):
                try:
                    if o is not None:
                        o.disconnect()
                except Exception:
                    pass

    def _open_dataset(self, LeRobotDataset, hw_to_dataset_features, robot, OBS_STR, ACTION, root_dir, repo_id, fps):
        def _create():
            features = {**hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
                        **hw_to_dataset_features(robot.action_features, ACTION)}
            return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root_dir,
                                         robot_type=robot.name, use_videos=True, vcodec=VCODEC,
                                         streaming_encoding=True)
        if not root_dir.exists():
            return _create(), 0
        try:
            ds = LeRobotDataset.resume(repo_id=repo_id, root=root_dir, streaming_encoding=True)
            return ds, ds.num_episodes
        except Exception:
            ep_dir = root_dir / "meta" / "episodes"
            if ep_dir.exists() and any(ep_dir.rglob("*.parquet")):
                raise
            shutil.rmtree(root_dir, ignore_errors=True)
            return _create(), 0

    # -- worker-thread record actions ----------------------------------------
    def _toggle(self, recording, dataset, sim, robot):
        if not recording:
            if self._get("saving"):
                return False                         # previous take still encoding
            if sim:
                _reset_sim_block(robot, self.rng)
            self._set(recording=True, status=("RECORDING", "#c00"))
            return True
        # Stop -> encode on a SEPARATE thread so the worker keeps streaming cameras /
        # driving the arm while the (possibly slow) video encode runs in the background.
        self._set(recording=False, saving=True, status=("Saving…", "#a60"))
        self._save_thread = threading.Thread(target=self._save_episode, args=(dataset,), daemon=True)
        self._save_thread.start()
        return False

    def _save_episode(self, dataset):
        try:
            dataset.save_episode(parallel_encoding=False)
            with self._lock:
                self._shared["episodes"] += 1
                self._shared["saving"] = False
                self._shared["status"] = ("idle", "#0a0")
        except Exception as exc:
            self._set(saving=False, status=(f"Save failed: {exc}", "#c00"))

    def _discard(self, recording, dataset):
        if not recording:
            return False
        dataset.clear_episode_buffer()
        self._set(recording=False, status=("idle (discarded)", "#0a0"))
        return False

    def _watch(self, dataset, LeRobotDataset, repo_id, root_dir, idx):
        self._set(status=("Preparing demo…", "#a60"))
        try:
            dataset.finalize()                       # make it readable for the viewer
            self._launch_exe("lerobot-dataset-viz", [
                "--repo-id", repo_id, "--root", str(root_dir), "--episode-index", str(idx)])
            dataset = LeRobotDataset.resume(repo_id=repo_id, root=root_dir)   # keep recording
            self._set(status=("idle", "#0a0"))
        except Exception as exc:
            self._set(status=(f"Watch failed: {exc}", "#c00"))
        return dataset

    def _watch_selected(self):
        sel = self.ep_list.curselection()
        if sel:
            self._cmd.put(("watch", sel[0]))

    def _delete_selected(self):
        sel = self.ep_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if self._get("recording") or self._get("saving"):
            messagebox.showinfo("Busy", "Stop / finish the current take first.")
            return
        if not messagebox.askyesno("Delete demo",
                                   f"Delete Episode {idx}? This rebuilds the dataset and can't be undone."):
            return
        # Stop the worker so the dataset's files are released, then rebuild without
        # this episode using lerobot-edit-dataset, then restart the worker.
        self._set(running=False, connected=False, status=(f"Deleting episode {idx}…", "#a60"))
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=45)
        threading.Thread(target=self._do_delete, args=(idx,), daemon=True).start()

    def _do_delete(self, idx):
        root_dir = REPO_ROOT / "data" / self.repo_id.replace("/", "__")
        tmp = root_dir.with_name(root_dir.name + "__editing")
        shutil.rmtree(tmp, ignore_errors=True)
        exe = os.path.join(os.path.dirname(sys.executable),
                           "lerobot-edit-dataset" + (".exe" if os.name == "nt" else ""))
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        try:
            r = subprocess.run(
                [exe, "--repo_id", self.repo_id, "--root", str(root_dir),
                 "--operation.type", "delete_episodes", "--operation.episode_indices", f"[{idx}]",
                 "--new_repo_id", self.repo_id, "--new_root", str(tmp)],
                env=env, capture_output=True, text=True, timeout=900)
            if r.returncode == 0 and tmp.exists():
                shutil.rmtree(root_dir, ignore_errors=True)
                tmp.rename(root_dir)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
                self._set(status=("Delete failed (see console)", "#c00"))
                print(r.stdout[-800:], r.stderr[-1600:])
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            self._set(status=(f"Delete failed: {exc}", "#c00"))
        self.root.after(0, self._start_worker)   # reconnect to the rebuilt dataset

    # -- settings ------------------------------------------------------------
    def _change_backend(self):
        self.sim = self._backend_var.get() == "sim"
        _save_settings({"rec_backend": "sim" if self.sim else "real"})
        self._reconnect()

    def _change_input(self):
        self.input_mode = self._input_var.get()
        _save_settings({"input_mode": self.input_mode})
        self._refresh_legend()
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

    def _reconnect(self):
        self._set(running=False)
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=4)
        for w in self.cam_box.winfo_children():
            w.destroy()
        self.cam_labels = {}
        ttk.Label(self.cam_box, text="Connecting…", font=("Segoe UI", 12)).pack(padx=80, pady=60)
        self.ep_list.delete(0, "end")
        self._refresh_header()
        self._start_worker()

    # -- tools (own window/console) ------------------------------------------
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
        self._set(running=False)
        if self._worker is not None:
            self._worker.join(timeout=10)   # let it finalize + flush an in-flight save
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
    # Data is already finalized in _on_close. LeRobot's camera/encoder threads can be
    # non-daemon and a flaky-camera disconnect can block, which would hang the terminal
    # after the window closes — so force a clean process exit.
    os._exit(0)


if __name__ == "__main__":
    main()
