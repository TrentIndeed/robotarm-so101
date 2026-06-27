"""Calibrated "reach toward the cursor" control for the SO-101.

Point at a spot on the table in the desk camera and the arm reaches there. Rather
than full inverse kinematics (which would need an exact URDF, camera calibration,
and a normalized<->radian joint mapping the hardware doesn't hand us), this learns
the map *directly from real arm poses*:

  Calibration: jog the arm so the gripper touches a spot on the table, then click
  that spot in the desk camera. Repeat for ~6-8 spots spread across the workspace.
  Each sample is (desk-image pixel u,v in 0..1) -> (the arm's joint angles there).

  Fit: a low-order polynomial surface  q_j = f_j(u, v)  per controlled joint,
  least-squares over the samples. Anchored to actual poses, so it's exact at the
  calibrated spots and interpolates smoothly between them. No kinematics or units.

  Runtime: cursor (u,v) over the desk camera -> evaluate the surfaces -> joint
  targets; the controller eases toward them (rate-limited) so motion stays smooth.

ReachController exposes the same interface as XboxTeleopController / DesktopController
(connect / seed_targets / compute_action / disconnect, plus .dt and .cfg), so it drops
into the same record/teleop worker. Gripper is on the mouse buttons, a manual height
nudge on the scroll wheel.
"""

from __future__ import annotations

import json
import threading

import numpy as np

from . import REPO_ROOT, load_config
from .controller import GRIPPER_MAX, GRIPPER_MIN, JOINT_MAX, JOINT_MIN, _clip

# Joints the reach map drives. wrist_roll + gripper stay under manual/button control:
# pan aims the base azimuth, lift+elbow set reach/height, wrist_flex keeps the
# gripper angled down at the table.
CONTROLLED = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

CALIB_PATH = REPO_ROOT / ".ik_calib.json"

# ---- tunables ----
EASE = 0.18           # fraction of the remaining gap closed per tick (smooth follow)
REACH_MAX_SPEED = 38.0  # top speed cap (units/sec) for EVERY joint that follows the cursor
SCROLL_STEP = 5.0     # elbow units per wheel notch (manual height fine-tune)
GRIP_SPEED = 90.0     # gripper units/sec while a mouse button is held


def _features(u: float, v: float, degree: int) -> np.ndarray:
    """Polynomial feature vector for an image point. degree 1 -> [1,u,v];
    degree 2 -> [1,u,v,u^2,uv,v^2]."""
    if degree >= 2:
        return np.array([1.0, u, v, u * u, u * v, v * v])
    return np.array([1.0, u, v])


def _degree_for(n_samples: int) -> int:
    """Use a quadratic surface when there's enough data, else fall back to linear."""
    return 2 if n_samples >= 6 else 1


def fit_calibration(samples: list[dict]) -> dict:
    """Fit q_j = f_j(u, v) per controlled joint from calibration samples.

    samples: [{"u":float, "v":float, "joints":{joint:value, ...}}, ...]
    Returns {"degree":int, "coeffs":{joint:[...]}}.
    """
    if len(samples) < 3:
        raise ValueError(f"Need at least 3 calibration points, got {len(samples)}.")
    degree = _degree_for(len(samples))
    X = np.array([_features(s["u"], s["v"], degree) for s in samples])
    coeffs = {}
    for j in CONTROLLED:
        y = np.array([float(s["joints"][j]) for s in samples])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        coeffs[j] = c.tolist()
    return {"degree": degree, "coeffs": coeffs}


def eval_calibration(calib: dict, u: float, v: float) -> dict:
    """Evaluate the fitted surfaces at image point (u, v) -> {joint: value}."""
    f = _features(u, v, calib["degree"])
    return {j: float(np.dot(c, f)) for j, c in calib["coeffs"].items()}


def save_calibration(calib: dict) -> None:
    CALIB_PATH.write_text(json.dumps(calib, indent=2), encoding="utf-8")


def load_calibration() -> dict | None:
    try:
        calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        if "coeffs" in calib and "degree" in calib:
            return calib
    except (FileNotFoundError, ValueError, KeyError):
        pass
    return None


class ReachController:
    """Cursor-on-the-table -> the arm reaches there, using a fitted calibration."""

    def __init__(self):
        self.cfg = load_config("teleop")
        self.dt = 1.0 / self.cfg["control_hz"]
        self._joints = load_config("robot")["joints"]
        self.targets: dict[str, float] = {}
        self.calib = load_calibration()

        self._lock = threading.Lock()
        self._u = self._v = 0.5     # latest cursor position over the desk view (0..1)
        self._active = False        # is the cursor currently over the desk view
        self._scroll = 0.0
        self._lclick = False
        self._rclick = False

    @property
    def calibrated(self) -> bool:
        return self.calib is not None

    # -- lifecycle (match the other controllers) ----------------------------
    def connect(self):
        if self.calibrated:
            print("Reach control: point at the table in the desk camera; the arm reaches there. "
                  "Scroll = height, L/R click = gripper.")
        else:
            print("Reach control: NOT CALIBRATED yet. Use Tools -> Calibrate reach first.")

    def disconnect(self):
        pass

    def seed_targets(self, observation):
        for j in self._joints:
            self.targets[j] = float(observation.get(f"{j}.pos", 0.0))

    # -- event handlers (Tk main thread) ------------------------------------
    def on_cursor(self, u, v, active):
        with self._lock:
            self._u, self._v, self._active = u, v, active

    def on_scroll(self, delta):
        with self._lock:
            self._scroll += delta

    def set_click(self, which, down):
        if which == "l":
            self._lclick = down
        elif which == "r":
            self._rclick = down

    # -- per-tick (worker thread) -------------------------------------------
    def compute_action(self):
        with self._lock:
            u, v, active = self._u, self._v, self._active
            scroll = self._scroll
            self._scroll = 0.0

        # Cursor on the table -> ease the reach joints toward the fitted pose, but cap
        # every joint's per-tick move so the whole arm never lunges (top-speed limit).
        if active and self.calib is not None:
            goal = eval_calibration(self.calib, u, v)
            max_step = REACH_MAX_SPEED * self.dt
            for j, val in goal.items():
                val = _clip(val, JOINT_MIN, JOINT_MAX)
                delta = (val - self.targets[j]) * EASE
                delta = max(-max_step, min(max_step, delta))
                self.targets[j] += delta

        # Scroll = manual height fine-tune (nudges the elbow).
        if scroll:
            self.targets["elbow_flex"] = _clip(
                self.targets["elbow_flex"] + (scroll / 120.0) * SCROLL_STEP, JOINT_MIN, JOINT_MAX)

        # Gripper on the mouse buttons (hold to move).
        grip = GRIP_SPEED * self.dt
        if self._lclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] + grip, GRIPPER_MIN, GRIPPER_MAX)
        elif self._rclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] - grip, GRIPPER_MIN, GRIPPER_MAX)

        return {f"{j}.pos": v for j, v in self.targets.items()}
