"""Hybrid mouse+keyboard control for the SO-101, with a calibrated mouse mapping.

Scheme:
  * A / D  -> base   (shoulder_pan)      keyboard jog
  * W / S  -> shoulder (shoulder_lift)   keyboard jog
  * Q / E  -> wrist twist (wrist_roll)   keyboard jog
  * mouse over the desk camera -> elbow + wrist bend (elbow_flex, wrist_flex)
  * L / R click -> gripper open / close

The mouse part is calibrated rather than a raw velocity: instead of full inverse
kinematics (which would need an exact URDF, camera calibration, and a normalized<->
radian joint map the hardware doesn't give us), it learns the map *directly from real
poses*:

  Calibration: relax the arm, move the gripper by hand to each dot shown on the desk
  camera, press Enter. Each sample is (desk-image pixel u,v) -> (elbow_flex, wrist_flex)
  at that pose. A low-order polynomial surface q_j = f_j(u, v) is least-squares fit per
  joint — exact at the calibrated dots, smooth between, no kinematics or units.

  Runtime: the cursor (u,v) over the desk camera evaluates the surfaces; elbow + wrist
  bend ease toward them (rate-limited + speed-capped) while you aim base/shoulder/twist
  with the keyboard.

ReachController exposes the same interface as XboxTeleopController / DesktopController
(connect / seed_targets / compute_action / disconnect, plus .dt and .cfg), so it drops
into the same record/teleop worker.
"""

from __future__ import annotations

import json
import threading

import numpy as np

from . import REPO_ROOT, load_config
from .controller import GRIPPER_MAX, GRIPPER_MIN, JOINT_MAX, JOINT_MIN, _clip

# Joints the MOUSE drives (and that calibration fits): elbow + wrist bend. The base
# and shoulder are jogged with WASD and the wrist twist with Q/E (keyboard), so the
# calibration only needs these two.
CONTROLLED = ["elbow_flex", "wrist_flex"]

# Keyboard jog: key (lowercase Tk keysym) -> (joint, direction).
_KEYS = {
    "a": ("shoulder_pan", -1), "d": ("shoulder_pan", +1),    # base
    "w": ("shoulder_lift", -1), "s": ("shoulder_lift", +1),  # shoulder (W = raise)
    "q": ("wrist_roll", -1), "e": ("wrist_roll", +1),        # wrist twist
}

CALIB_PATH = REPO_ROOT / ".ik_calib.json"

# ---- tunables ----
KEY_SPEED = 60.0      # normalized units/sec while a jog key is held
EASE = 0.18           # fraction of the remaining gap closed per tick (smooth follow)
REACH_MAX_SPEED = 38.0  # top speed cap (units/sec) for the mouse-driven joints
GRIP_SPEED = 90.0     # gripper units/sec while a mouse button is held


def fit_calibration(samples: list[dict]) -> dict:
    """Build the cursor -> (elbow, wrist) map from calibration samples.

    samples: [{"u":float, "v":float, "joints":{joint:value, ...}}, ...]

    The dots are a regular grid, so we store the captured joint value at each grid
    node and interpolate BILINEARLY at runtime. Unlike a global polynomial, bilinear
    interpolation is exact at every dot and stays a blend of the four surrounding real
    poses in between — it can't overshoot to an extreme (e.g. the arm shooting straight
    up) in the middle of the workspace. Falls back to a least-squares polynomial only
    if the samples don't form a complete grid.
    """
    if len(samples) < 3:
        raise ValueError(f"Need at least 3 calibration points, got {len(samples)}.")
    grid = _try_grid(samples)
    return grid if grid is not None else _poly_fit(samples)


def _try_grid(samples: list[dict]) -> dict | None:
    us = sorted({round(s["u"], 4) for s in samples})
    vs = sorted({round(s["v"], 4) for s in samples})
    if len(us) < 2 or len(vs) < 2 or len(us) * len(vs) != len(samples):
        return None
    nodes = {j: [[None] * len(us) for _ in vs] for j in CONTROLLED}
    for s in samples:
        ci, ri = us.index(round(s["u"], 4)), vs.index(round(s["v"], 4))
        for j in CONTROLLED:
            nodes[j][ri][ci] = float(s["joints"][j])
    if any(val is None for j in CONTROLLED for row in nodes[j] for val in row):
        return None      # incomplete grid -> let the polynomial fallback handle it
    return {"type": "grid", "u": us, "v": vs, "nodes": nodes}


def _poly_fit(samples: list[dict]) -> dict:
    degree = 2 if len(samples) >= 6 else 1
    X = np.array([_features(s["u"], s["v"], degree) for s in samples])
    coeffs = {}
    for j in CONTROLLED:
        y = np.array([float(s["joints"][j]) for s in samples])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        coeffs[j] = c.tolist()
    return {"type": "poly", "degree": degree, "coeffs": coeffs}


def _features(u: float, v: float, degree: int) -> np.ndarray:
    if degree >= 2:
        return np.array([1.0, u, v, u * u, u * v, v * v])
    return np.array([1.0, u, v])


def _cell(coords: list[float], x: float) -> tuple[int, float]:
    """Lower index + fraction for x within a sorted coordinate list, clamped to the
    ends (so a cursor outside the calibrated grid sticks to the edge, never extrapolates)."""
    if x <= coords[0]:
        return 0, 0.0
    if x >= coords[-1]:
        return len(coords) - 2, 1.0
    for i in range(len(coords) - 1):
        if x < coords[i + 1]:
            return i, (x - coords[i]) / (coords[i + 1] - coords[i])
    return len(coords) - 2, 1.0


def eval_calibration(calib: dict, u: float, v: float) -> dict:
    """Evaluate the calibrated map at image point (u, v) -> {joint: value}."""
    if calib.get("type") == "grid":
        us, vs = calib["u"], calib["v"]
        ci, fu = _cell(us, u)
        ri, fv = _cell(vs, v)
        out = {}
        for j, grid in calib["nodes"].items():
            top = grid[ri][ci] * (1 - fu) + grid[ri][ci + 1] * fu
            bot = grid[ri + 1][ci] * (1 - fu) + grid[ri + 1][ci + 1] * fu
            out[j] = top * (1 - fv) + bot * fv
        return out
    f = _features(u, v, calib["degree"])      # legacy polynomial calibration
    return {j: float(np.dot(c, f)) for j, c in calib["coeffs"].items()}


def save_calibration(calib: dict) -> None:
    CALIB_PATH.write_text(json.dumps(calib, indent=2), encoding="utf-8")


def load_calibration() -> dict | None:
    try:
        calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        if calib.get("type") == "grid" and "nodes" in calib:
            return calib
        if "coeffs" in calib and "degree" in calib:     # legacy polynomial file
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
        # Base angle the elbow+wrist map was calibrated against (base stays on A/D).
        self.base_reference = (self.calib or {}).get("base_reference")

        self._keys = {k: False for k in _KEYS}     # pre-seeded so no dict resize races
        self._lock = threading.Lock()
        self._u = self._v = 0.5     # latest cursor position over the desk view (0..1)
        self._active = False        # is the cursor currently over the desk view
        self._lclick = False
        self._rclick = False

    @property
    def calibrated(self) -> bool:
        return self.calib is not None

    # -- lifecycle (match the other controllers) ----------------------------
    def connect(self):
        if self.calibrated:
            print("Reach control: WASD = base/shoulder, Q/E = wrist twist, mouse on the desk "
                  "camera = elbow + wrist bend, L/R click = gripper.")
        else:
            print("Reach control: NOT CALIBRATED yet. Press 'Calibrate reach' first.")

    def disconnect(self):
        pass

    def seed_targets(self, observation):
        for j in self._joints:
            self.targets[j] = float(observation.get(f"{j}.pos", 0.0))

    # -- event handlers (Tk main thread) ------------------------------------
    def set_key(self, keysym, down):
        if keysym in self._keys:
            self._keys[keysym] = down

    def on_cursor(self, u, v, active):
        with self._lock:
            self._u, self._v, self._active = u, v, active

    def on_scroll(self, delta):
        pass   # scroll unused in this scheme (elbow is on the mouse)

    def set_click(self, which, down):
        if which == "l":
            self._lclick = down
        elif which == "r":
            self._rclick = down

    # -- per-tick (worker thread) -------------------------------------------
    def compute_action(self):
        # Keyboard jog: base + shoulder (WASD) and wrist twist (Q/E).
        step = KEY_SPEED * self.dt
        for k, (joint, d) in _KEYS.items():
            if self._keys.get(k):
                self.targets[joint] = _clip(self.targets[joint] + d * step, JOINT_MIN, JOINT_MAX)

        with self._lock:
            u, v, active = self._u, self._v, self._active

        # Cursor on the desk camera -> ease elbow + wrist bend toward the fitted pose,
        # capping the per-tick move so they never lunge (top-speed limit).
        if active and self.calib is not None:
            goal = eval_calibration(self.calib, u, v)
            max_step = REACH_MAX_SPEED * self.dt
            for j, val in goal.items():
                val = _clip(val, JOINT_MIN, JOINT_MAX)
                delta = (val - self.targets[j]) * EASE
                delta = max(-max_step, min(max_step, delta))
                self.targets[j] += delta

        # Gripper on the mouse buttons (hold to move).
        grip = GRIP_SPEED * self.dt
        if self._lclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] + grip, GRIPPER_MIN, GRIPPER_MAX)
        elif self._rclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] - grip, GRIPPER_MIN, GRIPPER_MAX)

        return {f"{j}.pos": v for j, v in self.targets.items()}
