"""Keyboard + mouse control for the SO-101 (desktop teleop).

Mapping (the user's scheme):
  * A / D   -> rotate base   (shoulder_pan)
  * W / S   -> raise / lower (shoulder_lift)
  * Q / E   -> reach in/out  (elbow_flex)
  * mouse motion over the control pad -> the two wrist joints (roll / tilt)
  * left mouse button (hold)  -> open the gripper
  * right mouse button (hold) -> close the gripper

Same normalized ``{joint}.pos`` interface as XboxTeleopController, so it drops into
the same record/teleop loop. Tkinter event handlers (app main thread) latch key /
mouse / click state; ``compute_action()`` (called on the worker thread) reads it.
Motion is incremental + rate-limited, so it stays smooth. Tunables up top.
"""

from __future__ import annotations

import threading

from . import load_config
from .controller import GRIPPER_MAX, GRIPPER_MIN, JOINT_MAX, JOINT_MIN, _clip

# key (Tk keysym, lowercase) -> (joint, direction)
_KEYS = {
    "a": ("shoulder_pan", -1), "d": ("shoulder_pan", +1),
    "w": ("shoulder_lift", -1), "s": ("shoulder_lift", +1),   # W = raise, S = lower
    "q": ("elbow_flex", -1), "e": ("elbow_flex", +1),
}

# ---- tunables ----
KEY_SPEED = 60.0       # normalized units/sec while a key is held
WRIST_SPEED = 70.0     # units/sec wrist motion at full cursor offset from the view center
WRIST_DEADZONE = 0.12  # central neutral zone (fraction of half-width) where the wrist holds
GRIP_SPEED = 90.0      # units/sec while a mouse button is held
SCROLL_STEP = 6.0      # units of elbow motion per mouse-wheel notch


def _dz(v):
    """Zero inside the central deadzone; clamp the rest to [-1, 1]."""
    if abs(v) < WRIST_DEADZONE:
        return 0.0
    return max(-1.0, min(1.0, v))


class DesktopController:
    def __init__(self):
        self.cfg = load_config("teleop")
        self.dt = 1.0 / self.cfg["control_hz"]
        self._joints = load_config("robot")["joints"]
        self.targets: dict[str, float] = {}

        self._keys = {k: False for k in _KEYS}     # pre-seeded so no dict resize races
        self._lock = threading.Lock()
        self._mx = self._my = 0.0      # latest cursor position within the camera view
        self._mw = self._mh = 0.0      # that view's size (for the centre point)
        self._mactive = False          # is the cursor currently over a control surface
        self._lclick = False
        self._rclick = False
        self._scroll = 0.0

    # -- lifecycle (match XboxTeleopController) ------------------------------
    def connect(self):
        print("Keyboard+mouse control: A/D base, W/S lift, Q/E reach, "
              "mouse pad = wrist, L/R click = gripper.")

    def disconnect(self):
        pass

    def seed_targets(self, observation):
        for j in self._joints:
            self.targets[j] = float(observation.get(f"{j}.pos", 0.0))

    # -- event handlers (called on the Tk main thread) ----------------------
    def set_key(self, keysym, down):
        if keysym in self._keys:
            self._keys[keysym] = down

    def on_mouse(self, x, y, w, h):
        with self._lock:
            self._mx, self._my, self._mw, self._mh = x, y, w, h
            self._mactive = True

    def on_mouse_leave(self):
        with self._lock:
            self._mactive = False

    def set_click(self, which, down):
        if which == "l":
            self._lclick = down
        elif which == "r":
            self._rclick = down

    def on_scroll(self, delta):
        with self._lock:
            self._scroll += delta

    # -- per-tick (called on the worker thread) -----------------------------
    def compute_action(self):
        step = KEY_SPEED * self.dt
        for k, (joint, d) in _KEYS.items():
            if self._keys.get(k):
                self.targets[joint] = _clip(self.targets[joint] + d * step, JOINT_MIN, JOINT_MAX)

        with self._lock:
            active = self._mactive
            mx, my, mw, mh = self._mx, self._my, self._mw, self._mh
            scroll = self._scroll
            self._scroll = 0.0
        if scroll:
            self.targets["elbow_flex"] = _clip(
                self.targets["elbow_flex"] + (scroll / 120.0) * SCROLL_STEP, JOINT_MIN, JOINT_MAX)
        # Wrist = where the cursor sits relative to the view centre (centre = neutral/hold).
        if active and mw > 1 and mh > 1:
            ox = _dz((mx - mw / 2.0) / (mw / 2.0))
            oy = _dz((my - mh / 2.0) / (mh / 2.0))
            wstep = WRIST_SPEED * self.dt
            if ox:
                self.targets["wrist_roll"] = _clip(self.targets["wrist_roll"] - ox * wstep,
                                                   JOINT_MIN, JOINT_MAX)
            if oy:
                self.targets["wrist_flex"] = _clip(self.targets["wrist_flex"] + oy * wstep,
                                                   JOINT_MIN, JOINT_MAX)

        grip = GRIP_SPEED * self.dt
        if self._lclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] + grip, GRIPPER_MIN, GRIPPER_MAX)
        elif self._rclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] - grip, GRIPPER_MIN, GRIPPER_MAX)

        return {f"{j}.pos": v for j, v in self.targets.items()}
