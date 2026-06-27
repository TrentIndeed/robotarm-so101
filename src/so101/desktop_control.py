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
MOUSE_SENS = 0.35      # units of wrist motion per pixel of mouse movement
GRIP_SPEED = 90.0      # units/sec while a mouse button is held
SCROLL_STEP = 6.0      # units of elbow motion per mouse-wheel notch


class DesktopController:
    def __init__(self):
        self.cfg = load_config("teleop")
        self.dt = 1.0 / self.cfg["control_hz"]
        self._joints = load_config("robot")["joints"]
        self.targets: dict[str, float] = {}

        self._keys = {k: False for k in _KEYS}     # pre-seeded so no dict resize races
        self._lock = threading.Lock()
        self._mdx = 0.0
        self._mdy = 0.0
        self._last = None
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

    def on_mouse(self, x, y):
        with self._lock:
            if self._last is not None:
                self._mdx += x - self._last[0]
                self._mdy += y - self._last[1]
            self._last = (x, y)

    def on_mouse_leave(self):
        with self._lock:
            self._last = None

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
            dx, dy = self._mdx, self._mdy
            scroll = self._scroll
            self._mdx = self._mdy = self._scroll = 0.0
        if scroll:
            self.targets["elbow_flex"] = _clip(
                self.targets["elbow_flex"] + (scroll / 120.0) * SCROLL_STEP, JOINT_MIN, JOINT_MAX)
        if dx:
            self.targets["wrist_roll"] = _clip(self.targets["wrist_roll"] - dx * MOUSE_SENS,
                                               JOINT_MIN, JOINT_MAX)
        if dy:
            self.targets["wrist_flex"] = _clip(self.targets["wrist_flex"] + dy * MOUSE_SENS,
                                               JOINT_MIN, JOINT_MAX)

        grip = GRIP_SPEED * self.dt
        if self._lclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] + grip, GRIPPER_MIN, GRIPPER_MAX)
        elif self._rclick:
            self.targets["gripper"] = _clip(self.targets["gripper"] - grip, GRIPPER_MIN, GRIPPER_MAX)

        return {f"{j}.pos": v for j, v in self.targets.items()}
