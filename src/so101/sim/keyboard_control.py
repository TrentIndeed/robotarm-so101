"""Keyboard fallback for the practice sim — drive the arm with no controller.

Presents the same interface the practice loop expects from XboxTeleopController
(``dt``, ``cfg``, ``connect`` / ``seed_targets`` / ``compute_action`` / ``disconnect``)
plus an ``on_key(keycode)`` hook the MuJoCo viewer calls on each key press. Because
the viewer only reports key-down events (which auto-repeat when held), motion is
incremental: each press nudges a joint by a fixed step.

This is sim-only — it relies on the viewer window for key events. The real arm
(``so101.xbox_teleop``) still needs the Xbox controller.

Key map:
    A / D   shoulder pan -/+        W / S   shoulder lift +/-
    I / K   elbow flex   +/-        J / L   wrist roll  -/+
    T / G   wrist flex   +/-        F       gripper open/close toggle
    R       respawn block (handled by the practice loop)
"""

from __future__ import annotations

from .. import load_config
from ..controller import GRIPPER_MAX, GRIPPER_MIN, JOINT_MAX, JOINT_MIN, _clip

STEP = 6.0  # normalized units moved per key press

# keycode (GLFW = uppercase ASCII for letters) -> (joint, direction)
_KEYMAP = {
    ord("A"): ("shoulder_pan", -1), ord("D"): ("shoulder_pan", +1),
    ord("W"): ("shoulder_lift", +1), ord("S"): ("shoulder_lift", -1),
    ord("I"): ("elbow_flex", +1), ord("K"): ("elbow_flex", -1),
    ord("J"): ("wrist_roll", -1), ord("L"): ("wrist_roll", +1),
    ord("T"): ("wrist_flex", +1), ord("G"): ("wrist_flex", -1),
}
_GRIPPER_KEY = ord("F")


class KeyboardController:
    """Incremental keyboard teleop with the controller's normalized interface."""

    def __init__(self):
        self.cfg = load_config("teleop")
        self.dt = 1.0 / self.cfg["control_hz"]
        self._joints = load_config("robot")["joints"]
        self.targets: dict[str, float] = {}

    # -- lifecycle (match XboxTeleopController) ------------------------------
    def connect(self) -> None:
        print("Keyboard mode: A/D W/S I/K J/L T/G move joints, F toggles gripper.")

    def seed_targets(self, observation: dict) -> None:
        for joint in self._joints:
            self.targets[joint] = float(observation.get(f"{joint}.pos", 0.0))

    def disconnect(self) -> None:
        pass

    # -- input --------------------------------------------------------------
    def on_key(self, keycode: int) -> None:
        if keycode in _KEYMAP:
            joint, direction = _KEYMAP[keycode]
            lo, hi = (GRIPPER_MIN, GRIPPER_MAX) if joint == "gripper" else (JOINT_MIN, JOINT_MAX)
            self.targets[joint] = _clip(self.targets[joint] + direction * STEP, lo, hi)
        elif keycode == _GRIPPER_KEY:
            g = self.cfg["gripper"]
            # Toggle: if more than halfway open, close; else open.
            mid = (g["open_pos"] + g["closed_pos"]) / 2
            self.targets["gripper"] = float(
                g["closed_pos"] if self.targets["gripper"] > mid else g["open_pos"]
            )

    def compute_action(self) -> dict:
        return {f"{j}.pos": v for j, v in self.targets.items()}
