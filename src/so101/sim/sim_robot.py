"""A MuJoCo-backed stand-in for the SO-101 follower.

Exposes the subset of the LeRobot robot interface the Xbox controller needs:
``get_observation()`` and ``send_action()`` keyed by ``"<joint>.pos"`` in the same
normalized units as the real arm (joints: -100..100, gripper: 0 closed .. 100 open).
That means the exact same XboxTeleopController loop drives the sim and the hardware.

SimRobot owns the MjModel / MjData; ``practice.py`` reuses ``.model`` / ``.data`` for
the viewer, the block, and scoring.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).with_name("so101.xml")

# The five revolute joints, in the order the controller expects.
REVOLUTE_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
FINGER_JOINTS = ["left_finger", "right_finger"]


class SimRobot:
    """Maps normalized joint commands to MuJoCo position actuators."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        # Cache joint qpos addresses, ranges, and actuator ids by name.
        self._qadr: dict[str, int] = {}
        self._range: dict[str, tuple[float, float]] = {}
        self._act: dict[str, int] = {}
        for name in REVOLUTE_JOINTS + FINGER_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._qadr[name] = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self._range[name] = (float(lo), float(hi))
            self._act[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        mujoco.mj_forward(self.model, self.data)

    # -- normalization helpers ----------------------------------------------
    def _norm_to_angle(self, joint: str, n: float) -> float:
        lo, hi = self._range[joint]
        return lo + (n + 100.0) / 200.0 * (hi - lo)

    def _angle_to_norm(self, joint: str, a: float) -> float:
        lo, hi = self._range[joint]
        return (a - lo) / (hi - lo) * 200.0 - 100.0

    # -- LeRobot-like interface ---------------------------------------------
    def get_observation(self) -> dict:
        obs: dict = {}
        for j in REVOLUTE_JOINTS:
            angle = float(self.data.qpos[self._qadr[j]])
            obs[f"{j}.pos"] = self._angle_to_norm(j, angle)
        # Report gripper from the left finger's travel (0..range -> 0..100).
        lo, hi = self._range["left_finger"]
        pos = float(self.data.qpos[self._qadr["left_finger"]])
        obs["gripper.pos"] = (pos - lo) / (hi - lo) * 100.0
        return obs

    def send_action(self, action: dict) -> None:
        for j in REVOLUTE_JOINTS:
            key = f"{j}.pos"
            if key in action:
                self.data.ctrl[self._act[j]] = self._norm_to_angle(j, action[key])
        if "gripper.pos" in action:
            lo, hi = self._range["left_finger"]
            n = max(0.0, min(100.0, action["gripper.pos"]))
            target = lo + (n / 100.0) * (hi - lo)
            for f in FINGER_JOINTS:
                self.data.ctrl[self._act[f]] = target
