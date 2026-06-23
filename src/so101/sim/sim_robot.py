"""A PyBullet-backed stand-in for the SO-101 follower.

Exposes the subset of the LeRobot robot interface the Xbox controller needs:
``get_observation()`` and ``send_action()`` keyed by ``"<joint>.pos"`` in the same
normalized units as the real arm (joints: -100..100, gripper: 0 closed .. 100 open).
That means the exact same XboxTeleopController loop drives the sim and the hardware.
"""

from __future__ import annotations

from pathlib import Path

import pybullet as p

URDF_PATH = Path(__file__).with_name("so101.urdf")

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
    """Wraps a loaded URDF and maps normalized joint commands to PyBullet motors."""

    def __init__(self, base_position=(0.0, 0.0, 0.0)):
        self.body_id = p.loadURDF(
            str(URDF_PATH),
            basePosition=base_position,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        # name -> joint index, plus cached limits for normalization.
        self._idx: dict[str, int] = {}
        self._limits: dict[str, tuple[float, float]] = {}
        for j in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, j)
            name = info[1].decode()
            self._idx[name] = j
            self._limits[name] = (info[8], info[9])  # lower, upper

    # -- normalization helpers ----------------------------------------------
    def _norm_to_angle(self, joint: str, n: float) -> float:
        lo, hi = self._limits[joint]
        return lo + (n + 100.0) / 200.0 * (hi - lo)

    def _angle_to_norm(self, joint: str, a: float) -> float:
        lo, hi = self._limits[joint]
        return (a - lo) / (hi - lo) * 200.0 - 100.0

    def _grip_to_pos(self, n: float) -> float:
        # gripper 0..100 (closed..open) -> finger travel 0..upper limit
        _, hi = self._limits["left_finger"]
        return max(0.0, min(1.0, n / 100.0)) * hi

    # -- LeRobot-like interface ---------------------------------------------
    def get_observation(self) -> dict:
        obs: dict = {}
        for j in REVOLUTE_JOINTS:
            angle = p.getJointState(self.body_id, self._idx[j])[0]
            obs[f"{j}.pos"] = self._angle_to_norm(j, angle)
        # Report gripper from the left finger's travel.
        _, hi = self._limits["left_finger"]
        pos = p.getJointState(self.body_id, self._idx["left_finger"])[0]
        obs["gripper.pos"] = (pos / hi) * 100.0 if hi else 0.0
        return obs

    def send_action(self, action: dict) -> None:
        for j in REVOLUTE_JOINTS:
            key = f"{j}.pos"
            if key in action:
                p.setJointMotorControl2(
                    self.body_id,
                    self._idx[j],
                    p.POSITION_CONTROL,
                    targetPosition=self._norm_to_angle(j, action[key]),
                    force=8.0,
                    maxVelocity=3.0,
                )
        if "gripper.pos" in action:
            target = self._grip_to_pos(action["gripper.pos"])
            for f in FINGER_JOINTS:
                p.setJointMotorControl2(
                    self.body_id,
                    self._idx[f],
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=20.0,
                    maxVelocity=1.0,
                )

    @property
    def gripper_link_index(self) -> int:
        """Index of the gripper base link (handy for camera-follow / contacts)."""
        return self._idx["wrist_roll"]
