"""A MuJoCo-backed stand-in for the SO-101 follower.

Implements the same interface LeRobot's real robot exposes, so the same teleop,
recording, and policy-eval code drives both:

  * ``get_observation()``  -> {"<motor>.pos": float (normalized), "<cam>": HxWx3 uint8}
  * ``send_action(dict)``  -> takes {"<motor>.pos": float (normalized)}
  * ``observation_features`` / ``action_features`` / ``name`` -> dataset feature spec
  * ``connect()`` / ``disconnect()``

Normalized units match the real arm (joints: -100..100, gripper: 0 closed .. 100
open). Cameras (``gripper`` wrist cam, ``desk`` overview) are rendered offscreen
from the MuJoCo model at the resolutions in config/cameras.yaml, so sim
observations are keyed and shaped identically to the real cameras.

SimRobot owns the MjModel / MjData; callers reuse ``.model`` / ``.data`` for the
viewer, the block, and physics stepping.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from .. import load_config

MODEL_PATH = Path(__file__).with_name("so101.xml")

# Joints in the order the controller / dataset expect.
REVOLUTE_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
FINGER_JOINTS = ["left_finger", "right_finger"]
MOTORS = REVOLUTE_JOINTS + ["gripper"]


class SimRobot:
    """Maps normalized joint commands to MuJoCo actuators; renders sim cameras."""

    name = "so101_sim"

    def __init__(self, use_cameras: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.use_cameras = use_cameras

        # Camera resolutions come from config/cameras.yaml so sim == real shapes.
        self._cam_cfg = load_config("cameras")["cameras"] if use_cameras else {}
        self._renderers: dict = {}  # built in connect()

        # Cache joint qpos addresses, ranges, and actuator ids by name.
        self._qadr: dict[str, int] = {}
        self._range: dict[str, tuple[float, float]] = {}
        self._act: dict[str, int] = {}
        for nm in REVOLUTE_JOINTS + FINGER_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)
            self._qadr[nm] = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self._range[nm] = (float(lo), float(hi))
            self._act[nm] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)

        mujoco.mj_forward(self.model, self.data)

    # -- lifecycle (match LeRobot Robot) ------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        if self.use_cameras:
            for cam, c in self._cam_cfg.items():
                self._renderers[cam] = mujoco.Renderer(self.model, height=c["height"], width=c["width"])

    def disconnect(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    # -- feature spec (consumed by lerobot.datasets.feature_utils) -----------
    @property
    def observation_features(self) -> dict:
        feats: dict = {f"{m}.pos": float for m in MOTORS}
        for cam, c in self._cam_cfg.items():
            feats[cam] = (c["height"], c["width"], 3)
        return feats

    @property
    def action_features(self) -> dict:
        return {f"{m}.pos": float for m in MOTORS}

    # -- normalization helpers ----------------------------------------------
    def _norm_to_angle(self, joint: str, n: float) -> float:
        lo, hi = self._range[joint]
        return lo + (n + 100.0) / 200.0 * (hi - lo)

    def _angle_to_norm(self, joint: str, a: float) -> float:
        lo, hi = self._range[joint]
        return (a - lo) / (hi - lo) * 200.0 - 100.0

    # -- observation / action ------------------------------------------------
    def get_observation(self) -> dict:
        obs: dict = {}
        for j in REVOLUTE_JOINTS:
            angle = float(self.data.qpos[self._qadr[j]])
            obs[f"{j}.pos"] = self._angle_to_norm(j, angle)
        lo, hi = self._range["left_finger"]
        pos = float(self.data.qpos[self._qadr["left_finger"]])
        obs["gripper.pos"] = (pos - lo) / (hi - lo) * 100.0

        for cam, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=cam)
            obs[cam] = renderer.render()  # HxWx3 uint8 RGB
        return obs

    def send_action(self, action: dict) -> dict:
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
        return action

    def set_pose(self, observation: dict) -> None:
        """Snap the sim joints to match normalized positions (visual mirror, no physics).

        Used to mirror the real arm: feed it ``real_robot.get_observation()`` and the
        sim shows the arm's live measured pose.
        """
        for j in REVOLUTE_JOINTS:
            key = f"{j}.pos"
            if key in observation:
                self.data.qpos[self._qadr[j]] = self._norm_to_angle(j, observation[key])
        if "gripper.pos" in observation:
            lo, hi = self._range["left_finger"]
            n = max(0.0, min(100.0, observation["gripper.pos"]))
            pos = lo + (n / 100.0) * (hi - lo)
            for f in FINGER_JOINTS:
                self.data.qpos[self._qadr[f]] = pos
        mujoco.mj_forward(self.model, self.data)
