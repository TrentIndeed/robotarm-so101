"""Build a connected SO-101 follower from config files.

Centralizes the LeRobot robot construction so both ``xbox_teleop`` and ``record``
use exactly the same arm + camera setup.
"""

from __future__ import annotations

from . import load_config
from .cameras import build_camera_configs


def build_robot(with_cameras: bool = True):
    """Create (but do not connect) an SO101Follower from config/robot.yaml.

    Set ``with_cameras=False`` for plain teleop where you don't need video — it
    starts faster and avoids holding the cameras open.
    """
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    rcfg = load_config("robot")
    cameras = build_camera_configs() if with_cameras else {}

    config = SO101FollowerConfig(
        port=rcfg["port"],
        id=rcfg["id"],
        cameras=cameras,
    )
    return SO101Follower(config)
