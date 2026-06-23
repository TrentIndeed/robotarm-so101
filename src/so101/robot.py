"""Build a robot backend — real SO-101 follower or the MuJoCo sim — from config.

``make_robot(sim=...)`` is the single swap point: the rest of the project
(teleop, recording, policy eval) treats both backends identically because they
share the same ``get_observation`` / ``send_action`` / feature interface.

Real arm config comes from config/robot.yaml + config/cameras.yaml.
"""

from __future__ import annotations

from . import load_config
from .cameras import build_camera_configs


def make_robot(sim: bool = False, use_cameras: bool = True):
    """Return a connected-capable robot backend (not yet connected).

    sim=True  -> MuJoCo SimRobot (no hardware needed).
    sim=False -> real LeRobot SO-101 follower (Feetech bus + USB cameras).
    use_cameras=False skips cameras (faster; for plain teleop without recording).
    """
    if sim:
        from .sim.sim_robot import SimRobot
        return SimRobot(use_cameras=use_cameras)

    # Real hardware (lerobot 0.5.x: SO-100/101 unified under SOFollower).
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    rcfg = load_config("robot")
    cameras = build_camera_configs() if use_cameras else {}
    config = SOFollowerRobotConfig(port=rcfg["port"], id=rcfg["id"], cameras=cameras)
    return SO101Follower(config)


# Backwards-compatible alias.
def build_robot(with_cameras: bool = True):
    return make_robot(sim=False, use_cameras=with_cameras)
