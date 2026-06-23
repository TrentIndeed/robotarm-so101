"""Pick-and-place practice game for the SO-101, driven by the Xbox controller.

A 3D arm sits on a desk. Use the controller to pick up the block and drop it on the
green target pad. Each success scores a point and respawns the block somewhere new —
so you can rehearse the real task before any motors are wired up.

    python -m so101.sim.practice
    python -m so101.sim.practice --seed 0     # deterministic block positions

Controls are identical to the real teleop (see config/teleop.yaml):
    left stick   shoulder pan / lift        right stick   wrist roll / elbow
    triggers     wrist flex down / up       A / B         gripper open / close
    Back/View    emergency hold             R (keyboard)  respawn block

This uses the SAME XboxTeleopController as the hardware, so muscle memory transfers.
"""

from __future__ import annotations

import argparse
import random
import time

import pybullet as p
import pybullet_data

from ..controller import XboxTeleopController
from .sim_robot import SimRobot

# Workspace bounds on the desk top (meters) where the block may spawn.
SPAWN_X = (0.14, 0.24)
SPAWN_Y = (-0.16, 0.16)
DESK_TOP_Z = 0.0
BLOCK_HALF = 0.0125          # 2.5 cm cube
TARGET_XY = (0.20, 0.16)     # where to place the block
TARGET_RADIUS = 0.05


def _build_scene():
    """Floor, desk slab, target pad. Returns the target pad body id."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", basePosition=[0, 0, -0.4])

    # Desk slab: top surface sits at z = 0 (the arm base mounts here).
    desk_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.32, 0.2])
    desk_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.32, 0.2],
                                   rgbaColor=[0.55, 0.38, 0.24, 1])
    p.createMultiBody(0, desk_col, desk_vis, basePosition=[0.1, 0, -0.2])

    # Target pad: flat translucent green marker, visual only (no collision).
    pad_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[TARGET_RADIUS, TARGET_RADIUS, 0.002],
                                  rgbaColor=[0.1, 0.8, 0.2, 0.55])
    pad = p.createMultiBody(0, -1, pad_vis,
                            basePosition=[TARGET_XY[0], TARGET_XY[1], DESK_TOP_Z + 0.002])
    return pad


def _spawn_block(rng: random.Random, existing=None):
    """Create (or move) the pick-up block to a random spot, away from the target."""
    for _ in range(20):
        x = rng.uniform(*SPAWN_X)
        y = rng.uniform(*SPAWN_Y)
        if (x - TARGET_XY[0]) ** 2 + (y - TARGET_XY[1]) ** 2 > (2 * TARGET_RADIUS) ** 2:
            break
    z = DESK_TOP_Z + BLOCK_HALF
    if existing is not None:
        p.resetBasePositionAndOrientation(existing, [x, y, z], [0, 0, 0, 1])
        p.resetBaseVelocity(existing, [0, 0, 0], [0, 0, 0])
        return existing

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF] * 3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[BLOCK_HALF] * 3,
                              rgbaColor=[0.9, 0.2, 0.2, 1])
    block = p.createMultiBody(0.04, col, vis, basePosition=[x, y, z])
    p.changeDynamics(block, -1, lateralFriction=1.2)
    return block


def _on_target(block) -> bool:
    """True when the block is resting inside the target pad."""
    (bx, by, bz), _ = p.getBasePositionAndOrientation(block)
    lin, _ = p.getBaseVelocity(block)
    speed = (lin[0] ** 2 + lin[1] ** 2 + lin[2] ** 2) ** 0.5
    in_circle = (bx - TARGET_XY[0]) ** 2 + (by - TARGET_XY[1]) ** 2 < TARGET_RADIUS ** 2
    resting = bz < DESK_TOP_Z + BLOCK_HALF + 0.01 and speed < 0.03
    return in_circle and resting


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 Xbox pick-and-place practice sim")
    parser.add_argument("--seed", type=int, default=None, help="fix block spawn positions")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide PyBullet's side panels
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=55,
                                 cameraPitch=-35, cameraTargetPosition=[0.12, 0.04, 0.05])

    _build_scene()
    robot = SimRobot(base_position=(0, 0, 0))
    block = _spawn_block(rng)

    ctrl = XboxTeleopController()
    ctrl.connect()
    ctrl.seed_targets(robot.get_observation())

    hud = p.addUserDebugText("", [0.0, -0.28, 0.18], textColorRGB=[1, 1, 1], textSize=1.4)
    score = 0
    cooldown = 0
    dt = ctrl.dt
    # PyBullet's default internal timestep is 1/240 s; step enough to advance one
    # control tick (dt) of simulated time per loop.
    substeps = max(1, round(240 * dt))

    print("Practice sim running. Place the red block on the green pad. Close window or Ctrl+C to quit.")
    try:
        while True:
            t0 = time.perf_counter()

            action = ctrl.compute_action()
            robot.send_action(action)
            for _ in range(substeps):
                p.stepSimulation()

            # Keyboard R = manual respawn.
            keys = p.getKeyboardEvents()
            if ord("r") in keys and keys[ord("r")] & p.KEY_WAS_TRIGGERED:
                block = _spawn_block(rng, block)

            # Scoring with a short cooldown so one placement counts once.
            if cooldown > 0:
                cooldown -= 1
            elif _on_target(block):
                score += 1
                cooldown = int(2 * ctrl.cfg["control_hz"])  # ~2 s
                print(f"Nice! Score: {score}")
                block = _spawn_block(rng, block)

            hud = p.addUserDebugText(
                f"Score: {score}    [R] respawn   place red block on green pad",
                [0.0, -0.28, 0.18], textColorRGB=[1, 1, 1], textSize=1.4,
                replaceItemUniqueId=hud,
            )

            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except (KeyboardInterrupt, p.error):
        print(f"\nDone. Final score: {score}")
    finally:
        ctrl.disconnect()
        try:
            p.disconnect()
        except p.error:
            pass


if __name__ == "__main__":
    main()
