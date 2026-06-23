"""Pick-and-place practice game for the SO-101, driven by the Xbox controller.

A 3D arm sits on a desk. Use the controller to pick up the block and drop it on the
green target pad. Each success scores a point and respawns the block somewhere new —
so you can rehearse the real task before any motors are wired up.

    python -m so101.sim.practice
    python -m so101.sim.practice --seed 0     # deterministic block positions

Controls are identical to the real teleop (see config/teleop.yaml):
    left stick   shoulder pan / lift        right stick   wrist roll / elbow
    triggers     wrist flex down / up       A / B         gripper open / close
    Back/View    emergency hold             R (keyboard, in viewer)  respawn block

This uses the SAME XboxTeleopController as the hardware, so muscle memory transfers.
"""

from __future__ import annotations

import argparse
import random
import time

import mujoco
import mujoco.viewer

from ..controller import XboxTeleopController
from .sim_robot import SimRobot

# Workspace bounds on the desk top (meters) where the block may spawn.
SPAWN_X = (0.14, 0.24)
SPAWN_Y = (-0.16, 0.16)
DESK_TOP_Z = 0.0
BLOCK_HALF = 0.0125          # 2.5 cm cube (matches so101.xml)
TARGET_XY = (0.2, 0.16)      # matches the target site in so101.xml
TARGET_RADIUS = 0.05


class Block:
    """Helper around the free-floating block body in the model."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
        self.qadr = model.jnt_qposadr[jid]   # 7 values: xyz + quat
        self.vadr = model.jnt_dofadr[jid]    # 6 values: lin + ang vel
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    def respawn(self, rng: random.Random) -> None:
        for _ in range(20):
            x = rng.uniform(*SPAWN_X)
            y = rng.uniform(*SPAWN_Y)
            if (x - TARGET_XY[0]) ** 2 + (y - TARGET_XY[1]) ** 2 > (2 * TARGET_RADIUS) ** 2:
                break
        self.data.qpos[self.qadr:self.qadr + 7] = [x, y, DESK_TOP_Z + BLOCK_HALF, 1, 0, 0, 0]
        self.data.qvel[self.vadr:self.vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def on_target(self) -> bool:
        bx, by, bz = self.data.xpos[self.bid]
        vx, vy, vz = self.data.qvel[self.vadr:self.vadr + 3]
        speed = (vx * vx + vy * vy + vz * vz) ** 0.5
        in_circle = (bx - TARGET_XY[0]) ** 2 + (by - TARGET_XY[1]) ** 2 < TARGET_RADIUS ** 2
        resting = bz < DESK_TOP_Z + BLOCK_HALF + 0.01 and speed < 0.03
        return in_circle and resting


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 Xbox pick-and-place practice sim")
    parser.add_argument("--seed", type=int, default=None, help="fix block spawn positions")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    robot = SimRobot()
    model, data = robot.model, robot.data
    block = Block(model, data)
    block.respawn(rng)

    ctrl = XboxTeleopController()
    ctrl.connect()
    ctrl.seed_targets(robot.get_observation())

    # Viewer key handling: 'R' (keycode 82) requests a respawn on the main loop.
    respawn_req = [False]

    def on_key(keycode):
        if keycode in (ord("R"), ord("r")):
            respawn_req[0] = True

    score = 0
    cooldown = 0
    dt = ctrl.dt
    n_steps = max(1, round(dt / model.opt.timestep))

    print("Practice sim running. Place the red block on the green pad.")
    print("Close the viewer window or press Ctrl+C to quit.\n")

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()

                action = ctrl.compute_action()
                robot.send_action(action)
                for _ in range(n_steps):
                    mujoco.mj_step(model, data)

                if respawn_req[0]:
                    respawn_req[0] = False
                    block.respawn(rng)

                if cooldown > 0:
                    cooldown -= 1
                elif block.on_target():
                    score += 1
                    cooldown = int(2 * ctrl.cfg["control_hz"])  # ~2 s lockout
                    print(f"Nice! Score: {score}")
                    block.respawn(rng)

                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.disconnect()
        print(f"\nDone. Final score: {score}")


if __name__ == "__main__":
    main()
