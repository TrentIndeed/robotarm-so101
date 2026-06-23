"""Pick-and-place practice game for the SO-101.

A 3D arm sits on a desk. Pick up the block and drop it on the green target pad. Each
success scores a point and respawns the block somewhere new — so you can rehearse the
real task before any motors are wired up.

    python -m so101.sim.practice              # drive with the Xbox controller
    python -m so101.sim.practice --keyboard   # no controller? drive with the keyboard
    python -m so101.sim.practice --seed 0     # deterministic block positions

Xbox controls are identical to the real teleop (see config/teleop.yaml):
    left stick   shoulder pan / lift        right stick   wrist roll / elbow
    triggers     wrist flex down / up       A / B         gripper open / close
    Back/View    emergency hold

Keyboard controls (sim only):
    A/D W/S I/K J/L T/G  move joints        F   gripper toggle

In both modes, press R in the window to respawn the block. The live score and the
control map are shown as an on-screen HUD.
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

_FONT = mujoco.mjtFontScale.mjFONTSCALE_150
_TOPLEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT
_BOTTOMLEFT = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
_TOP = mujoco.mjtGridPos.mjGRID_TOP

# Two-column control legends (left = input, right = what it does).
_LEGEND = {
    "xbox": (
        "L-stick\nR-stick\nTriggers\nA / B\nBack/View\nR",
        "pan / lift\nroll / elbow\nwrist flex\ngripper\nhold\nrespawn",
    ),
    "keyboard": (
        "A/D  W/S\nI/K  J/L\nT / G\nF\nR",
        "pan  lift\nelbow  roll\nwrist flex\ngripper\nrespawn",
    ),
}


def _hud(score: int, mode: str, flash: str):
    """Build the overlay text: score top-left, controls bottom-left, flash on top."""
    left, right = _LEGEND[mode]
    items = [
        (_FONT, _TOPLEFT, "SO-101 practice\nScore", f"\n{score}"),
        (_FONT, _BOTTOMLEFT, left, right),
    ]
    if flash:
        items.append((_FONT, _TOP, flash, ""))
    return items


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


def run(keyboard: bool = False, seed: int | None = None) -> None:
    rng = random.Random(seed)
    mode = "keyboard" if keyboard else "xbox"

    robot = SimRobot()
    model, data = robot.model, robot.data
    block = Block(model, data)
    block.respawn(rng)

    if keyboard:
        from .keyboard_control import KeyboardController
        ctrl = KeyboardController()
    else:
        ctrl = XboxTeleopController()
    ctrl.connect()
    ctrl.seed_targets(robot.get_observation())

    # Viewer key handling: 'R' respawns; everything else goes to a keyboard controller.
    respawn_req = [False]

    def on_key(keycode):
        if keycode in (ord("R"), ord("r")):
            respawn_req[0] = True
        elif hasattr(ctrl, "on_key"):
            ctrl.on_key(keycode)

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

                flash = ""
                if cooldown > 0:
                    cooldown -= 1
                    flash = "Nice!  +1"
                elif block.on_target():
                    score += 1
                    cooldown = int(2 * ctrl.cfg["control_hz"])  # ~2 s lockout
                    print(f"Nice! Score: {score}")
                    block.respawn(rng)

                viewer.set_texts(_hud(score, mode, flash))
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.disconnect()
        print(f"\nDone. Final score: {score}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 pick-and-place practice sim")
    parser.add_argument("--keyboard", action="store_true",
                        help="drive with the keyboard instead of the Xbox controller")
    parser.add_argument("--seed", type=int, default=None, help="fix block spawn positions")
    args = parser.parse_args()
    run(keyboard=args.keyboard, seed=args.seed)


if __name__ == "__main__":
    main()
