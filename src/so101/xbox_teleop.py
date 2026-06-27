"""Drive the SO-101 follower with an Xbox controller (no recording).

Usage:
    python -m so101.xbox_teleop            # connect arm + pad, start driving
    python -m so101.xbox_teleop --debug    # print live axis/button values, no arm
    python -m so101.xbox_teleop --no-cameras

Use ``--debug`` first to confirm which axis/button numbers your controller reports,
then set them in config/teleop.yaml. Once the sticks move the right joints, record
episodes with ``python -m so101.record``.
"""

from __future__ import annotations

import argparse
import time

from .controller import XboxTeleopController


def debug_loop() -> None:
    """Print every axis and button so you can fill in config/teleop.yaml."""
    import pygame

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No controller detected.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()}")
    print("Move sticks / press buttons. Ctrl+C to quit.\n")

    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            buttons = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            print(f"axes={axes}  buttons_down={buttons}        ", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        pygame.quit()


def teleop_loop(with_cameras: bool = False) -> None:
    from .robot import build_robot

    # Cameras default off — teleop only commands joints; opening them just risks
    # a camera error (and isn't needed until you record).
    robot = build_robot(with_cameras=with_cameras)
    ctrl = XboxTeleopController()

    robot.connect()
    ctrl.connect()
    try:
        # Seed targets from the current pose so the arm doesn't jump on the first tick.
        ctrl.seed_targets(robot.get_observation())
        print("Driving. Back/View button = emergency hold. Ctrl+C to stop.")

        dt = ctrl.dt
        while True:
            t0 = time.perf_counter()
            action = ctrl.compute_action()
            robot.send_action(action)
            # Keep a steady control rate.
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ctrl.disconnect()
        robot.disconnect()


def _open_cv_cameras():
    """Open the configured cameras directly via cv2 (display only, decoupled from the
    robot). A flaky camera then just shows black instead of hanging the robot's
    camera warmup. Returns a list of (name, VideoCapture, cv2_rotate_code)."""
    import cv2

    from . import load_config
    from .cameras import _cv2_rotate_code

    caps = []
    for name, c in load_config("cameras")["cameras"].items():
        cap = cv2.VideoCapture(int(c["index_or_path"]), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        caps.append((name, cap, _cv2_rotate_code(c.get("rotation"))))
    return caps


_TILE_W, _TILE_H, _TILE_M = 320, 240, 12  # camera overlay tile size + margin (px)


def _camera_overlays(caps, viewport):
    """Read each capture and build (MjrRect, RGB) overlays stacked down the right
    edge of the viewer. The viewer flips vertically itself, so pass top-down RGB."""
    import cv2
    import mujoco

    overlays = []
    k = 0
    for name, cap, rot in caps:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        frame = cv2.resize(frame, (_TILE_W, _TILE_H))
        cv2.putText(frame, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # set_images wants RGB
        left = max(0, viewport.width - _TILE_W - _TILE_M)
        bottom = max(0, viewport.height - _TILE_M - k * (_TILE_H + _TILE_M) - _TILE_H)
        overlays.append((mujoco.MjrRect(left, bottom, _TILE_W, _TILE_H), rgb))
        k += 1
    return overlays


def mirror_loop() -> None:
    """Drive the real arm with a 3D MuJoCo twin and the live camera feeds overlaid
    inside the viewer (down the right edge) — no separate window."""
    import mujoco.viewer

    from .robot import make_robot
    from .sim.sim_robot import SimRobot

    # Joints-only robot -> connects fast and reliably (no camera warmup to hang on).
    real = make_robot(sim=False, use_cameras=False)
    sim = SimRobot(use_cameras=False)
    ctrl = XboxTeleopController()
    caps = _open_cv_cameras()   # cameras opened separately, best-effort

    real.connect()
    ctrl.connect()
    ctrl.seed_targets(real.get_observation())
    print("Driving the REAL arm; 3D twin + camera feeds (right edge). Back/View = hold, "
          "Ctrl+C or close the window to stop.")
    dt = ctrl.dt
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                action = ctrl.compute_action()
                real.send_action(action)
                sim.set_pose(real.get_observation())   # mirror measured pose

                overlays = _camera_overlays(caps, viewer.viewport)
                if overlays:
                    viewer.set_images(overlays)
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        for _name, cap, _rot in caps:
            cap.release()
        ctrl.disconnect()
        real.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Xbox teleop for the SO-101 follower")
    parser.add_argument("--debug", action="store_true", help="print controller axes/buttons and exit")
    parser.add_argument("--mirror", action="store_true",
                        help="drive the real arm with a 3D MuJoCo window mirroring it")
    # Teleop doesn't use camera images, so cameras are OFF by default — opt in with
    # --cameras (only useful to confirm the cameras are wired before recording).
    parser.add_argument("--cameras", action="store_true", help="also open the cameras")
    args = parser.parse_args()

    if args.debug:
        debug_loop()
    elif args.mirror:
        mirror_loop()
    else:
        teleop_loop(with_cameras=args.cameras)


if __name__ == "__main__":
    main()
