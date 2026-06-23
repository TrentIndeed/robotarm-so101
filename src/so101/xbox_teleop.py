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


def teleop_loop(with_cameras: bool) -> None:
    from .robot import build_robot

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Xbox teleop for the SO-101 follower")
    parser.add_argument("--debug", action="store_true", help="print controller axes/buttons and exit")
    parser.add_argument("--no-cameras", action="store_true", help="skip opening cameras (faster)")
    args = parser.parse_args()

    if args.debug:
        debug_loop()
    else:
        teleop_loop(with_cameras=not args.no_cameras)


if __name__ == "__main__":
    main()
