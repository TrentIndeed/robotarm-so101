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
import threading
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


class _CameraStream:
    """Background-threaded camera reader. The control loop never calls cap.read()
    directly, so a slow/disconnected camera can't stall it or feed corrupt frames —
    a dead camera just yields None (-> a 'no signal' placeholder) instead of blocking
    or, after a Windows index reshuffle, pulling some OTHER camera's frames."""

    def __init__(self, name, index, width, height, rotation):
        self.name = name
        self.index = int(index)
        self.width = width
        self.height = height
        self.rot = rotation
        self._latest = None
        self._run = True
        self._lock = threading.Lock()
        self._cap = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        import cv2

        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _loop(self):
        import cv2

        fails = 0
        while self._run:
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                fails = 0
                if self.rot is not None:
                    frame = cv2.rotate(frame, self.rot)
                with self._lock:
                    self._latest = frame
                continue

            fails += 1
            if fails >= 5:                       # gone -> show placeholder, not stale frames
                with self._lock:
                    self._latest = None
            # Reopen periodically so a brief disconnect recovers on its own: a stale
            # VideoCapture handle never reads again, but a fresh one (like a restart) does.
            if fails % 10 == 0:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = self._open()
            time.sleep(0.1)

    def read(self):
        with self._lock:
            return self._latest

    def release(self):
        self._run = False
        try:
            self._cap.release()
        except Exception:
            pass


def _open_cv_cameras():
    """Open the configured cameras as background-threaded streams (display only,
    decoupled from the robot AND from the control loop)."""
    from . import load_config
    from .cameras import _cv2_rotate_code

    streams = []
    for name, c in load_config("cameras")["cameras"].items():
        streams.append(_CameraStream(name, c["index_or_path"], c["width"], c["height"],
                                     _cv2_rotate_code(c.get("rotation"))))
    return streams


_MARGIN = 14  # px padding around each camera inside the panel


def _placeholder(name):
    """A black 'no signal' tile for a disconnected camera (keeps the slot, never
    shows another camera's frames)."""
    import cv2
    import numpy as np

    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, f"{name}: no signal", (40, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 220), 2)
    return img


def _read_caps(streams):
    """Grab the latest frame from each stream (non-blocking) -> list of (name, BGR
    frame), substituting a placeholder for any disconnected camera."""
    frames = []
    for s in streams:
        frame = s.read()
        frames.append((s.name, frame if frame is not None else _placeholder(s.name)))
    return frames


def _build_panel(frames, viewport):
    """Composite (name, BGR frame) tiles into ONE solid panel covering the right side
    of the viewer (black backing, so the sim never shows through). Returns a single
    (MjrRect, RGB) overlay; the viewer flips vertically itself."""
    import cv2
    import mujoco
    import numpy as np

    W, H = viewport.width, viewport.height
    if not frames or W < 2 or H < 2:
        return []
    n, m = len(frames), _MARGIN
    panel_w = min(W // 2, 820)            # right-side panel width (keep room for the sim)
    panel = np.zeros((H, panel_w, 3), np.uint8)
    cell_h = H // n
    for k, (name, frame) in enumerate(frames):
        fh, fw = frame.shape[:2]
        scale = min((panel_w - 2 * m) / fw, (cell_h - 2 * m) / fh)
        tw, th = max(1, int(fw * scale)), max(1, int(fh * scale))
        tile = cv2.resize(frame, (tw, th))
        cv2.putText(tile, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y0 = k * cell_h + (cell_h - th) // 2       # centered in its cell, top-down
        x0 = (panel_w - tw) // 2
        panel[y0:y0 + th, x0:x0 + tw] = tile
    rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    return [(mujoco.MjrRect(W - panel_w, 0, panel_w, H), rgb)]


def mirror_loop() -> None:
    """Teleop UI: drive the real arm with the Xbox controller, a 3D MuJoCo twin, the
    gripper + desk camera feeds on the right, and the MuJoCo menu/settings on the left.
    All serial I/O stays on this (main) thread — no key callbacks touching the bus."""
    import mujoco.viewer

    from .robot import make_robot
    from .sim.sim_robot import SimRobot

    # Joints-only robot -> connects fast and reliably (no camera warmup to hang on).
    real = make_robot(sim=False, use_cameras=False)
    sim = SimRobot(use_cameras=False)
    ctrl = XboxTeleopController()
    caps = _open_cv_cameras()   # robot cameras (threaded, auto-recovering)

    real.connect()
    ctrl.connect()
    ctrl.seed_targets(real.get_observation())
    print("Mirror UI — Xbox control. Back/View = hold. Ctrl+C or close the window to stop.")
    dt = ctrl.dt
    try:
        with mujoco.viewer.launch_passive(
            sim.model, sim.data, show_left_ui=True, show_right_ui=False
        ) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                real.send_action(ctrl.compute_action())
                sim.set_pose(real.get_observation())

                vp = viewer.viewport          # None during teardown -> skip overlays
                if vp is not None:
                    overlays = _build_panel(_read_caps(caps), vp)
                    if overlays:
                        viewer.set_images(overlays)
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        for stream in caps:
            stream.release()
        ctrl.disconnect()
        real.disconnect()


def vision_loop(cam_index: int) -> None:
    """Drive the real arm by tracking your arm/hand with a webcam (experimental)."""
    from .robot import make_robot
    from .vision_control import VisionController

    real = make_robot(sim=False, use_cameras=False)
    ctrl = VisionController(cam_index=cam_index)

    real.connect()
    ctrl.connect()
    ctrl.seed_targets(real.get_observation())
    print("Vision teleop running. SPACE in the feedback window = engage/disengage. Ctrl+C to stop.")
    dt = ctrl.dt
    try:
        while True:
            t0 = time.perf_counter()
            real.send_action(ctrl.compute_action())
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ctrl.disconnect()
        real.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Xbox teleop for the SO-101 follower")
    parser.add_argument("--debug", action="store_true", help="print controller axes/buttons and exit")
    parser.add_argument("--mirror", action="store_true",
                        help="drive the real arm with a 3D MuJoCo window mirroring it")
    parser.add_argument("--vision", action="store_true",
                        help="control by webcam arm/hand tracking instead of the pad (experimental)")
    parser.add_argument("--cam", type=int, default=2, help="operator webcam index for --vision")
    # Teleop doesn't use camera images, so cameras are OFF by default — opt in with
    # --cameras (only useful to confirm the cameras are wired before recording).
    parser.add_argument("--cameras", action="store_true", help="also open the cameras")
    args = parser.parse_args()

    if args.debug:
        debug_loop()
    elif args.vision:
        vision_loop(args.cam)
    elif args.mirror:
        mirror_loop()
    else:
        teleop_loop(with_cameras=args.cameras)


if __name__ == "__main__":
    main()
