"""Camera helpers for the SO-101 setup.

Two cameras are used:
  * ``gripper`` — wrist-mounted, close-up view of the grasp.
  * ``desk``    — fixed view of the whole workspace.

Run ``python -m so101.cameras --list`` to discover which OpenCV indices map to
which physical camera, then write the indices into ``config/cameras.yaml``.

``build_camera_configs()`` turns ``config/cameras.yaml`` into LeRobot
``OpenCVCameraConfig`` objects keyed by name, ready to hand to the robot config.
"""

from __future__ import annotations

import argparse

from . import load_config


def build_camera_configs() -> dict:
    """Build LeRobot OpenCV camera configs from config/cameras.yaml.

    Returns a dict like ``{"gripper": OpenCVCameraConfig(...), "desk": ...}``.
    """
    # Imported lazily so `--list` works even before lerobot is installed.
    from lerobot.cameras.opencv import OpenCVCameraConfig

    cfg = load_config("cameras")["cameras"]
    cameras: dict = {}
    for name, c in cfg.items():
        kwargs = dict(index_or_path=c["index_or_path"], width=c["width"],
                      height=c["height"], fps=c["fps"])
        # Optional per-camera rotation (0/90/180/270) and pixel format (e.g. MJPG).
        if c.get("rotation"):
            kwargs["rotation"] = c["rotation"]
        if c.get("fourcc"):
            kwargs["fourcc"] = c["fourcc"]
        cameras[name] = OpenCVCameraConfig(**kwargs)
    return cameras


def _cv2_rotate_code(rotation):
    """Map a config rotation (90/180/270/-90) to a cv2.rotate code, or None."""
    import cv2

    return {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation)


def list_cameras(max_index: int = 8) -> None:
    """Probe OpenCV camera indices 0..max_index and report which ones open."""
    import cv2

    print("Probing camera indices (this can take a few seconds)...\n")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DSHOW = fast/reliable on Windows
        if cap.isOpened():
            ok, frame = cap.read()
            shape = frame.shape if ok and frame is not None else "no frame"
            print(f"  index {i}: OPEN   ({shape})")
            found.append(i)
        cap.release()

    if not found:
        print("  No cameras found. Check USB connections / drivers.")
    else:
        print(
            f"\nFound indices: {found}\n"
            "Assign them to 'gripper' and 'desk' in config/cameras.yaml.\n"
            "Tip: cover one camera with your hand and re-run to tell them apart."
        )


def preview(name: str) -> None:
    """Open a live window for one configured camera (press 'q' to quit)."""
    import cv2

    cfg = load_config("cameras")["cameras"]
    if name not in cfg:
        raise SystemExit(f"Unknown camera '{name}'. Options: {list(cfg)}")

    c = cfg[name]
    cap = cv2.VideoCapture(int(c["index_or_path"]), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera '{name}' at index {c['index_or_path']}.")

    rot = _cv2_rotate_code(c.get("rotation"))  # match what recording will capture
    print(f"Previewing '{name}' — press 'q' in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rot is not None:
            frame = cv2.rotate(frame, rot)
        cv2.imshow(f"so101: {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 camera utilities")
    parser.add_argument("--list", action="store_true", help="probe and list camera indices")
    parser.add_argument("--preview", metavar="NAME", help="preview a configured camera (gripper/desk)")
    args = parser.parse_args()

    if args.preview:
        preview(args.preview)
    else:
        # Default action is --list; it's the most common thing you want.
        list_cameras()


if __name__ == "__main__":
    main()
