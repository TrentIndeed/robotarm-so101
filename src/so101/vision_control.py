"""Control the SO-101 by watching your arm with a webcam (experimental).

A separate webcam points at YOU. MediaPipe tracks your arm and hand:

  * **Arm-puppet** (when your arm is in frame): your shoulder/elbow drive the robot's
    shoulder_pan / shoulder_lift / elbow_flex.
  * **Hand-pointer fallback** (when the arm isn't reliably tracked): your hand's
    position in the frame drives pan / lift.
  * **Gripper**: pinch (thumb-to-index) opens/closes it in both modes.

A **clutch** (press SPACE in the feedback window) engages/disengages control so you
can reposition without the robot drifting — control uses relative motion from the
moment you engage, so engaging never makes the arm jump. Heavy smoothing tames jitter.

This produces the same normalized ``{joint}.pos`` actions as the Xbox controller, so it
plugs into the same teleop loop. It's experimental and less precise than the pad —
great for play/gross motion, but keep the controller for recording training data.

The mapping constants below are rough; calibrate them live (the feedback window shows
the tracked skeleton, the mode, the clutch state, and each joint target).
"""

from __future__ import annotations

import urllib.request

import numpy as np

from . import REPO_ROOT, load_config
from .controller import GRIPPER_MAX, GRIPPER_MIN, JOINT_MAX, JOINT_MIN, _clip

# MediaPipe model bundles (downloaded once into models/).
_MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

# Pose landmark indices (MediaPipe), by side: (shoulder, elbow, wrist).
ARM_LANDMARKS = {"left": (11, 13, 15), "right": (12, 14, 16)}
# Hand landmark indices.
H_WRIST, H_THUMB_TIP, H_INDEX_TIP, H_MIDDLE_MCP = 0, 4, 8, 9

# ---- Tunables: calibrate these live ----------------------------------------
PAN_RANGE = 0.28      # |wrist_x - shoulder_x| (frac of width) that maps to full pan
LIFT_RANGE = 0.28     # (shoulder_y - elbow_y) (frac of height) that maps to full lift
ELBOW_MIN_DEG, ELBOW_MAX_DEG = 45.0, 165.0   # your elbow angle -> robot elbow extremes
PINCH_CLOSED, PINCH_OPEN = 0.35, 1.1         # thumb-index dist / hand-scale -> grip
EMA = 0.30            # smoothing factor (higher = snappier, lower = smoother)
VIS_THRESH = 0.6      # min landmark visibility to trust the arm (puppet mode)
# ----------------------------------------------------------------------------

VISION_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex"]   # gripper handled separately


# ---- pure helpers (unit-testable, no camera) -------------------------------
def angle_deg(a, b, c) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees (a,b,c are 3-vectors)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    cos = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def lin_map(v, in_lo, in_hi, out_lo, out_hi) -> float:
    """Linearly map v from [in_lo,in_hi] to [out_lo,out_hi], clipped to the output."""
    if in_hi == in_lo:
        return out_lo
    t = (v - in_lo) / (in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def _ema(old: float, new: float, a: float = EMA) -> float:
    return (1 - a) * old + a * new


class VisionController:
    """Webcam arm/hand tracking -> normalized SO-101 joint targets."""

    def __init__(self, cam_index: int = 2, show_window: bool = True, arm: str = "left"):
        self.cfg = load_config("teleop")
        self.dt = 1.0 / self.cfg["control_hz"]
        self.cam_index = cam_index
        self.show_window = show_window          # own cv2 window (standalone) vs headless
        self.arm = arm if arm in ARM_LANDMARKS else "left"
        self._joints = load_config("robot")["joints"]
        self.targets: dict[str, float] = {}
        self.last_frame = None                  # latest annotated frame (for embedding)

        self.cap = None
        self.pose = None
        self.hands = None
        self.engaged = False          # clutch (start disengaged for safety)
        self._ref = None              # reference raw values captured on engage
        self._base = None             # robot targets captured on engage

    @property
    def _arm_idx(self):
        return ARM_LANDMARKS[self.arm]

    # -- lifecycle (matches XboxTeleopController) ----------------------------
    def connect(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        _ensure_models()
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open operator webcam at index {self.cam_index}. "
                "Pass a different --cam index (it must be a camera pointed at you)."
            )
        models = REPO_ROOT / "models"
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models / "hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_hands=1))
        self._mp = mp
        print("Vision control ready. SPACE in the feedback window = engage/disengage clutch.")

    def seed_targets(self, observation: dict) -> None:
        for j in self._joints:
            self.targets[j] = float(observation.get(f"{j}.pos", 0.0))

    def disconnect(self) -> None:
        import cv2

        if self.cap is not None:
            self.cap.release()
        for lm in (self.pose, self.hands):
            if lm is not None:
                lm.close()
        cv2.destroyAllWindows()

    # -- per-tick ------------------------------------------------------------
    def compute_action(self) -> dict:
        import cv2

        ok, frame = self.cap.read()
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)   # mirror so motion feels natural
            raw, mode, overlay = self._track(frame)
            self._apply(raw)
            self.last_frame = self._draw(overlay, mode)   # annotated frame (BGR)
            if self.show_window:
                cv2.imshow("so101 vision control", self.last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    self.toggle_clutch()
                elif key in (ord("a"), ord("A")):
                    self.swap_arm()
        return {f"{j}.pos": v for j, v in self.targets.items()}

    # -- internals -----------------------------------------------------------
    def _track(self, frame):
        """Return (raw_targets|None, mode_str, frame_for_overlay)."""
        import cv2

        rgb = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pres = self.pose.detect(rgb)
        hres = self.hands.detect(rgb)

        raw: dict[str, float] = {}
        mode = "HOLD"

        # Gripper from pinch (works in both modes whenever a hand is seen).
        if hres.hand_landmarks:
            hl = hres.hand_landmarks[0]
            scale = _dist(hl[H_WRIST], hl[H_MIDDLE_MCP]) + 1e-6
            pinch = _dist(hl[H_THUMB_TIP], hl[H_INDEX_TIP]) / scale
            raw["gripper"] = lin_map(pinch, PINCH_CLOSED, PINCH_OPEN, GRIPPER_MIN, GRIPPER_MAX)

        # Arm-puppet if the chosen arm is reliably visible, else hand-pointer.
        si, ei, wi = self._arm_idx
        arm_ok = (pres.pose_landmarks and pres.pose_world_landmarks
                  and _visible(pres.pose_landmarks[0], (si, ei, wi), VIS_THRESH))
        if arm_ok:
            mode = f"PUPPET ({self.arm})"
            lm = pres.pose_landmarks[0]
            sh, el, wr = lm[si], lm[ei], lm[wi]
            raw["shoulder_pan"] = lin_map(wr.x - sh.x, -PAN_RANGE, PAN_RANGE, JOINT_MIN, JOINT_MAX)
            raw["shoulder_lift"] = lin_map(sh.y - el.y, -LIFT_RANGE, LIFT_RANGE, JOINT_MIN, JOINT_MAX)
            wl = pres.pose_world_landmarks[0]
            ang = angle_deg(_xyz(wl[si]), _xyz(wl[ei]), _xyz(wl[wi]))
            raw["elbow_flex"] = lin_map(ang, ELBOW_MIN_DEG, ELBOW_MAX_DEG, JOINT_MIN, JOINT_MAX)
        elif hres.hand_landmarks:
            mode = "POINTER"
            hl = hres.hand_landmarks[0]
            cx = sum(p.x for p in hl) / len(hl)
            cy = sum(p.y for p in hl) / len(hl)
            raw["shoulder_pan"] = lin_map(cx - 0.5, -0.35, 0.35, JOINT_MIN, JOINT_MAX)
            raw["shoulder_lift"] = lin_map(0.5 - cy, -0.35, 0.35, JOINT_MIN, JOINT_MAX)

        return (raw or None), mode, (frame, pres, hres)

    def _apply(self, raw):
        """Blend raw targets into self.targets (relative-on-engage + EMA)."""
        if not self.engaged or raw is None:
            self._ref = None
            return
        # Capture references on the engage edge so there's no jump.
        if self._ref is None:
            self._ref = dict(raw)
            self._base = {j: self.targets[j] for j in VISION_JOINTS}
        for j in VISION_JOINTS:
            if j in raw and j in self._ref:
                tgt = self._base[j] + (raw[j] - self._ref[j])    # relative motion
                self.targets[j] = _clip(_ema(self.targets[j], tgt), JOINT_MIN, JOINT_MAX)
        if "gripper" in raw:                                      # gripper is absolute
            self.targets["gripper"] = _clip(_ema(self.targets["gripper"], raw["gripper"]),
                                            GRIPPER_MIN, GRIPPER_MAX)

    def toggle_clutch(self) -> None:
        self.engaged = not self.engaged
        self._ref = None
        print("Clutch", "ENGAGED" if self.engaged else "disengaged")

    def swap_arm(self) -> None:
        self.arm = "right" if self.arm == "left" else "left"
        self._ref = None
        print("Tracking", self.arm, "arm")

    def _draw(self, overlay, mode):
        import cv2

        frame, pres, hres = overlay
        h, w = frame.shape[:2]
        # Arm skeleton (the tracked side).
        if pres.pose_landmarks:
            lm = pres.pose_landmarks[0]
            pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self._arm_idx]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], (0, 220, 0), 3)
            for p in pts:
                cv2.circle(frame, p, 6, (0, 220, 0), -1)
        # Pinch line.
        if hres.hand_landmarks:
            hl = hres.hand_landmarks[0]
            t = (int(hl[H_THUMB_TIP].x * w), int(hl[H_THUMB_TIP].y * h))
            i = (int(hl[H_INDEX_TIP].x * w), int(hl[H_INDEX_TIP].y * h))
            cv2.line(frame, t, i, (0, 180, 255), 2)
        clutch = "ENGAGED" if self.engaged else "OFF (SPACE)"
        col = (0, 0, 255) if self.engaged else (180, 180, 180)
        cv2.putText(frame, f"{mode}  clutch:{clutch}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        # Target bars.
        for n, j in enumerate(VISION_JOINTS + ["gripper"]):
            v = self.targets.get(j, 0.0)
            lo = GRIPPER_MIN if j == "gripper" else JOINT_MIN
            hi = GRIPPER_MAX if j == "gripper" else JOINT_MAX
            frac = (v - lo) / (hi - lo)
            y = 60 + n * 26
            cv2.rectangle(frame, (10, y), (210, y + 16), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, y), (10 + int(200 * frac), y + 16), (0, 200, 0), -1)
            cv2.putText(frame, j, (220, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame


# ---- module helpers --------------------------------------------------------
def _xyz(lm):
    return (lm.x, lm.y, lm.z)


def _dist(p, q) -> float:
    return float(((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5)


def _visible(landmarks, idxs, thresh) -> bool:
    return all(getattr(landmarks[i], "visibility", 1.0) >= thresh for i in idxs)


def _ensure_models() -> None:
    models = REPO_ROOT / "models"
    models.mkdir(exist_ok=True)
    for name, url in _MODELS.items():
        path = models / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
