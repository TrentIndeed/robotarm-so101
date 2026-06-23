"""Map an Xbox controller to SO-101 joint commands.

Velocity / incremental control: each tick, stick & trigger values (after deadzone)
scale that joint's speed and integrate into a held target. Letting go holds the
joint. Targets stay clipped to LeRobot's normalized range so the arm can't be
driven past its limits.

The mapping (which axis drives which joint, speeds, deadzone) all comes from
``config/teleop.yaml`` so you can retune without touching code.
"""

from __future__ import annotations

from . import load_config

# Normalized position limits after LeRobot calibration.
JOINT_MIN, JOINT_MAX = -100.0, 100.0
GRIPPER_MIN, GRIPPER_MAX = 0.0, 100.0


def _deadzone(value: float, dz: float) -> float:
    """Zero out small stick noise; rescale the rest so motion starts smoothly."""
    if abs(value) < dz:
        return 0.0
    # Re-map [dz, 1] -> [0, 1] preserving sign, so there's no jump at the edge.
    return (value - dz * (1 if value > 0 else -1)) / (1 - dz)


class XboxTeleopController:
    """Reads a pygame joystick and produces SO-101 action dicts."""

    def __init__(self):
        self.cfg = load_config("teleop")
        self.dt = 1.0 / self.cfg["control_hz"]
        self.deadzone = self.cfg["deadzone"]
        self.global_speed = self.cfg["global_speed"]

        self._joint = self.cfg["joints"] if "joints" in self.cfg else load_config("robot")["joints"]

        self.joystick = None          # set in connect()
        self.targets: dict[str, float] = {}   # joint -> normalized target
        self._prev_buttons: dict[int, bool] = {}

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No controller detected. Plug in / turn on the Xbox controller and retry."
            )
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Controller: {self.joystick.get_name()}")

    def seed_targets(self, observation: dict) -> None:
        """Initialize targets from the arm's current pose so it never jumps.

        ``observation`` is what ``robot.get_observation()`` returns; joint
        positions live under ``"<joint>.pos"`` keys.
        """
        for joint in self._joint:
            key = f"{joint}.pos"
            self.targets[joint] = float(observation.get(key, 0.0))

    # -- per-tick ------------------------------------------------------------
    def _axis(self, idx: int) -> float:
        return _deadzone(self.joystick.get_axis(idx), self.deadzone)

    def _pressed(self, button: int) -> bool:
        """Edge-triggered: True only on the tick the button goes down."""
        now = bool(self.joystick.get_button(button))
        was = self._prev_buttons.get(button, False)
        self._prev_buttons[button] = now
        return now and not was

    def compute_action(self) -> dict:
        """Pump controller events and return the next ``{joint}.pos`` action dict."""
        import pygame

        pygame.event.pump()
        cfg = self.cfg

        # Emergency hold: keep current targets, ignore all sticks while held.
        estop = cfg.get("buttons", {}).get("emergency_stop")
        if estop is not None and self.joystick.get_button(estop):
            return {f"{j}.pos": v for j, v in self.targets.items()}

        # `speed` is in normalized-units-per-second at full deflection, so a per-tick
        # delta is value * speed * dt (scaled by the live global multiplier).
        gs = self.global_speed * self.dt

        # Stick-driven joints.
        for joint, m in cfg["axes"].items():
            v = self._axis(m["axis"])
            if m.get("invert"):
                v = -v
            self.targets[joint] = _clip(
                self.targets[joint] + v * m["speed"] * gs, JOINT_MIN, JOINT_MAX
            )

        # wrist_flex from the two triggers. Triggers rest near -1, press toward +1,
        # so normalize each into 0..1 of "how pressed".
        wf = cfg["wrist_flex"]
        up = (self.joystick.get_axis(wf["axis_up"]) + 1) / 2
        down = (self.joystick.get_axis(wf["axis_down"]) + 1) / 2
        self.targets["wrist_flex"] = _clip(
            self.targets["wrist_flex"] + (up - down) * wf["speed"] * gs,
            JOINT_MIN,
            JOINT_MAX,
        )

        # Gripper toggles open/closed on button edges.
        g = cfg["gripper"]
        if self._pressed(g["open_button"]):
            self.targets["gripper"] = float(g["open_pos"])
        elif self._pressed(g["close_button"]):
            self.targets["gripper"] = float(g["closed_pos"])

        return {f"{j}.pos": v for j, v in self.targets.items()}

    def disconnect(self) -> None:
        import pygame

        if self.joystick is not None:
            pygame.joystick.quit()
        pygame.quit()


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x
