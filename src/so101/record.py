"""Record pick-and-place episodes into a LeRobot dataset, driving with the Xbox pad.

Each episode is one attempt at: *pick up the small object and place it at the target*.
Both cameras (gripper + desk), the joint states, and the commanded actions are saved
in sync at the control rate, ready for policy training.

Usage:
    python -m so101.record --num-episodes 20

Controls while recording (button numbers from config/teleop.yaml):
    sticks/triggers/A/B   drive the arm (same as teleop)
    Start/Menu            save the current episode and move to the next
    X                     discard and re-record the current episode
    Ctrl+C                stop and save the dataset as-is

The dataset is written under data/<repo-id> by default (git-ignored).

NOTE: the LeRobot dataset API evolves between releases. This uses the
``build_dataset_frame`` / ``hw_to_dataset_features`` building blocks from the
record pipeline; if your installed lerobot version differs, adjust the imports here.
"""

from __future__ import annotations

import argparse
import time

from . import REPO_ROOT
from .controller import XboxTeleopController
from .robot import build_robot

DEFAULT_TASK = "Pick up the small object and place it at the target."


def record(num_episodes: int, task: str, repo_id: str, fps: int | None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

    robot = build_robot(with_cameras=True)
    ctrl = XboxTeleopController()
    fps = fps or ctrl.cfg["control_hz"]
    btn = ctrl.cfg["buttons"]

    robot.connect()
    ctrl.connect()

    # Build dataset feature spec from the robot's own observation/action signatures.
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    act_features = hw_to_dataset_features(robot.action_features, "action")
    features = {**obs_features, **act_features}

    root = REPO_ROOT / "data" / repo_id.replace("/", "__")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot.name,
        root=root,
        use_videos=True,
    )

    print(f"Recording to {root}")
    print(f"Task: {task}\n")

    dt = 1.0 / fps
    try:
        episode = 0
        while episode < num_episodes:
            print(f"--- Episode {episode + 1}/{num_episodes} — Start=save, X=re-record ---")
            ctrl.seed_targets(robot.get_observation())

            cancelled = False
            while True:
                t0 = time.perf_counter()

                obs = robot.get_observation()
                action = ctrl.compute_action()
                robot.send_action(action)

                # Episode control buttons (read after compute_action pumped events).
                if ctrl.joystick.get_button(btn["episode_done"]):
                    break
                if ctrl.joystick.get_button(btn["episode_cancel"]):
                    cancelled = True
                    break

                obs_frame = build_dataset_frame(dataset.features, obs, prefix="observation")
                act_frame = build_dataset_frame(dataset.features, action, prefix="action")
                dataset.add_frame({**obs_frame, **act_frame}, task=task)

                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

            if cancelled:
                dataset.clear_episode_buffer()
                print("  re-recording this episode.\n")
                _wait_release(ctrl, btn["episode_cancel"])
            else:
                dataset.save_episode()
                episode += 1
                print("  saved.\n")
                _wait_release(ctrl, btn["episode_done"])
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        ctrl.disconnect()
        robot.disconnect()
        print(f"\nDone. {episode} episode(s) in {root}")


def _wait_release(ctrl, button: int) -> None:
    """Block until a button is let go, so one press isn't read twice."""
    import pygame

    while True:
        pygame.event.pump()
        if not ctrl.joystick.get_button(button):
            return
        time.sleep(0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record SO-101 pick-and-place episodes")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--repo-id",
        default="local/so101_pick_place",
        help="dataset id (also the folder name under data/)",
    )
    parser.add_argument("--fps", type=int, default=None, help="defaults to teleop control_hz")
    args = parser.parse_args()

    record(args.num_episodes, args.task, args.repo_id, args.fps)


if __name__ == "__main__":
    main()
