"""Record pick-and-place episodes into a LeRobot dataset, driving with the Xbox pad.

Works against either backend — the real SO-101 follower or the MuJoCo sim — via
``--sim``. Both produce byte-for-byte identical dataset formats (same
``observation.state`` / ``observation.images.*`` / ``action`` features), so a
policy can be trained on sim data and later run on the real arm unchanged.

    python -m so101.record --num-episodes 20            # real arm
    python -m so101.record --sim --num-episodes 20      # MuJoCo sim, no hardware

Each episode is one attempt at: *pick up the small object and place it at the target*.
Both cameras (gripper + desk), joint states, and commanded actions are saved in sync.

Controls while recording (button numbers from config/teleop.yaml):
    sticks/triggers/A/B   drive the arm
    Start/Menu            save the current episode and move to the next
    X                     discard and re-record the current episode
    Ctrl+C                stop and save the dataset as-is

The gripper + desk camera feeds are shown in OpenCV windows so you can see what the
policy will see (in sim, that's how you watch the arm). Data is written under
data/<repo-id> by default (git-ignored).
"""

from __future__ import annotations

import argparse
import random
import time

import mujoco

from . import REPO_ROOT
from .controller import XboxTeleopController
from .robot import make_robot

DEFAULT_TASK = "Pick up the small object and place it at the target."

# Block spawn region on the sim desk (matches so101.xml / sim.practice).
_SPAWN_X, _SPAWN_Y, _BLOCK_HALF = (0.14, 0.24), (-0.16, 0.16), 0.0125


def _reset_sim_block(robot, rng: random.Random) -> None:
    """Respawn the sim block at a random spot and settle it (sim mode only)."""
    m, d = robot.model, robot.data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
    qadr = m.jnt_qposadr[jid]
    x, y = rng.uniform(*_SPAWN_X), rng.uniform(*_SPAWN_Y)
    d.qpos[qadr:qadr + 7] = [x, y, _BLOCK_HALF, 1, 0, 0, 0]
    d.qvel[m.jnt_dofadr[jid]:m.jnt_dofadr[jid] + 6] = 0
    mujoco.mj_forward(m, d)


def _show(obs: dict, cam_names) -> None:
    """Display the camera feeds (RGB arrays) in OpenCV windows."""
    import cv2

    for cam in cam_names:
        if cam in obs:
            cv2.imshow(f"so101: {cam}", cv2.cvtColor(obs[cam], cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def record(num_episodes: int, task: str, repo_id: str, fps: int | None, sim: bool, display: bool) -> None:
    from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STR

    robot = make_robot(sim=sim, use_cameras=True)
    ctrl = XboxTeleopController()
    fps = fps or ctrl.cfg["control_hz"]
    btn = ctrl.cfg["buttons"]
    cam_names = [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]

    robot.connect()
    ctrl.connect()

    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
        **hw_to_dataset_features(robot.action_features, ACTION),
    }
    root = REPO_ROOT / "data" / repo_id.replace("/", "__")
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=root,
        robot_type=robot.name, use_videos=True,
    )

    rng = random.Random()
    dt = 1.0 / fps
    n_steps = max(1, round(dt / robot.model.opt.timestep)) if sim else 0

    print(f"Recording to {root}\nTask: {task}\nBackend: {'SIM' if sim else 'REAL'}\n")

    episode = 0
    try:
        while episode < num_episodes:
            print(f"--- Episode {episode + 1}/{num_episodes} — Start=save, X=re-record ---")
            if sim:
                _reset_sim_block(robot, rng)
            ctrl.seed_targets(robot.get_observation())

            cancelled = False
            while True:
                t0 = time.perf_counter()

                obs = robot.get_observation()
                action = ctrl.compute_action()
                robot.send_action(action)
                if sim:
                    robot.step(n_steps)
                if display:
                    _show(obs, cam_names)

                if ctrl.joystick.get_button(btn["episode_done"]):
                    break
                if ctrl.joystick.get_button(btn["episode_cancel"]):
                    cancelled = True
                    break

                obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**obs_frame, **act_frame, "task": task})

                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

            if cancelled:
                dataset.clear_episode_buffer()
                print("  re-recording this episode.\n")
                _wait_release(ctrl, btn["episode_cancel"])
            else:
                # Synchronous encoding: avoids Windows ProcessPool fragility; the
                # ~1-2 s encode between episodes is fine for a teleop workflow.
                dataset.save_episode(parallel_encoding=False)
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
    parser.add_argument("--sim", action="store_true", help="record from the MuJoCo sim instead of hardware")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--repo-id", default="local/so101_pick_place",
                        help="dataset id (also the folder name under data/)")
    parser.add_argument("--fps", type=int, default=None, help="defaults to teleop control_hz")
    parser.add_argument("--no-display", action="store_true", help="don't open camera preview windows")
    args = parser.parse_args()

    record(args.num_episodes, args.task, args.repo_id, args.fps, args.sim, not args.no_display)


if __name__ == "__main__":
    main()
