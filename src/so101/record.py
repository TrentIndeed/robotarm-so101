"""Record pick-and-place episodes into a LeRobot dataset, driving with the Xbox pad.

Works against either backend — the real SO-101 follower or the MuJoCo sim — via
``--sim``. Both produce byte-for-byte identical dataset formats (same
``observation.state`` / ``observation.images.*`` / ``action`` features), so a
policy can be trained on sim data and later run on the real arm unchanged.

    python -m so101.record --num-episodes 20            # real arm
    python -m so101.record --sim --num-episodes 20      # MuJoCo sim, no hardware

Each episode is one attempt at: *pick up the small object and place it at the target*.
Both cameras (gripper + desk), joint states, and commanded actions are saved in sync.

Controls (button numbers from config/teleop.yaml):
    sticks/triggers/A/B   drive the arm (works in idle too)
    Start/Menu            begin recording; press again to stop + save the episode
    X                     discard the take you're currently recording
    Ctrl+C                stop and save the dataset as-is

Frames are only captured between Start (begin) and Start (stop). While idle you can
reposition the object / return the arm home without it being recorded.

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


def _show(obs: dict, cam_names, recording: bool) -> None:
    """Display the camera feeds with a REC / idle indicator."""
    import cv2

    label = "* REC" if recording else "idle"
    color = (0, 0, 255) if recording else (0, 200, 0)
    for cam in cam_names:
        if cam in obs:
            img = cv2.cvtColor(obs[cam], cv2.COLOR_RGB2BGR)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow(f"so101: {cam}", img)
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

    # Edge-triggered button reads — compute_action() pumps pygame events each tick.
    prev: dict[int, bool] = {}

    def pressed(b: int) -> bool:
        now = bool(ctrl.joystick.get_button(b))
        was = prev.get(b, False)
        prev[b] = now
        return now and not was

    print(f"Recording to {root}\nTask: {task}\nBackend: {'SIM' if sim else 'REAL'}\n")
    print("Start = begin recording, Start again = stop + save. X = discard current take.")
    print("Between takes (idle) you can reposition the object — that is NOT recorded.\n")

    if sim:
        _reset_sim_block(robot, rng)
    ctrl.seed_targets(robot.get_observation())

    episode = 0
    recording = False
    try:
        while episode < num_episodes:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            action = ctrl.compute_action()
            robot.send_action(action)
            if sim:
                robot.step(n_steps)

            # Start toggles recording on/off (off -> save). X discards the active take.
            if pressed(btn["episode_done"]):
                if not recording:
                    recording = True
                    print(f"* REC  episode {episode + 1}/{num_episodes} ...")
                else:
                    dataset.save_episode(parallel_encoding=False)  # sync = robust on Windows
                    episode += 1
                    recording = False
                    print(f"  saved ({episode}/{num_episodes}). Reposition, then Start for the next.\n")
                    if sim and episode < num_episodes:
                        _reset_sim_block(robot, rng)
            elif recording and pressed(btn["episode_cancel"]):
                dataset.clear_episode_buffer()
                recording = False
                print("  discarded. Start to retry.\n")
                if sim:
                    _reset_sim_block(robot, rng)

            # Only capture frames while actively recording (idle/reposition is skipped).
            if recording:
                obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
                dataset.add_frame({**obs_frame, **act_frame, "task": task})

            if display:
                _show(obs, cam_names, recording)

            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        # REQUIRED: flushes buffered episode metadata + parquet footers. Without it
        # the dataset is invalid (can't be viewed or trained on).
        dataset.finalize()
        ctrl.disconnect()
        robot.disconnect()
        print(f"\nDone. {episode} episode(s) saved + finalized in {root}")


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
