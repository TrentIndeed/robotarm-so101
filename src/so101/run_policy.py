"""Run a trained LeRobot/PyTorch policy on either backend — sim or real arm.

This is the payoff of the shared interface: train one policy, then evaluate it in
the MuJoCo sim or on the physical SO-101 by flipping a single flag.

    # evaluate in sim (no hardware)
    python -m so101.run_policy --checkpoint outputs/train/.../pretrained_model --dataset local/so101_pick_place --sim

    # run the same policy on the real arm
    python -m so101.run_policy --checkpoint outputs/train/.../pretrained_model --dataset local/so101_pick_place

The policy consumes exactly the observation features both backends emit
(observation.state + observation.images.gripper/desk) and produces a 6-DOF action
that maps straight back to "<joint>.pos" — so no per-backend glue is needed.

NOTE: this entrypoint is written against the lerobot 0.5.x inference API
(predict_action + make_robot_action + make_policy/make_pre_post_processors) and is
exercised once you have a trained checkpoint. Until then there's nothing to run.
"""

from __future__ import annotations

import argparse
import time

from . import REPO_ROOT
from .robot import make_robot

DEFAULT_TASK = "Pick up the small object and place it at the target."


def run(checkpoint: str, dataset_repo_id: str, sim: bool, task: str, hz: float, display: bool) -> None:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.feature_utils import hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.device_utils import get_safe_torch_device

    # --- load the policy + its normalization processors from the checkpoint ---
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint
    ds_root = REPO_ROOT / "data" / dataset_repo_id.replace("/", "__")
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=ds_root)

    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, pretrained_path=checkpoint, dataset_stats=ds_meta.stats
    )
    device = get_safe_torch_device(policy_cfg.device)
    policy.eval()

    # --- the backend: identical interface either way ---
    robot = make_robot(sim=sim, use_cameras=True)
    cam_names = [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]
    ds_features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
        **hw_to_dataset_features(robot.action_features, ACTION),
    }

    robot.connect()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    dt = 1.0 / hz
    n_steps = max(1, round(dt / robot.model.opt.timestep)) if sim else 0
    print(f"Running policy on {'SIM' if sim else 'REAL'} arm. Ctrl+C to stop.")

    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            action_tensor = predict_action(
                obs, policy, device, preprocessor, postprocessor,
                use_amp=False, task=task, robot_type=robot.name,
            )
            robot.send_action(make_robot_action(action_tensor, ds_features))
            if sim:
                robot.step(n_steps)
            if display:
                _show(obs, cam_names)

            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()


def _show(obs: dict, cam_names) -> None:
    import cv2

    for cam in cam_names:
        if cam in obs:
            cv2.imshow(f"so101: {cam}", cv2.cvtColor(obs[cam], cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def _latest_checkpoint() -> str | None:
    """Newest trained checkpoint under outputs/train/<policy>/checkpoints/<step>/pretrained_model."""
    cps = sorted((REPO_ROOT / "outputs" / "train").glob("*/checkpoints/*/pretrained_model"),
                 key=lambda p: p.stat().st_mtime)
    return str(cps[-1]) if cps else None


def run_from_launcher(argv: list[str]) -> None:
    """`./run policy [--real] [--checkpoint ...]` — run the latest policy with sane defaults."""
    p = argparse.ArgumentParser(prog="./run policy",
                                description="Run your latest trained policy (sim by default).")
    p.add_argument("--checkpoint", default=None, help="default: newest under outputs/train/")
    p.add_argument("--dataset", default=None, help="for normalization stats; default: your recording dataset")
    p.add_argument("--real", action="store_true", help="run on the REAL arm (default: MuJoCo sim)")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args(argv)

    ckpt = a.checkpoint or _latest_checkpoint()
    if not ckpt:
        raise SystemExit("No trained checkpoint found under outputs/train/. Train one first "
                         "(see TRAINING.md) or pass --checkpoint.")
    dataset = a.dataset
    if not dataset:
        try:
            import json
            dataset = json.loads((REPO_ROOT / ".app_settings.json").read_text()).get("rec_repo")
        except Exception:
            dataset = None
        dataset = dataset or "local/so101_pick_place"

    print(f"Policy:     {ckpt}")
    print(f"Dataset:    {dataset}  (normalization stats)")
    print(f"Backend:    {'REAL arm — keep a hand near the e-stop' if a.real else 'MuJoCo sim'}")
    run(ckpt, dataset, sim=not a.real, task=a.task, hz=a.hz, display=not a.no_display)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained policy on the SO-101 (sim or real)")
    parser.add_argument("--checkpoint", required=True, help="path to a trained policy (pretrained_model dir)")
    parser.add_argument("--dataset", required=True, help="training dataset repo-id (for normalization stats)")
    parser.add_argument("--sim", action="store_true", help="run in the MuJoCo sim instead of on hardware")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--hz", type=float, default=30.0, help="control loop rate")
    parser.add_argument("--no-display", action="store_true", help="don't open camera preview windows")
    args = parser.parse_args()

    run(args.checkpoint, args.dataset, args.sim, args.task, args.hz, not args.no_display)


if __name__ == "__main__":
    main()
