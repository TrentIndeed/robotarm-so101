"""Fine-tune a LeRobot policy on a recorded SO-101 dataset (SmolVLA by default).

SmolVLA is a small vision-language-action model. It fine-tunes from the pretrained
``lerobot/smolvla_base`` (pulled from the Hub) and is language-conditioned on each
demo's task string — which the recorder already stores. ACT is also supported as a
lighter from-scratch baseline (no Hub download, trains on a weaker GPU/CPU).

    # fine-tune SmolVLA on a GPU (RunPod, etc.)
    python -m so101.train --policy smolvla --device cuda

    # just print the exact command (e.g. to paste into a RunPod shell)
    python -m so101.train --policy smolvla --print-only

    # lighter ACT baseline
    python -m so101.train --policy act --device cuda

Then evaluate the checkpoint with so101.run_policy (sim or real). See TRAINING.md for
the full RunPod walkthrough. SmolVLA needs a CUDA GPU with plenty of VRAM — lower
``--batch-size`` (e.g. 32 or 16) if you hit out-of-memory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import REPO_ROOT

DEFAULT_DATASET = "local/so101_pick_place"
SMOLVLA_BASE = "lerobot/smolvla_base"


def build_args(policy: str, dataset: str, device: str, steps: int,
               batch_size: int, output_dir: str) -> list[str]:
    """The lerobot-train CLI arguments for this run."""
    root = REPO_ROOT / "data" / dataset.replace("/", "__")
    args = [
        f"--dataset.repo_id={dataset}",
        f"--dataset.root={root}",
        f"--output_dir={output_dir}",
        f"--policy.device={device}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--wandb.enable=false",
    ]
    if policy == "smolvla":
        # Fine-tune from the pretrained SmolVLA base (downloaded from the Hub).
        args.append(f"--policy.path={SMOLVLA_BASE}")
    else:
        args.append(f"--policy.type={policy}")
    return args


def _train_exe() -> str:
    """lerobot-train sits next to this interpreter (venv Scripts/bin); fall back to PATH."""
    name = "lerobot-train" + (".exe" if os.name == "nt" else "")
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else "lerobot-train"


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a policy on an SO-101 dataset.")
    p.add_argument("--policy", choices=["smolvla", "act"], default="smolvla")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset repo-id (folder under data/)")
    p.add_argument("--device", default="cuda", help="cuda | cpu | mps")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64, help="lower (32/16) if you OOM")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--print-only", action="store_true", help="print the command, don't run it")
    a = p.parse_args()

    out = a.output_dir or str(REPO_ROOT / "outputs" / "train" / a.policy)
    cmd = [_train_exe(), *build_args(a.policy, a.dataset, a.device, a.steps, a.batch_size, out)]

    print("Training command:\n  " + " \\\n    ".join(cmd) + "\n")
    if a.print_only:
        return

    # so101/__init__ sets HF_HUB_OFFLINE=1 so local datasets never hit the Hub, but
    # SmolVLA must DOWNLOAD lerobot/smolvla_base — re-enable Hub access for the child.
    env = dict(os.environ, HF_HUB_OFFLINE="0")
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
