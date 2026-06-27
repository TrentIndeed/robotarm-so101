"""SO-101 Xbox-teleoperated pick-and-place project."""

import os
from pathlib import Path

import yaml

# Datasets here are local-only (repo_id "local/..."), so never reach out to the
# HuggingFace Hub — otherwise loading/resuming a local dataset 401s. Override by
# exporting HF_HUB_OFFLINE=0 if you ever genuinely need the Hub from this process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Repo root = three levels up from this file (src/so101/__init__.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def load_config(name: str) -> dict:
    """Load config/<name>.yaml as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["REPO_ROOT", "CONFIG_DIR", "load_config"]
