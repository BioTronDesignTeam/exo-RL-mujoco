from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import yaml


def resolve_config_path(path: str | Path) -> Path:
    """Resolve archived config names used by historical commands and run protocols."""
    target = Path(path)
    root = Path(__file__).resolve().parents[1] / "configs"
    manifest = root / "archive" / "manifest.json"
    if target.is_file() or target.resolve().parent != root or not manifest.is_file():
        return target
    archived = json.loads(manifest.read_text(encoding="utf-8-sig"))["legacy_paths"].get(target.name)
    return root / archived if archived is not None else target


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply overrides like ["agent.algo=ddpg", "train.total_steps=5000"]."""
    with open(resolve_config_path(path)) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, value = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def pick_device(name: str = "auto") -> str:
    import torch
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_torch_threads(requested: int, num_envs: int) -> int:
    """Leave cores for env worker processes. 0 = auto; with one env torch's default is kept."""
    import torch
    if requested <= 0 and num_envs > 1:
        # >= 2: the learner is the bottleneck; 1 thread was slower even with 8 busy workers
        requested = max(2, min(torch.get_num_threads(), (os.cpu_count() or 2) - num_envs))
    if requested > 0:
        torch.set_num_threads(requested)
    return torch.get_num_threads()


def deepcopy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def atomic_write(path: str, write) -> None:
    """write(tmp_path), then rename over `path`, so readers never see a partial file.
    On Windows the rename fails while another process has `path` open; retry briefly."""
    tmp = f"{path}.tmp"
    write(tmp)
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, path)
