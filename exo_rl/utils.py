from __future__ import annotations

import copy
import os
import time

import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply overrides like ["agent.algo=ddpg", "train.total_steps=5000"]."""
    with open(path) as f:
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
