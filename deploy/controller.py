"""Deployment controller: sensors in, exo torque out.

Uses the SAME ObservationBuilder and ExoTorquePipeline classes as training, configured
from deploy_config.json, so the network sees identical inputs and its output goes through
identical feedforward / saturation / rate limiting (TODO 3). The network is fixed: no
learning happens on the device.

tests/test_consistency.py replays a simulated sensor log through this class and checks
that the torques match the training environment exactly.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from exo_rl.control.observation import ObservationBuilder
from exo_rl.control.torque_pipeline import ExoTorquePipeline, LimbParams


class DeployController:
    def __init__(self, policy, deploy_cfg: dict):
        """policy: TorchScript module / nn.Module / callable(np.ndarray obs) -> np.ndarray action."""
        self.cfg = deploy_cfg
        dt = float(deploy_cfg["control_dt"])
        self.obs_builder = ObservationBuilder(deploy_cfg["observation"], dt)
        self.pipeline = ExoTorquePipeline(deploy_cfg["exo"], LimbParams(**deploy_cfg["limb"]), dt)
        self.policy = policy
        if self.obs_builder.dim != deploy_cfg["obs_dim"]:
            raise ValueError(f"obs dim mismatch: builder {self.obs_builder.dim} vs policy {deploy_cfg['obs_dim']}")
        self.reset()

    @classmethod
    def from_export(cls, export_dir: str):
        with open(os.path.join(export_dir, "deploy_config.json")) as f:
            cfg = json.load(f)
        policy = torch.jit.load(os.path.join(export_dir, "policy.pt"), map_location="cpu")
        policy.eval()
        return cls(policy, cfg)

    def reset(self):
        self.obs_builder.reset()
        self.pipeline.reset()
        self.prev_action = np.zeros(4)

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(self.policy, torch.nn.Module) or isinstance(self.policy, torch.jit.ScriptModule):
            with torch.no_grad():
                a = self.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).numpy()[0]
        else:
            a = np.asarray(self.policy(obs))
        return np.clip(a, -1.0, 1.0)

    def step(self, sensors: dict) -> np.ndarray:
        """One control tick. Same order as ExoAssistEnv:
        obs_k = build(sensors_k, action_{k-1});  action_k = policy(obs_k);
        tau_k = pipeline(action_k, sensors_k)
        Returns torque for [hip_l, knee_l, hip_r, knee_r] (Nm, + = flexion).
        """
        obs = self.obs_builder(sensors, self.prev_action)
        action = self._infer(obs)
        tau = self.pipeline(action, np.asarray(sensors["q_exo"], dtype=float), float(sensors["trunk_pitch"]))
        self.prev_action = action
        return tau
