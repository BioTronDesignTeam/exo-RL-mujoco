"""Experimental batched CUDA physics with a CPU control bridge.

Physics integrates in MuJoCo Warp, not in CPU subprocesses. Controllers,
reference trajectories, observations and rewards remain on the host. This
bridge is a compatibility starting point, not a GPU-resident high-throughput
environment. It deliberately rejects model randomization: all worlds share
one nominal model. GPU trajectories need not match CPU trajectories exactly.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .vector import Transition, make_env


class WarpRunner:
    overlap = False

    def __init__(self, cfg, num_envs, seed, lean_info=True):
        if num_envs < 1:
            raise ValueError("train.num_envs must be positive")
        if cfg.get("randomization", {}).get("enabled", False):
            raise ValueError("warp currently requires randomization.enabled=false (shared nominal model)")
        try:
            import warp as wp
            import mujoco_warp as mjw
        except ImportError as exc:
            raise ValueError("warp backend requires requirements-gpu.txt and an NVIDIA CUDA GPU") from exc
        wp.init()
        self.device = cfg["train"].get("warp_device", "cuda:0")
        try:
            device = wp.get_device(self.device)
        except Exception as exc:
            raise ValueError(f"Warp CUDA device unavailable: {self.device}") from exc
        if not device.is_cuda:
            raise ValueError("warp simulation requires a CUDA device; CPU fallback is disabled")
        self.wp, self.mjw = wp, mjw
        self.num_envs = num_envs
        self.envs = [make_env(cfg, seed + i, lean_info) for i in range(num_envs)]
        self.options = cfg["train"]
        self.actions = None
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.envs[0].hm.model)
            self.data = mjw.make_data(
                self.envs[0].hm.model, nworld=num_envs,
                nconmax=int(self.options.get("warp_nconmax", 128)),
                njmax=int(self.options.get("warp_njmax", 512)),
            )

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.stack([r[0] for r in results]), [r[1]["task"] for r in results]

    def sample_actions(self):
        return np.stack([env.action_space.sample() for env in self.envs])

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4):
            raise ValueError(f"expected actions {(self.num_envs, 4)}, got {actions.shape}")
        self.actions = actions.copy()

    def _physics(self):
        # Upload integration state and applied forces for all worlds together.
        # Warmstart is kept per world, and reset MjData clears it independently.
        fields = ("qpos", "qvel", "qacc_warmstart", "qfrc_applied")
        for field in fields:
            getattr(self.data, field).assign(np.stack([
                getattr(env.hm.data, field) for env in self.envs
            ]).astype(np.float32))
        self.data.time.assign(np.array([env.hm.data.time for env in self.envs], np.float32))
        self.mjw.step(self.model, self.data)
        # Read every substep: host controllers require the updated state.
        overflow = self.data.overflow.numpy()
        if np.any(overflow):
            raise RuntimeError("MuJoCo Warp buffer overflow; increase train.warp_nconmax/warp_njmax")
        values = {name: getattr(self.data, name).numpy() for name in
                  ("qpos", "qvel", "qacc_warmstart", "sensordata", "time")}
        for i, env in enumerate(self.envs):
            d = env.hm.data
            d.qpos[:] = values["qpos"][i]
            d.qvel[:] = values["qvel"][i]
            d.time = float(values["time"][i])
            # Refresh CPU mass/bias terms needed by HumanController.mj_mulM.
            # This forward evaluation does not integrate CPU physics.
            mujoco.mj_forward(env.hm.model, d)
            d.qacc_warmstart[:] = values["qacc_warmstart"][i]
            d.sensordata[:] = values["sensordata"][i]

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError("call step_async before step_wait")
        steps = [env.step_substeps(action) for env, action in zip(self.envs, self.actions)]
        with self.wp.ScopedDevice(self.device):
            for _ in range(self.envs[0].frame_skip):
                for step in steps:
                    next(step)
                self._physics()
        obs, transitions = [], []
        for env, step in zip(self.envs, steps):
            try:
                next(step)
            except StopIteration as done:
                o, r, term, trunc, info = done.value
            else:
                raise RuntimeError("environment substep count mismatch")
            transition = Transition(o, r, term, trunc, info, None)
            if term or trunc:
                o, reset_info = env.reset()
                transition = transition._replace(reset_task=reset_info["task"])
            obs.append(o)
            transitions.append(transition)
        self.actions = None
        return np.stack(obs), transitions

    def close(self, terminate=False):
        for env in self.envs:
            env.close()
