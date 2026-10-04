"""Warm start: a privileged teacher, behavior-cloned into the TD3 actor before RL fine-tuning.

In simulation we know the torque the human needs (env info["tau_needed"]). The teacher
supplies a fraction k of it:

    a_teacher = clip(k * tau_needed / action_scale, -1, 1)

Labels. The exo torque commanded after observation o_t acts during the following control
steps (rate limit, latency, and the human's one-step reaction), so the label for o_t uses
tau_needed from `label_offset_steps` steps after the step o_t led into (default 1): the
student learns to anticipate. The teacher that DRIVES the exo during data collection can
only use the past (the latest tau_needed), so it lags; the student can do better.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .agents.td3 import ReplayBuffer


def teacher_action(tau_needed, fraction: float, action_scale) -> np.ndarray:
    return np.clip(fraction * np.asarray(tau_needed) / np.asarray(action_scale, dtype=float), -1.0, 1.0)


def causal_teacher(env, fraction: float):
    """Policy for exo_rl.evaluation: acts on the env's latest tau_needed (not on the obs)."""
    scale = env.cfg["exo"]["action_scale"]
    return lambda obs: teacher_action(env.tau_needed, fraction, scale)


def label_episode(obs: list, tau_needed: list, fraction: float, action_scale, offset: int):
    """obs[j] -> teacher action from tau_needed[j + offset], where tau_needed[j] is from the
    step taken right after obs[j]. Trailing observations without a label are dropped."""
    n = len(tau_needed) - offset
    if n <= 0:
        return np.zeros((0, len(obs[0]) if obs else 0), np.float32), np.zeros((0, 4), np.float32)
    x = np.asarray(obs[:n], np.float32)
    y = teacher_action(np.asarray(tau_needed[offset:offset + n]), fraction, action_scale).astype(np.float32)
    return x, y


def train_bc(actor, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 256, lr: float = 1e-3,
             device="cpu") -> list:
    """Supervised regression of the actor onto teacher actions. Returns mean MSE per epoch."""
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(y, device=device)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        tot = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.mse_loss(actor(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        losses.append(tot / len(xt))
    return losses


@torch.no_grad()
def mse(actor, x: np.ndarray, y: np.ndarray, device="cpu") -> float:
    return float(F.mse_loss(actor(torch.as_tensor(x, device=device)), torch.as_tensor(y, device=device)))


class Collector:
    """Steps the envs (an exo_rl.envs.vector runner), executing the teacher's action mixed with
    the current BC policy, and turns trajectories into (observation, teacher label) pairs.
    Every transition is also stored in `buffer` (if given) to prefill RL later."""

    def __init__(self, runner, ws: dict, action_scale, buffer: ReplayBuffer | None = None):
        self.runner, self.scale, self.buffer = runner, action_scale, buffer
        self.fraction, self.offset = float(ws["teacher_fraction"]), int(ws["label_offset_steps"])
        self.noise = float(ws["collect_noise"])
        n = runner.num_envs
        self.obs, _ = runner.reset()
        self.last_tau = np.zeros((n, 4))                  # the teacher can only use the past
        self.hist = [([], []) for _ in range(n)]           # per env: observations, tau_needed

    def _flush(self, i, xs, ys):
        o, tau = self.hist[i]
        self.hist[i] = ([], [])
        if len(tau) <= self.offset:              # too short to label (e.g. just flushed)
            return
        x, y = label_episode(o, tau, self.fraction, self.scale, self.offset)
        xs.append(x)
        ys.append(y)

    def collect(self, n_steps: int, actor=None, mix: float = 0.0, device="cpu"):
        """About n_steps transitions; executed action = (1-mix)*teacher + mix*actor + noise."""
        xs, ys = [], []
        n = self.runner.num_envs
        for _ in range(int(np.ceil(n_steps / n))):
            a = teacher_action(self.last_tau, self.fraction, self.scale)
            if actor is not None and mix > 0:
                with torch.no_grad():
                    a_bc = actor(torch.as_tensor(self.obs, dtype=torch.float32, device=device)).cpu().numpy()
                a = (1 - mix) * a + mix * a_bc
            a = np.clip(a + np.random.normal(0, self.noise, a.shape), -1.0, 1.0)
            self.runner.step_async(a)
            obs2, trans = self.runner.step_wait()
            for i, t in enumerate(trans):
                if self.buffer is not None:
                    self.buffer.add(self.obs[i], a[i], t.reward, t.next_obs, t.terminated)
                self.hist[i][0].append(self.obs[i])
                self.hist[i][1].append(t.info["tau_needed"])
                self.last_tau[i] = t.info["tau_needed"]
                if t.terminated or t.truncated:
                    self._flush(i, xs, ys)
                    self.last_tau[i] = 0.0
            self.obs = obs2
        for i in range(n):                       # episodes continue next round; label what we have
            self._flush(i, xs, ys)
        return np.concatenate(xs), np.concatenate(ys)
