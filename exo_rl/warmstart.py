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
