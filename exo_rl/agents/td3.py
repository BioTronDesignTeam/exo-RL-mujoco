"""TD3 (Fujimoto et al., 2018) with a DDPG mode for comparison.

algo: td3   twin critics, clipped double-Q, target policy smoothing, delayed actor updates
algo: ddpg  single critic, no smoothing, actor updated every step

Fine-tuning from a warm start (train.py --init-from) can regularize the actor toward a frozen
copy of its initial weights (pi_bc), as in TD3+BC (Fujimoto & Gu, 2021):
    actor_loss = -Q(s, pi(s)) / mean|Q| + w * ||pi(s) - pi_bc(s)||^2
and can freeze the actor while the fresh critic catches up. Without set_bc_reference() the
loss is plain -Q, exactly as before.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import atomic_write


def mlp(inp, out, hidden, out_act=None):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden, nn.Tanh())

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, twin=True):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden) if twin else None

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), (self.q2(x) if self.q2 is not None else None)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.not_done = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def __len__(self):
        return self.size if self.full else self.ptr

    def add(self, o, a, r, o2, terminated):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i] = o, a, r, o2
        self.not_done[i] = 0.0 if terminated else 1.0   # truncation still bootstraps
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, n, device):
        idx = np.random.randint(0, len(self), n)
        f = lambda x: torch.as_tensor(x[idx], device=device)
        return f(self.obs), f(self.act), f(self.rew), f(self.next_obs), f(self.not_done)


class TD3Agent:
    def __init__(self, obs_dim, act_dim, cfg: dict, device="cpu"):
        self.cfg = cfg
        self.algo = cfg.get("algo", "td3").lower()
        assert self.algo in ("td3", "ddpg")
        self.device = torch.device(device)
        hidden = cfg.get("hidden", [256, 256])
        twin = self.algo == "td3"
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden, twin).to(self.device)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg["actor_lr"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(cfg["critic_lr"]))
        self.gamma, self.tau = float(cfg["gamma"]), float(cfg["tau"])
        self.policy_noise = float(cfg["policy_noise"]) if twin else 0.0
        self.noise_clip = float(cfg["noise_clip"])
        self.policy_delay = int(cfg["policy_delay"]) if twin else 1
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.updates = 0
        self.actor_bc = None             # frozen reference policy for BC regularization

    def set_bc_reference(self):
        """Freeze a copy of the current actor as pi_bc (normalizes the Q term from now on)."""
        self.actor_bc = copy.deepcopy(self.actor).requires_grad_(False)

    @torch.no_grad()
    def act(self, obs, noise=0.0):
        """obs: (obs_dim,) -> (act_dim,), or a batch (n, obs_dim) -> (n, act_dim)."""
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        single = o.ndim == 1
        a = self.actor(o.unsqueeze(0) if single else o).cpu().numpy()
        if noise > 0:
            a = a + np.random.normal(0, noise, size=a.shape)
        a = np.clip(a, -1.0, 1.0)
        return a[0] if single else a

    def update(self, buf: ReplayBuffer, batch_size: int, actor_frozen: bool = False, bc_weight: float = 0.0) -> dict:
        o, a, r, o2, nd = buf.sample(batch_size, self.device)
        with torch.no_grad():
            a2 = self.actor_t(o2)
            if self.policy_noise > 0:
                eps = (torch.randn_like(a2) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a2 = (a2 + eps).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_t(o2, a2)
            q_t = torch.min(q1_t, q2_t) if q2_t is not None else q1_t
            y = r + nd * self.gamma * q_t
        q1, q2 = self.critic(o, a)
        critic_loss = F.mse_loss(q1, y) + (F.mse_loss(q2, y) if q2 is not None else 0.0)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": critic_loss.item(), "q_mean": q1.mean().item()}
        self.updates += 1
        if self.updates % self.policy_delay == 0:
            if not actor_frozen:
                pi = self.actor(o)
                q = self.critic(o, pi)[0]
                if self.actor_bc is None:
                    actor_loss = -q.mean()
                else:
                    actor_loss = -q.mean() / q.abs().mean().detach().clamp_min(1e-6)
                    if bc_weight > 0:
                        with torch.no_grad():
                            a_bc = self.actor_bc(o)
                        actor_loss = actor_loss + bc_weight * ((pi - a_bc) ** 2).sum(-1).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                info["actor_loss"] = actor_loss.item()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p)
        return info

    def save(self, path, step: int | None = None):
        """Atomic write; `step` (env steps, optional) is metadata for tools like watch_training."""
        ck = {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(), "critic_t": self.critic_t.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "cfg": self.cfg, "obs_dim": self.obs_dim, "act_dim": self.act_dim, "updates": self.updates,
            "step": step, "actor_bc": self.actor_bc.state_dict() if self.actor_bc is not None else None,
        }
        atomic_write(str(path), lambda tmp: torch.save(ck, tmp))

    def init_from(self, path):
        """Copy actor, critic and their targets from a checkpoint (e.g. bc.pt) to fine-tune.
        Optimizers stay fresh. Raises ValueError if the network shapes don't match."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if (ck["obs_dim"], ck["act_dim"]) != (self.obs_dim, self.act_dim):
            raise ValueError(
                f"--init-from {path}: checkpoint has obs {ck['obs_dim']} / act {ck['act_dim']}, but this "
                f"config gives obs {self.obs_dim} / act {self.act_dim}. The `observation` settings "
                f"(history, use_phase, include_prev_action) must match the checkpoint's.")
        try:
            for k in ("actor", "critic", "actor_t", "critic_t"):
                getattr(self, k).load_state_dict(ck[k])
        except RuntimeError as e:
            raise ValueError(f"--init-from {path}: network shapes differ (agent.hidden / agent.algo must "
                             f"match the checkpoint's: hidden={ck['cfg'].get('hidden')}, "
                             f"algo={ck['cfg'].get('algo')}).\n{e}") from None

    @classmethod
    def load(cls, path, device="cpu"):
        """path: file path or file-like object. agent.step = env steps at save time (or None)."""
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ck["obs_dim"], ck["act_dim"], ck["cfg"], device)
        for k in ("actor", "critic", "actor_t", "critic_t", "actor_opt", "critic_opt"):
            getattr(agent, k).load_state_dict(ck[k])
        agent.updates = ck["updates"]
        agent.step = ck.get("step")
        if ck.get("actor_bc") is not None:
            agent.set_bc_reference()
            agent.actor_bc.load_state_dict(ck["actor_bc"])
        return agent
