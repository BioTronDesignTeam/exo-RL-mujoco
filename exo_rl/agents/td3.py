"""TD3 (Fujimoto et al., 2018) with a DDPG mode for comparison.

algo: td3   twin critics, clipped double-Q, target policy smoothing, delayed actor updates
algo: ddpg  single critic, no smoothing, actor updated every step
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @torch.no_grad()
    def act(self, obs, noise=0.0):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).cpu().numpy()[0]
        if noise > 0:
            a = a + np.random.normal(0, noise, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def update(self, buf: ReplayBuffer, batch_size: int) -> dict:
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
            actor_loss = -self.critic(o, self.actor(o))[0].mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - self.tau).add_(self.tau * p)
            info["actor_loss"] = actor_loss.item()
        return info

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(), "critic_t": self.critic_t.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "cfg": self.cfg, "obs_dim": self.obs_dim, "act_dim": self.act_dim, "updates": self.updates,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ck["obs_dim"], ck["act_dim"], ck["cfg"], device)
        for k in ("actor", "critic", "actor_t", "critic_t", "actor_opt", "critic_opt"):
            getattr(agent, k).load_state_dict(ck[k])
        agent.updates = ck["updates"]
        return agent
