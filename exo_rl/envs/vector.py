"""Experience collection from one env (in-process) or several (subprocesses).

Both runners expose the same interface so train.py / pretrain_bc.py have one loop:

    obs, tasks = runner.reset()                    # obs: (num_envs, obs_dim)
    runner.step_async(actions)                     # actions: (num_envs, act_dim)
    ...                                            # learn here while the envs simulate
    obs, transitions = runner.step_wait()          # obs to act on next; one Transition per env

Transition.next_obs is always the TRUE next observation. When an env finished this step
it is the episode's final observation (not the first observation of the next episode,
which is what `obs` holds for that env), and Transition.reset_task names the new episode.

SingleRunner steps inside step_wait (overlap=False), so the caller keeps the classic
step -> store -> update order. VectorRunner uses gymnasium's AsyncVectorEnv with same-step
autoreset; step_async() returns immediately so updates overlap the simulation.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import gymnasium as gym
import numpy as np

from .exo_env import ExoAssistEnv


class Transition(NamedTuple):
    next_obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    reset_task: str | None          # task of the episode that started (only if done)


def make_env(cfg: dict, seed: int, lean_info: bool = True) -> ExoAssistEnv:
    """Top-level (picklable) env factory for subprocess workers."""
    return ExoAssistEnv(cfg, seed=seed, lean_info=lean_info)


def unbatch_info(infos: dict, i: int) -> dict:
    """Env i's entry of a gymnasium vector info dict (arrays + `_key` masks, nested)."""
    out = {}
    for k, v in infos.items():
        if k.startswith("_") or not infos.get(f"_{k}", np.ones(i + 1, bool))[i]:
            continue
        out[k] = unbatch_info(v, i) if isinstance(v, dict) else v[i]
    return out


class SingleRunner:
    overlap = False

    def __init__(self, cfg: dict, seed: int, lean_info: bool = True):
        self.env = make_env(cfg, seed, lean_info)
        self.num_envs = 1
        self._action = None

    def reset(self):
        obs, info = self.env.reset()
        return obs[None], [info["task"]]

    def sample_actions(self):
        return self.env.action_space.sample()[None]

    def step_async(self, actions):
        self._action = actions[0]

    def step_wait(self):
        obs2, r, term, trunc, info = self.env.step(self._action)
        tr = Transition(obs2, r, term, trunc, info, None)
        if term or trunc:
            obs2, reset_info = self.env.reset()
            tr = tr._replace(reset_task=reset_info["task"])
        return obs2[None], [tr]

    def close(self, terminate=False):
        pass


class VectorRunner:
    overlap = True

    def __init__(self, cfg: dict, num_envs: int, seed: int, lean_info: bool = True):
        fns = [functools.partial(make_env, cfg, seed + i, lean_info) for i in range(num_envs)]
        self.envs = gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.num_envs = num_envs

    def reset(self):
        obs, infos = self.envs.reset()          # each env keeps its constructor seed
        return obs, [str(t) for t in infos["task"]]

    def sample_actions(self):
        return self.envs.action_space.sample()

    def step_async(self, actions):
        self.envs.step_async(np.asarray(actions, dtype=np.float32))

    def step_wait(self):
        obs, rew, term, trunc, infos = self.envs.step_wait()
        out = []
        for i in range(self.num_envs):
            if term[i] or trunc[i]:
                out.append(Transition(infos["final_obs"][i], float(rew[i]), bool(term[i]), bool(trunc[i]),
                                      unbatch_info(infos["final_info"], i), str(infos["task"][i])))
            else:
                out.append(Transition(obs[i], float(rew[i]), False, False, unbatch_info(infos, i), None))
        return obs, out

    def close(self, terminate=False):
        self.envs.close(terminate=terminate)


def make_runner(cfg: dict, num_envs: int, seed: int, lean_info: bool = True):
    backend = cfg.get("train", {}).get("sim_backend", "cpu")
    if backend == "warp":
        from .warp_runner import WarpRunner
        return WarpRunner(cfg, num_envs, seed, lean_info)
    if backend != "cpu":
        raise ValueError(f"unknown train.sim_backend: {backend!r}; choose cpu or warp")
    if num_envs <= 1:
        return SingleRunner(cfg, seed, lean_info)
    return VectorRunner(cfg, num_envs, seed, lean_info)
