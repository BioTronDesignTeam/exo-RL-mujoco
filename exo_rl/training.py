"""The TD3/DDPG training loop behind scripts/train.py.

One iteration of Trainer.run():

    1. act       random actions during the start_steps warm-up, else policy + exploration noise
    2. simulate  every env takes one control step (in worker processes when num_envs > 1)
    3. learn     updates_per_step gradient updates per collected transition; with parallel envs
                 this runs WHILE the envs simulate (step_async ... step_wait)
    4. store     transitions go into the replay buffer; finished episodes go to episodes.csv
    5. housekeep evaluation (eval.csv, best.pt), latest.pt / step_N.pt, --max-minutes stop

With the default config (num_envs=1, updates_per_step=1.0) training is reproducible bit for bit:
one env, step -> store -> update, a fixed order of random draws. CONTRIBUTING.md ("Defaults
must not change") explains how to check that a change keeps it that way.

Warm start (train.py --init-from, usually a bc.pt from pretrain_bc.py): see WarmStartSchedule.
"""
from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from .agents.td3 import ReplayBuffer, TD3Agent
from .envs.exo_env import ExoAssistEnv
from .envs.vector import make_runner
from .evaluation import FinishedEval, PeriodicEval, eval_score
from .utils import atomic_write, configure_torch_threads, pick_device


class CsvLog:
    """Append-friendly CSV writer: the header comes from the first row (or the existing file)."""

    def __init__(self, path, append=False):
        self.path, self.writer, self.f = path, None, None
        self.append = append and os.path.exists(path)

    def write(self, row: dict):
        if self.writer is None:
            fields = list(row)
            if self.append:
                with open(self.path) as f:
                    fields = next(csv.reader(f))
            self.f = open(self.path, "a" if self.append else "w", newline="")
            self.writer = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore", restval="")
            if not self.append:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.f.flush()


def crossed(prev: int, cur: int, every: int) -> bool:
    """True if a multiple of `every` lies in (prev, cur]; step counts advance by num_envs."""
    return cur // every > prev // every


@dataclass
class WarmStartSchedule:
    """Protects an initialized policy from the still-untrained critic (only with --init-from).

    Counted in env steps after learning starts (after start_steps):
      [0, freeze_steps)                      only the critic learns
      then for bc_decay_steps                actor loss = -Q/mean|Q| + w * ||pi - pi_init||^2,
                                             w decays linearly from bc_weight to bc_weight_min
    A floor above 0 keeps a small pull toward the initial policy for the rest of training;
    with 0, RL alone took over and performance dropped once w reached 0.
    """
    freeze_steps: int = 0
    bc_weight: float = 0.0
    bc_decay_steps: int = 0
    bc_weight_min: float = 0.0

    @classmethod
    def from_config(cls, agent_cfg: dict, active: bool) -> "WarmStartSchedule":
        if not active:
            return cls()
        return cls(int(agent_cfg.get("actor_freeze_steps", 0)), float(agent_cfg.get("bc_weight", 0.0)),
                   int(agent_cfg.get("bc_decay_steps", 0)), float(agent_cfg.get("bc_weight_min", 0.0)))

    def at(self, learning_steps: int) -> tuple[bool, float]:
        """(actor_frozen, bc_weight) after this many env steps of learning."""
        u = learning_steps - self.freeze_steps          # steps since the actor was unfrozen
        if self.bc_decay_steps > 0:
            frac = min(1.0, max(0.0, 1.0 - u / self.bc_decay_steps))
            w = self.bc_weight_min + (self.bc_weight - self.bc_weight_min) * frac
        else:
            w = self.bc_weight
        return u <= 0, w


class EpisodeLog:
    """Per-env running sums; writes one episodes.csv row per finished episode."""

    def __init__(self, csv_log: CsvLog, num_envs: int, tasks: list, first_episode: int = 0):
        self.log, self.tasks, self.episode = csv_log, list(tasks), first_episode
        self.ret, self.len = np.zeros(num_envs), np.zeros(num_envs, int)
        self.terms = [defaultdict(float) for _ in range(num_envs)]

    def add(self, step: int, transitions, upd_info: dict, steps_per_s: float) -> bool:
        """Accumulate one control step of every env. Returns True if any episode finished."""
        any_done = False
        for i, t in enumerate(transitions):
            self.ret[i] += t.reward
            self.len[i] += 1
            for k, v in t.info["reward_terms"].items():
                self.terms[i][k] += v
            if t.terminated or t.truncated:
                any_done = True
                self._finish(i, step, t, upd_info, steps_per_s)
        return any_done

    def _finish(self, i, step, t, upd_info, steps_per_s):
        row = {"step": step, "episode": self.episode, "env_id": i, "task": self.tasks[i],
               "return": round(float(self.ret[i]), 3), "length": int(self.len[i]),
               "terminated": int(t.terminated), "reason": t.info["terminated_reason"]}
        row.update({f"mean_{k}": round(v / self.len[i], 5) for k, v in self.terms[i].items()})
        row.update({k: round(v, 4) for k, v in upd_info.items()})
        self.log.write(row)
        if self.episode % 10 == 0:
            print(f"[train] step {step:>8d} ep {self.episode:>5d} ret {self.ret[i]:7.1f} len {self.len[i]:4d} "
                  f"effort {row['mean_effort']:.3f} ({self.tasks[i]}) {steps_per_s:.0f} steps/s")
        self.tasks[i] = t.reset_task
        self.ret[i], self.len[i], self.terms[i] = 0.0, 0, defaultdict(float)
        self.episode += 1


class Trainer:
    """Builds everything for one run directory and trains. See scripts/train.py for the CLI."""

    def __init__(self, cfg: dict, run_dir: str, name: str, resume: bool = False,
                 init_from: str | None = None, init_buffer: str | None = None):
        self.cfg, self.run_dir, self.name = cfg, run_dir, name
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.snap_dir = os.path.join(run_dir, "resume")
        tr, ag = cfg["train"], cfg["agent"]

        # Keep this order: seeding and object construction consume random numbers, so
        # reordering them changes training results.
        seed = int(cfg.get("seed", 0))
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = pick_device(tr.get("device", "auto"))
        self.num_envs = int(tr.get("num_envs", 1))
        self.updates_per_step = float(tr.get("updates_per_step", 1.0))
        threads = configure_torch_threads(int(tr.get("torch_threads", 0)), self.num_envs)

        eval_env = ExoAssistEnv(cfg, seed=seed + 1)
        obs_dim, act_dim = eval_env.observation_space.shape[0], eval_env.action_space.shape[0]
        self.agent = TD3Agent(obs_dim, act_dim, ag, device)
        self.buf = ReplayBuffer(obs_dim, act_dim, int(ag["buffer_size"]))
        self.ev_log = CsvLog(os.path.join(run_dir, "eval.csv"), append=resume)
        ep_csv = CsvLog(os.path.join(run_dir, "episodes.csv"), append=resume)
        print(f"[train] {self.agent.algo.upper()} on {device} | obs {obs_dim} act {act_dim} | "
              f"{self.num_envs} env(s), {self.updates_per_step:g} updates/step, {threads} torch threads | "
              f"run dir {run_dir}")

        # Where training starts from: scratch, a checkpoint (+ prefilled buffer), or a snapshot.
        self.start, self.best, baseline, episode = 1, -np.inf, None, 0
        self.init_from, self.prefilled = init_from, 0
        if init_from and not resume:
            self.agent.init_from(init_from)                  # raises ValueError on shape mismatch
            if float(ag.get("bc_weight", 0.0)) > 0:
                self.agent.set_bc_reference()
            print(f"[train] initialized actor/critic from {init_from}")
        if init_buffer and not resume:
            self.prefilled = self.buf.add_from_file(init_buffer)
            print(f"[train] prefilled the replay buffer with {self.prefilled} transitions from {init_buffer}")
        if resume:
            st = self._load_snapshot(device)
            self.start, episode, self.best = st["step"] + 1, st["episode"], st["best"]
            baseline = st.get("baseline")
            self.init_from = st.get("init_from")             # keeps the warm-start schedule
            self.prefilled = st.get("prefilled", 0)
            np.random.seed(seed + self.start)
            print(f"[train] resumed at step {self.start} (buffer {len(self.buf)})")
        self.runner = make_runner(cfg, self.num_envs, seed + (self.start if resume else 0))  # new seeds on resume

        self.start_steps, self.total_steps = int(tr["start_steps"]), int(tr["total_steps"])
        if self.prefilled >= int(ag["batch_size"]):
            self.start_steps = 0                             # the prefilled buffer replaces the warm-up
        self.batch_size, self.expl = int(ag["batch_size"]), float(ag["expl_noise"])
        self.warm = WarmStartSchedule.from_config(ag, active=bool(self.init_from))
        self.upd_info = {"critic_loss": float("nan"), "q_mean": float("nan"), "actor_loss": float("nan")}
        self.upd_acc = 0.0
        if self.init_from:
            self.upd_info["bc_weight"] = self.warm.bc_weight
            print(f"[train] warm start: actor frozen for {self.warm.freeze_steps} steps, then BC weight "
                  f"{self.warm.bc_weight:g} -> {self.warm.bc_weight_min:g} over {self.warm.bc_decay_steps} "
                  f"steps; start_steps collected with the initial policy")

        self.best_metric = tr.get("best_metric", "mean")
        self.evaluator = PeriodicEval(cfg, eval_env, cfg["tasks"]["names"], int(tr["eval_episodes"]),
                                      self.ckpt_dir, async_=bool(tr.get("eval_async", False)))
        self.evaluator.baseline = baseline
        self.episodes = EpisodeLog(ep_csv, self.num_envs, [], episode)

    def run(self, max_minutes: float | None = None):
        """Train until train.total_steps, max_minutes (then snapshot for --resume), or Ctrl+C."""
        step = self.start - 1                                # transitions collected so far (all envs)
        obs, self.episodes.tasks = self.runner.reset()
        t0 = time.time()
        rate = lambda: (step - self.start + 1) / (time.time() - t0)  # noqa: E731
        try:
            while step < self.total_steps:
                actions = self._act(obs, step)
                self.runner.step_async(actions)
                if self.runner.overlap:
                    self._learn(step)                        # envs simulate while we learn
                obs2, trans = self.runner.step_wait()
                prev, step = step, step + self.num_envs
                for i, t in enumerate(trans):
                    self.buf.add(obs[i], actions[i], t.reward, t.next_obs, t.terminated)
                if not self.runner.overlap:
                    self._learn(step)                        # classic order: step -> store -> update
                obs = obs2

                any_done = self.episodes.add(step, trans, self.upd_info, rate())
                self._housekeeping(prev, step)

                # A single env stops only at an episode boundary; parallel envs may stop mid-episode.
                if max_minutes and time.time() - t0 > 60 * max_minutes and (self.runner.overlap or any_done):
                    print(f"[train] {step - self.start + 1} steps in {(time.time() - t0) / 60:.1f} min "
                          f"({rate():.0f} steps/s incl. evaluation)")
                    self._report(self.evaluator.poll(wait=True))
                    self._save_snapshot(step)
                    return
            self._report(self.evaluator.poll(wait=True))
        except KeyboardInterrupt:
            self.runner.close(terminate=True)
            self.runner = None
            self.agent.save(os.path.join(self.ckpt_dir, "latest.pt"), step)
            print(f"[train] interrupted at step {step}; saved latest.pt (no resume snapshot)")
            return
        finally:
            self.evaluator.close()
            if self.runner is not None:
                self.runner.close()

        self.agent.save(os.path.join(self.ckpt_dir, "latest.pt"), step)
        print(f"[train] done in {(time.time() - t0) / 60:.1f} min ({rate():.0f} steps/s incl. evaluation)")

    def _act(self, obs, step):
        if step + 1 <= self.start_steps and not self.init_from:
            return self.runner.sample_actions()              # random warm-up, independent per env
        return self.agent.act(obs, noise=self.expl)          # exploration noise drawn per env

    def _learn(self, n_collected: int):
        """This iteration's share of gradient updates: updates_per_step per collected transition."""
        if n_collected <= self.start_steps:
            return
        self.upd_acc += self.updates_per_step * self.num_envs
        n_upd = int(self.upd_acc)
        self.upd_acc -= n_upd
        frozen, bc_w = self.warm.at(n_collected - self.start_steps)
        for _ in range(n_upd):
            self.upd_info.update(self.agent.update(self.buf, self.batch_size, actor_frozen=frozen, bc_weight=bc_w))
        if self.init_from:
            self.upd_info["bc_weight"] = bc_w

    def _housekeeping(self, prev: int, step: int):
        """Evaluation and checkpoints whose interval was crossed by this iteration."""
        tr = self.cfg["train"]
        latest = os.path.join(self.ckpt_dir, "latest.pt")
        if crossed(prev, step, int(tr["eval_every"])):
            self._report(self.evaluator.run(self.agent, step))
            self.agent.save(latest, step)
        elif crossed(prev, step, int(tr.get("latest_every", 5000))):
            self.agent.save(latest, step)
        self._report(self.evaluator.poll())                 # finished background evaluations
        if crossed(prev, step, int(tr["save_every"])):
            self.agent.save(os.path.join(self.ckpt_dir, f"step_{step}.pt"), step)

    def _report(self, finished: list[FinishedEval]):
        """Log finished evaluations to eval.csv and keep best.pt (by train.best_metric)."""
        for ev in finished:
            res = ev.results
            mean_red = float(np.mean([v["effort_reduction_pct"] for v in res.values()]))
            fail = float(np.mean([v["fail_rate"] for v in res.values()]))
            for t, v in res.items():
                self.ev_log.write({"step": ev.step, "task": t, **{k: round(x, 4) for k, x in v.items()}})
            print(f"[eval ] step {ev.step}: mean effort reduction {mean_red:+.1f}% | fail rate {fail:.2f} | "
                  + " ".join(f"{t}:{v['effort_reduction_pct']:+.0f}%" for t, v in res.items()))
            score = eval_score(res, self.best_metric)
            if score > self.best:
                self.best = score
                ev.save_best(os.path.join(self.ckpt_dir, "best.pt"))
                print(f"[eval ] new best ({score:+.1f}, {self.best_metric})")
        self.evaluator.discard(finished)

    def _save_snapshot(self, step: int):
        os.makedirs(self.snap_dir, exist_ok=True)
        self.agent.save(os.path.join(self.snap_dir, "agent.pt"))
        self.buf.save(os.path.join(self.snap_dir, "buffer.npz"))
        baseline = self.evaluator.baseline
        state = {"step": step, "episode": self.episodes.episode, "best": float(self.best),
                 "init_from": self.init_from, "prefilled": self.prefilled,
                 "baseline": {k: float(v) for k, v in baseline.items()} if baseline else None}

        def write_state(tmp):
            with open(tmp, "w") as f:
                yaml.safe_dump(state, f)
        atomic_write(os.path.join(self.snap_dir, "state.yaml"), write_state)   # last: snapshot complete
        print(f"[train] snapshot saved at step {step}; continue with --name {self.name} --resume")

    def _load_snapshot(self, device) -> dict:
        self.agent = TD3Agent.load(os.path.join(self.snap_dir, "agent.pt"), device)
        self.buf.load(os.path.join(self.snap_dir, "buffer.npz"))
        with open(os.path.join(self.snap_dir, "state.yaml")) as f:
            return yaml.safe_load(f)
