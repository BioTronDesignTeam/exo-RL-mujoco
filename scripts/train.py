"""Train a TD3/DDPG exo assistance policy.

    python scripts/train.py --config configs/default.yaml --name baseline
    python scripts/train.py --set agent.algo=ddpg --set train.total_steps=200000
    python scripts/train.py --name baseline --max-minutes 60      # stop cleanly, save snapshot
    python scripts/train.py --name baseline --resume               # continue from snapshot

Outputs in runs/<name>/:
    config.yaml        exact config used (needed for export/deploy)
    episodes.csv       per-episode return, length, task, mean of every reward term
    eval.csv           per-task effort reduction vs exo-off, tracking, torque stats
    checkpoints/       latest.pt, best.pt (by mean eval effort reduction), step_N.pt
    resume/            snapshot (agent, replay buffer, counters) for --resume
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.agents.td3 import ReplayBuffer, TD3Agent  # noqa: E402
from exo_rl.envs.exo_env import ExoAssistEnv  # noqa: E402
from exo_rl.evaluation import evaluate  # noqa: E402
from exo_rl.utils import load_config, pick_device  # noqa: E402


class CsvLog:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. agent.algo=ddpg")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its snapshot")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop and snapshot after this long")
    args = ap.parse_args()

    name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", name)
    if args.resume:
        cfg = load_config(os.path.join(run_dir, "config.yaml"), args.set)
    else:
        cfg = load_config(args.config, args.set)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:     # persist overrides (also on resume)
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr = cfg["train"]
    device = pick_device(tr.get("device", "auto"))

    env = ExoAssistEnv(cfg, seed=seed)
    eval_env = ExoAssistEnv(cfg, seed=seed + 1)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = TD3Agent(obs_dim, act_dim, cfg["agent"], device)
    buf = ReplayBuffer(obs_dim, act_dim, int(cfg["agent"]["buffer_size"]))
    ep_log = CsvLog(os.path.join(run_dir, "episodes.csv"), append=args.resume)
    ev_log = CsvLog(os.path.join(run_dir, "eval.csv"), append=args.resume)
    tasks = cfg["tasks"]["names"]
    print(f"[train] {agent.algo.upper()} on {device} | obs {obs_dim} act {act_dim} | run dir {run_dir}")

    ep_i, best, baseline, start = 0, -np.inf, None, 1
    snap_dir = os.path.join(run_dir, "resume")
    if args.resume:
        agent = TD3Agent.load(os.path.join(snap_dir, "agent.pt"), device)
        data = np.load(os.path.join(snap_dir, "buffer.npz"))
        n = int(data["n"])
        for k in ("obs", "act", "rew", "next_obs", "not_done"):
            getattr(buf, k)[:n] = data[k]
        buf.ptr, buf.full = int(data["ptr"]), bool(data["full"])
        with open(os.path.join(snap_dir, "state.yaml")) as f:
            st = yaml.safe_load(f)
        start, ep_i, best = st["step"] + 1, st["episode"], st["best"]
        baseline = st.get("baseline")
        np.random.seed(seed + start)
        env = ExoAssistEnv(cfg, seed=seed + start)
        print(f"[train] resumed at step {start} (buffer {len(buf)})")

    def snapshot(step):
        os.makedirs(snap_dir, exist_ok=True)
        agent.save(os.path.join(snap_dir, "agent.pt"))
        n = len(buf)
        np.savez(os.path.join(snap_dir, "buffer.npz"), n=n, ptr=buf.ptr, full=buf.full,
                 **{k: getattr(buf, k)[:n] for k in ("obs", "act", "rew", "next_obs", "not_done")})
        with open(os.path.join(snap_dir, "state.yaml"), "w") as f:
            yaml.safe_dump({"step": step, "episode": ep_i, "best": float(best),
                            "baseline": {k: float(v) for k, v in baseline.items()} if baseline else None}, f)
        print(f"[train] snapshot saved at step {step}; continue with --name {name} --resume")

    obs, info = env.reset()
    ep_ret, ep_len, ep_terms = 0.0, 0, defaultdict(float)
    t0 = time.time()
    upd_info = {"critic_loss": float("nan"), "q_mean": float("nan"), "actor_loss": float("nan")}
    expl = float(cfg["agent"]["expl_noise"])

    for step in range(start, int(tr["total_steps"]) + 1):
        if step <= int(tr["start_steps"]):
            action = env.action_space.sample()
        else:
            action = agent.act(obs, noise=expl)
        obs2, r, term, trunc, inf = env.step(action)
        buf.add(obs, action, r, obs2, term)
        obs = obs2
        ep_ret += r
        ep_len += 1
        for k, v in inf["reward_terms"].items():
            ep_terms[k] += v

        if step > int(tr["start_steps"]):
            upd_info.update(agent.update(buf, int(cfg["agent"]["batch_size"])))

        if term or trunc:
            row = {"step": step, "episode": ep_i, "task": info["task"], "return": round(ep_ret, 3),
                   "length": ep_len, "terminated": int(term), "reason": inf["terminated_reason"]}
            row.update({f"mean_{k}": round(v / ep_len, 5) for k, v in ep_terms.items()})
            row.update({k: round(v, 4) for k, v in upd_info.items()})
            ep_log.write(row)
            if ep_i % 10 == 0:
                sps = (step - start + 1) / (time.time() - t0)
                print(f"[train] step {step:>8d} ep {ep_i:>5d} ret {ep_ret:7.1f} len {ep_len:4d} "
                      f"effort {row['mean_effort']:.3f} ({info['task']}) {sps:.0f} steps/s")
            obs, info = env.reset()
            ep_ret, ep_len, ep_terms = 0.0, 0, defaultdict(float)
            ep_i += 1

        if step % int(tr["eval_every"]) == 0:
            policy = lambda o: agent.act(o, noise=0.0)  # noqa: E731
            res, baseline = evaluate(eval_env, policy, tasks, int(tr["eval_episodes"]), baseline=baseline)
            mean_red = float(np.mean([v["effort_reduction_pct"] for v in res.values()]))
            fail = float(np.mean([v["fail_rate"] for v in res.values()]))
            for t, v in res.items():
                ev_log.write({"step": step, "task": t, **{k: round(x, 4) for k, x in v.items()}})
            print(f"[eval ] step {step}: mean effort reduction {mean_red:+.1f}% | fail rate {fail:.2f} | "
                  + " ".join(f"{t}:{v['effort_reduction_pct']:+.0f}%" for t, v in res.items()))
            agent.save(os.path.join(run_dir, "checkpoints", "latest.pt"))
            score = mean_red - 100.0 * fail
            if score > best:
                best = score
                agent.save(os.path.join(run_dir, "checkpoints", "best.pt"))
                print(f"[eval ] new best ({score:+.1f})")

        if step % int(tr["save_every"]) == 0:
            agent.save(os.path.join(run_dir, "checkpoints", f"step_{step}.pt"))

        if args.max_minutes and (time.time() - t0) > 60 * args.max_minutes and (term or trunc):
            snapshot(step)
            return

    agent.save(os.path.join(run_dir, "checkpoints", "latest.pt"))
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
