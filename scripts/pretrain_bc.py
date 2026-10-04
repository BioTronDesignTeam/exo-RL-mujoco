"""Warm start: behavior-clone a simple privileged teacher into the TD3 actor.

    python scripts/pretrain_bc.py --name bc                      # ~200k samples, parallel envs
    python scripts/train.py --name ws --init-from runs/bc/checkpoints/bc.pt   # RL fine-tuning

The teacher and the data collector live in exo_rl/warmstart.py; this file is the command line.
Data is collected in rounds with parallel envs (randomization on): the exo executes the
teacher's action plus small noise, mixed with the current BC policy in later rounds
(DAgger-style), so the dataset covers states the learned policy visits. After each round the
actor (same architecture as TD3) is trained with MSE on (observation -> teacher action).

Writes runs/<name>/:
    config.yaml          config used (the fine-tuning run should use the same observation settings)
    checkpoints/bc.pt    TD3Agent checkpoint: BC actor (+ target), freshly initialized critic
    checkpoints/bc_buffer.npz   the collected transitions (warmstart.save_buffer), to prefill
                         the replay buffer: train.py --init-buffer (the critic can learn at once)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.agents.td3 import ReplayBuffer, TD3Agent  # noqa: E402
from exo_rl.envs.exo_env import ExoAssistEnv  # noqa: E402
from exo_rl.envs.vector import make_runner  # noqa: E402
from exo_rl.evaluation import evaluate, no_exo_effort  # noqa: E402
from exo_rl.utils import configure_torch_threads, load_config, pick_device  # noqa: E402
from exo_rl.warmstart import Collector, causal_teacher, mse, train_bc  # noqa: E402


def report(cfg, agent, fraction, episodes):
    """Teacher and BC policy effort per task: vs exo off, and net vs not wearing the exo."""
    tasks = cfg["tasks"]["names"]
    env = ExoAssistEnv(cfg, seed=0)
    no_exo = no_exo_effort(cfg, tasks, episodes, seed0=2000)
    res_t, base = evaluate(env, causal_teacher(env, fraction), tasks, episodes, seed0=2000)
    res_b, _ = evaluate(env, lambda o: agent.act(o, 0.0), tasks, episodes, seed0=2000, baseline=base)
    net = lambda r, t: f"{100 * (1 - r[t]['effort'] / no_exo[t]):+13.1f}%" if no_exo else ""  # noqa: E731
    hdr = f"{'net vs no exo':>14s}" if no_exo else ""
    print(f"\n{'task':15s}{'policy':9s}{'effort':>8s}{'vs exo off':>12s}{hdr}{'fail':>6s}")
    for t in tasks:
        for label, r in (("teacher", res_t), ("BC", res_b)):
            print(f"{t:15s}{label:9s}{r[t]['effort']:8.3f}{r[t]['effort_reduction_pct']:+11.1f}%{net(r, t)}"
                  f"{r[t]['fail_rate']:6.2f}")
    print("(teacher acts on last step's tau_needed; BC on sensor observations only)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. warmstart.rounds=2")
    ap.add_argument("--steps", type=int, default=200000, help="dataset size (transitions)")
    ap.add_argument("--epochs", type=int, default=None, help="BC epochs per round (warmstart.epochs)")
    ap.add_argument("--teacher-fraction", type=float, default=None, help="warmstart.teacher_fraction")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ws = cfg["warmstart"]
    if args.epochs is not None:
        ws["epochs"] = args.epochs
    if args.teacher_fraction is not None:
        ws["teacher_fraction"] = args.teacher_fraction
    name = args.name or time.strftime("bc_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(cfg["train"].get("device", "auto"))
    num_envs = int(ws.get("num_envs", 0)) or min(8, os.cpu_count() or 1)
    configure_torch_threads(int(cfg["train"].get("torch_threads", 0)), num_envs)
    probe = ExoAssistEnv(cfg, seed=seed)
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.shape[0]
    runner = make_runner(cfg, num_envs, seed)
    agent = TD3Agent(obs_dim, act_dim, cfg["agent"], device)
    rounds = int(ws["rounds"])
    print(f"[bc] teacher fraction {ws['teacher_fraction']}, label offset {ws['label_offset_steps']} step(s) | "
          f"{args.steps} samples in {rounds} rounds on {num_envs} envs | {ws['epochs']} epochs/round")

    t0 = time.time()
    xs, ys = [], []
    try:
        buf = ReplayBuffer(obs_dim, act_dim, args.steps + rounds * num_envs) if ws.get("save_buffer", True) else None
        col = Collector(runner, ws, cfg["exo"]["action_scale"], buf)
        for r in range(rounds):
            mix = r / rounds
            tc = time.time()
            x, y = col.collect(args.steps // rounds, agent.actor if r > 0 else None, mix, device)
            xs.append(x)
            ys.append(y)
            X, Y = np.concatenate(xs), np.concatenate(ys)
            before = mse(agent.actor, x, y, device)       # new data under the current policy
            tt = time.time()
            losses = train_bc(agent.actor, X, Y, int(ws["epochs"]), int(ws["batch_size"]), float(ws["lr"]), device)
            print(f"[bc] round {r + 1}/{rounds}: executed {mix:.0%} BC | +{len(x)} samples ({tt - tc:.0f} s) | "
                  f"new-data MSE before {before:.4f} | train MSE {losses[0]:.4f} -> {losses[-1]:.4f} "
                  f"({time.time() - tt:.0f} s)")
    finally:
        runner.close()

    agent.actor_t.load_state_dict(agent.actor.state_dict())
    path = os.path.join(run_dir, "checkpoints", "bc.pt")
    agent.save(path, step=0)
    buf_path = os.path.join(run_dir, "checkpoints", "bc_buffer.npz")
    if buf is not None:
        buf.save(buf_path)
        print(f"[bc] saved {len(buf)} transitions to {buf_path}")
    print(f"[bc] saved {path} after {(time.time() - t0) / 60:.1f} min")
    report(cfg, agent, float(ws["teacher_fraction"]), int(ws.get("eval_episodes", 2)))
    print(f"\nFine-tune with: python scripts/train.py --name {name}_rl --config {run_dir}/config.yaml "
          f"--init-from {path}" + (f" --init-buffer {buf_path}" if buf is not None else ""))


if __name__ == "__main__":
    main()
