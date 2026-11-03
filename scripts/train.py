"""Train a TD3/DDPG exo assistance policy.

    python scripts/train.py --config configs/default.yaml --name baseline
    python scripts/train.py --set agent.algo=ddpg --set train.total_steps=200000
    python scripts/train.py --name baseline --max-minutes 60      # stop cleanly, save snapshot
    python scripts/train.py --name baseline --resume               # continue from snapshot
    python scripts/train.py --set train.num_envs=8 --set train.updates_per_step=0.25  # parallel
    python scripts/train.py --name ft --init-from runs/baseline/checkpoints/best.pt   # fine-tune
    python scripts/train.py --name ws --init-from runs/bc/checkpoints/bc.pt   # after pretrain_bc.py

With --init-from, the start_steps warm-up uses the initialized policy (+ exploration noise)
instead of random actions, and agent.actor_freeze_steps / bc_weight / bc_decay_steps protect
the initial policy from the still-untrained critic (see exo_rl/agents/td3.py).

The loop itself lives in exo_rl/training.py (Trainer); this file is only the command line.

Outputs in runs/<name>/:
    config.yaml        exact config used (needed for export/deploy)
    episodes.csv       per-episode return, length, task, env_id, mean of every reward term
    eval.csv           per-task effort reduction vs exo-off, tracking, torque stats
    checkpoints/       latest.pt (every train.latest_every steps), best.pt (by train.best_metric),
                       step_N.pt; all written atomically (temp file + rename)
    resume/            snapshot (agent, replay buffer, counters) for --resume
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.utils import load_config  # noqa: E402


def main():
    # Windows workers re-import this script; only the parent needs PyTorch/CUDA.
    from exo_rl.training import Trainer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default=None, help="run folder runs/NAME (default: timestamp)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. agent.algo=ddpg")
    ap.add_argument("--sim-backend", choices=["cpu", "warp"], help="physics backend (overrides config)")
    ap.add_argument("--num-envs", type=int, help="number of parallel worlds (overrides config)")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its snapshot")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop and snapshot after this long")
    ap.add_argument("--init-from", default=None,
                    help="start actor/critic (+targets) from this checkpoint; fresh optimizers, empty buffer")
    ap.add_argument("--init-buffer", default=None,
                    help="prefill the replay buffer (e.g. bc_buffer.npz from pretrain_bc.py); learning starts at once")
    args = ap.parse_args()
    if hasattr(signal, "SIGBREAK"):          # Windows: Ctrl+Break stops cleanly, like Ctrl+C
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    if args.sim_backend is not None:
        args.set.append(f"train.sim_backend={args.sim_backend}")
    if args.num_envs is not None:
        args.set.append(f"train.num_envs={args.num_envs}")

    name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", name)
    if args.resume:
        cfg = load_config(os.path.join(run_dir, "config.yaml"), args.set)
    else:
        cfg = load_config(args.config, args.set)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:     # persist overrides (also on resume)
        yaml.safe_dump(cfg, f, sort_keys=False)

    try:
        trainer = Trainer(cfg, run_dir, name, resume=args.resume, init_from=args.init_from,
                          init_buffer=args.init_buffer)
    except ValueError as e:                  # --init-from / --init-buffer don't fit this config
        sys.exit(f"[train] error: {e}")
    trainer.run(args.max_minutes)


if __name__ == "__main__":
    main()
