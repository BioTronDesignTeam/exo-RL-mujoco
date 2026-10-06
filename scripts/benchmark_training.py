"""Measure end-to-end training throughput; run on the target NVIDIA host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gpu_4060ti.yaml")
    ap.add_argument("--sim-backend", choices=["cpu", "warp"], default="warp")
    ap.add_argument("--envs", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--output", default=None, help="new output directory; existing directories are rejected")
    args = ap.parse_args()
    if args.minutes <= 0 or any(n <= 0 for n in args.envs):
        ap.error("minutes and environment counts must be positive")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output or f"runs/benchmark_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for n in args.envs:
        run = out / f"envs_{n}"
        # Hold gradient work at two updates per collection round across counts.
        settings = ["train.total_steps=1000000000", "train.eval_every=0",
                    "train.save_every=0", "train.latest_every=0",
                    f"train.updates_per_step={2 / n}", *args.set]
        cmd = [sys.executable, str(root / "scripts/train.py"), "--config", args.config,
               "--name", str(run), "--sim-backend", args.sim_backend,
               "--num-envs", str(n), "--max-minutes", str(args.minutes)]
        for setting in settings:
            cmd += ["--set", setting]
        started = time.perf_counter()
        subprocess.run(cmd, cwd=root, check=True)
        elapsed = time.perf_counter() - started
        state = yaml.safe_load((run / "resume/state.yaml").read_text())
        steps = int(state["step"])
        rate = steps / elapsed
        row = {"backend": args.sim_backend, "num_envs": n, "transitions": steps,
               "wall_seconds": elapsed, "transitions_per_second": rate,
               "projected_1m_minutes": 1000000 / rate / 60,
               "evaluation_enabled": False}
        rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)
    print(f"Saved {out / 'results.json'}; projections exclude evaluation and do not measure policy quality.")


if __name__ == "__main__":
    main()
