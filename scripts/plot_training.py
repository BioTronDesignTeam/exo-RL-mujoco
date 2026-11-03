"""Plot training progress from a run's episodes.csv and eval.csv.

    python scripts/plot_training.py --run runs/demo                      # -> runs/demo/training.png
    python scripts/plot_training.py --run runs/demo --smooth 50          # heavier smoothing
    python scripts/plot_training.py --run runs/a runs/b --out cmp.png    # compare runs

One run: smoothed total reward per episode, smoothed human effort, and eval effort reduction per task,
with the best.pt step marked. Several runs: the same graphs with one color per run, and the eval graph
shows each run's checkpoint score (default output runs/compare_training.png).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.plotting import plot_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run folder(s), e.g. runs/demo; several = compare")
    ap.add_argument("--smooth", type=int, default=20, help="moving-average window in episodes (1 = off)")
    ap.add_argument("--out", default=None,
                    help="PNG path (default: <run>/training.png, or runs/compare_training.png for several runs)")
    args = ap.parse_args()

    for run in args.run:
        if not os.path.isdir(run):
            ap.error(f"no such run folder: {run}")
    print(plot_training(args.run, args.out, args.smooth))


if __name__ == "__main__":
    main()
