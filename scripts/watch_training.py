"""Watch a policy while it trains: reloads checkpoints/latest.pt whenever training rewrites it.

    python scripts/watch_training.py --run runs/mk1                      # live viewer
    python scripts/watch_training.py --run runs/mk1 --tasks squat walk --speed 0.5
    python scripts/watch_training.py --run runs/mk1 --record-dir runs/mk1/videos   # headless

Run it in a second terminal next to train.py (training itself stays headless). The live
viewer cycles through --tasks, one episode each (nominal pilot), and switches to a new policy
as soon as one appears. --record-dir instead saves a --seconds video per task for every new
checkpoint: DIR/step_000120000_squat.mp4.

macOS: the live viewer needs `mjpython scripts/watch_training.py ...`.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.viewing import (CheckpointWatcher, LiveView, exit_after_viewer, macos_viewer_hint,  # noqa: E402
                            make_view_env, record, reset_view_env, view_config)


def poll(watcher) -> bool:
    if watcher.poll():
        print(f"[watch] loaded {watcher.path} ({watcher.label()})")
        return True
    return False


def run_live(env, watcher, args):
    macos_viewer_hint("scripts/watch_training.py")
    view = LiveView(env, args.camera, args.speed)
    if not view.has_text:
        print("[watch] this MuJoCo viewer has no text overlay; step and task are printed here instead")
    last_poll = -np.inf
    try:
        for task in itertools.cycle(args.tasks):
            obs = reset_view_env(env, task, "policy", args.seed)
            view.reset_camera(env)
            effort = []
            while view.running():
                if time.monotonic() - last_poll >= args.poll:
                    last_poll = time.monotonic()
                    poll(watcher)
                a = watcher.policy(obs) if watcher.policy else np.zeros(4)
                obs, _, term, trunc, info = env.step(a)
                effort.append(info["reward_terms"]["effort"])
                view.show(env, ("checkpoint\ntask", f"{watcher.label()}\n{task}"))
                if term or trunc:
                    end = f"ended: {info['terminated_reason']}" if term else "done"
                    print(f"[watch] {watcher.label():>12s}  {task:14s} mean effort {np.mean(effort):.3f} ({end})")
                    break
            if not view.running():
                break
    finally:
        view.close()


def run_record(env, watcher, args):
    if importlib.util.find_spec("imageio") is None:
        sys.exit('[watch] --record-dir needs imageio: pip install "imageio[ffmpeg]"')
    os.makedirs(args.record_dir, exist_ok=True)
    env.max_steps = 10 ** 9                      # videos run for --seconds, not the episode length
    print(f"[watch] recording {args.seconds:g} s per task into {args.record_dir} for each new checkpoint "
          f"(Ctrl+C to stop)")
    while True:
        if poll(watcher):
            tag = f"step_{watcher.step:09d}" if watcher.step is not None else time.strftime("time_%H%M%S")
            for task in args.tasks:
                reset_view_env(env, task, "policy", args.seed)
                path = os.path.join(args.record_dir, f"{tag}_{task}.mp4")
                n = record(env, watcher.policy, path, args.seconds, args.speed, args.width, args.height, args.camera)
                print(f"[watch] saved {path} ({n} frames)")
        time.sleep(args.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run folder (runs/NAME)")
    ap.add_argument("--poll", type=float, default=10.0, help="seconds between checks of latest.pt")
    ap.add_argument("--tasks", nargs="+", default=None, help="tasks to cycle (default: the run's training tasks)")
    ap.add_argument("--camera", choices=["track", "fixed"], default="track")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (0.25 = slow motion)")
    ap.add_argument("--record-dir", default=None, help="headless: save videos here instead of opening a window")
    ap.add_argument("--seconds", type=float, default=8.0, help="video length per task (--record-dir)")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=528)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)     # progress shows up promptly when piped

    cfg_path = os.path.join(args.run, "config.yaml")
    if not os.path.exists(cfg_path):
        print(f"[watch] waiting for {cfg_path} (training not started yet?)")
        while not os.path.exists(cfg_path):
            time.sleep(args.poll)
        time.sleep(0.5)                          # let train.py finish writing it
    cfg = view_config(args.run, "", ())
    args.tasks = args.tasks or list(cfg["tasks"]["names"])
    env = make_view_env(cfg, args.tasks[0], "policy", args.seed, endless=False)
    watcher = CheckpointWatcher(os.path.join(args.run, "checkpoints", "latest.pt"))
    if not os.path.exists(watcher.path):
        print(f"[watch] waiting for {watcher.path} (zero exo action until it appears)")
    try:
        run_record(env, watcher, args) if args.record_dir else run_live(env, watcher, args)
    except KeyboardInterrupt:
        pass
    if not args.record_dir:
        exit_after_viewer()


if __name__ == "__main__":
    main()
