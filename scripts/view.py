"""Watch the simulated human wearing the exo, live in the MuJoCo viewer or as a video.

    python scripts/view.py --task squat                          # exo worn, unpowered
    python scripts/view.py --run runs/demo --task walk           # trained policy drives the exo
    python scripts/view.py --task circuit --speed 0.5
    python scripts/view.py --run runs/demo --task squat --record squat.mp4   # no window needed
    python scripts/view.py --save-xml model.xml                  # dump the generated MJCF

macOS: the live viewer needs `mjpython scripts/view.py ...` (installed with mujoco).
Headless recording on Linux: MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.viewing import (LiveView, Stats, exit_after_viewer, load_policy, macos_viewer_hint,  # noqa: E402
                            make_view_env, record, view_config)


def run_live(env, policy, args):
    macos_viewer_hint("scripts/view.py")
    stats, obs = Stats(), np.concatenate(env.obs_builder.frames)
    view = LiveView(env, args.camera, args.speed)
    finished = False
    status, last_info = "Running", None
    hud, next_hud, previous_status = None, 0., None
    try:
        while view.running():
            if not finished:
                if args.seconds is not None and env.t >= args.seconds:
                    print("Simulation duration reached. Close the viewer window to exit.", flush=True)
                    finished = True
                    status = "Finished (close window to exit)"
                else:
                    a = policy(obs) if policy else np.zeros(4)
                    obs, _, term, _, info = env.step(a)
                    last_info = info
                    stats.update(env, info)
                    if term:
                        print(f"episode ended: {info['terminated_reason']}. Close the viewer window to exit.", flush=True)
                        finished = True
                        status = f"Stopped: {info['terminated_reason']}"
            if env.t >= next_hud or status != previous_status:
                labels, values = "Status\nSimulation time", f"{status}\n{env.t:.1f} s"
                if last_info is not None and "posture" in last_info:
                    angles = last_info["posture"]
                    labels += "\nPelvis\nSpine flexion\nTorso\nWorld X"
                    values += (f"\n{np.degrees(angles['pelvis']):.1f} deg"
                               f"\n{np.degrees(angles['lumbar']):.1f} deg"
                               f"\n{np.degrees(angles['torso']):.1f} deg"
                               f"\n{env.hm.data.qpos[env.hm.root_qpos[0]]:.2f} m")
                hud = labels, values
                next_hud, previous_status = env.t + .25, status
            view.show(env, hud)
    finally:
        view.close()
    return stats


def run_record(env, policy, args):
    stats = Stats()
    n = record(env, policy, args.record, args.seconds or 8.0, args.speed, args.width, args.height,
               args.camera, on_step=stats.update)
    print(f"saved {n} frames to {args.record}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="training run folder; drives the exo with its policy")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. exo.alpha=0.5")
    ap.add_argument("--mode", choices=["policy", "off", "feedforward", "teacher"],
                    help="teacher is a privileged simulation-only assistance baseline")
    ap.add_argument("--task", default="squat",
                    help="squat | high_knees | quick_steps | backward_walk | walk | jog | high_knees_travel | circuit")
    ap.add_argument("--camera", choices=["track", "fixed"], default="track",
                    help="track follows the pelvis; fixed is a static side view showing forward travel")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (0.25 = slow motion)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="simulation/video duration; live view keeps the final frame until you close the window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="random subject instead of nominal")
    ap.add_argument("--no-exo", action="store_true", help="hide/remove the CAD exo (human + payload only)")
    ap.add_argument("--record", help="write a video (.mp4/.gif) or last frame (.png) instead of opening a window")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=528)
    ap.add_argument("--save-xml", help="write the generated MJCF and exit")
    args = ap.parse_args()

    cfg = view_config(args.run, args.config, args.set, args.no_exo)
    mode = args.mode or ("policy" if args.run else "off")
    print(f"Preparing {args.task} ({mode}); Myo configs may include muscle settling...", flush=True)
    env = make_view_env(cfg, args.task, mode, args.seed, args.randomize)
    if args.save_xml:
        with open(args.save_xml, "w") as f:
            f.write(env.hm.xml)
        if env.exo_hw is not None:
            mesh_dir = os.path.join(os.path.dirname(os.path.abspath(args.save_xml)))
            for name, data in env.exo_hw.meshes("hi").items():
                with open(os.path.join(mesh_dir, name), "wb") as f:
                    f.write(data)
        print(f"wrote {args.save_xml} (+ exo meshes next to it); view with: python -m mujoco.viewer --mjcf={args.save_xml}")
        return
    if env.exo_hw is not None:
        print(env.exo_hw.report(env.hm.segments.thigh_len))
    print(f"task={args.task} mode={mode} policy={'yes' if args.run and mode == 'policy' else 'no'}")
    policy = load_policy(os.path.join(args.run, "checkpoints", args.ckpt))[0] if args.run and mode == "policy" else None
    if not args.record:
        print("Opening MuJoCo viewer; close its window to exit.", flush=True)
    stats = run_record(env, policy, args) if args.record else run_live(env, policy, args)
    if stats.effort:
        print(f"session mean human effort: {np.mean(stats.effort):.3f}")
    if not args.record:
        exit_after_viewer()


if __name__ == "__main__":
    main()
