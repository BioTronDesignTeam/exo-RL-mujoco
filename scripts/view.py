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
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402

from exo_rl.envs.exo_env import ExoAssistEnv  # noqa: E402
from exo_rl.sim.reference import ReferenceTrajectory  # noqa: E402
from exo_rl.utils import load_config  # noqa: E402


def make_env(args):
    cfg_path = os.path.join(args.run, "config.yaml") if args.run else args.config
    cfg = load_config(cfg_path, args.set)
    if args.no_exo:
        cfg.setdefault("exo_hardware", {})["enabled"] = False
    env = ExoAssistEnv(cfg, seed=args.seed)
    env.exo_quality = "hi"                     # detailed meshes for viewing
    mode = args.mode or ("policy" if args.run else "off")
    task = None if args.task == "circuit" else args.task
    env.reset(seed=args.seed, options={"task": task, "exo_mode": mode, "nominal": not args.randomize})
    if args.task == "circuit":                 # long, repeating circuit so it never ends mid-view
        names = [n for n in cfg["tasks"].get("ace_order", cfg["tasks"]["names"]) if n in env.motions]
        segs = [(env.motions[n], 4.0) for n in names] * 20      # ACE course order, repeated
        env.ref = ReferenceTrajectory(segs, blend=float(cfg["tasks"].get("blend_seconds", 0.5)))
        env.speeds = {m.name: env.hm.implied_speed(m) for m, _ in env.ref.segments}
        q, qd, _, tr, _ = env.ref.sample(0.0)
        env.hm.set_pose(q, tr, qd, root_vx=env._v_ref(0.0))
        env.human.reset()
    env.max_steps = 10 ** 9
    return env, cfg, mode


def make_policy(args):
    if not args.run:
        return None
    from exo_rl.agents.td3 import TD3Agent
    agent = TD3Agent.load(os.path.join(args.run, "checkpoints", args.ckpt))
    return lambda o: agent.act(o, 0.0)


CAMERA = "track"


def setup_camera(cam, env):
    if CAMERA == "fixed":        # static side view: the pilot visibly travels across the frame
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        x0 = float(env.hm.data.qpos[env.hm.root_qpos[0]])
        cam.lookat[:] = [x0 + 2.0, 0.0, 0.8]
        cam.distance, cam.azimuth, cam.elevation = 6.0, 90.0, -8.0
        return
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.hm.model.body("pelvis").id
    cam.distance, cam.azimuth, cam.elevation = 2.8, 110.0, -12.0


class Stats:
    def __init__(self):
        self.effort, self.last_print = [], -1.0

    def update(self, env, info, force=False):
        self.effort.append(info["reward_terms"]["effort"])
        if force or env.t - self.last_print >= 1.0:
            self.last_print = env.t
            print(f"t={env.t:6.1f}s  {info['task']:14s} phase {info['phase']:.2f}  effort {self.effort[-1]:.3f} "
                  f"(mean {np.mean(self.effort):.3f})  exo tau [hipL kneeL hipR kneeR] = {np.round(info['tau_exo_cmd'], 1)}")


def run_live(env, policy, args):
    import mujoco.viewer
    if platform.system() == "Darwin" and "mjpython" not in os.path.basename(sys.executable):
        print("On macOS the live viewer must be started with mjpython:\n"
              "    mjpython scripts/view.py ...\n(or use --record to make a video instead)")
    stats, obs = Stats(), np.concatenate(env.obs_builder.frames)
    dt = env.control_dt / args.speed
    with mujoco.viewer.launch_passive(env.hm.model, env.hm.data) as viewer:
        setup_camera(viewer.cam, env)
        t_next = time.perf_counter()
        while viewer.is_running() and (args.seconds is None or env.t < args.seconds):
            a = policy(obs) if policy else np.zeros(4)
            obs, _, term, _, info = env.step(a)
            stats.update(env, info)
            if term:
                print(f"episode ended: {info['terminated_reason']}")
                break
            viewer.sync()
            t_next += dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
    return stats


def run_record(env, policy, args):
    import imageio
    fps = 30
    seconds = args.seconds or 8.0
    vis = env.hm.model.vis.global_          # offscreen buffer must fit the requested size
    vis.offwidth, vis.offheight = max(vis.offwidth, args.width), max(vis.offheight, args.height)
    renderer = mujoco.Renderer(env.hm.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    setup_camera(cam, env)
    stats, obs = Stats(), np.concatenate(env.obs_builder.frames)
    frames, next_frame_t = [], 0.0
    while env.t < seconds:
        a = policy(obs) if policy else np.zeros(4)
        obs, _, term, _, info = env.step(a)
        stats.update(env, info)
        if env.t >= next_frame_t:                 # one frame per 1/fps of video time
            renderer.update_scene(env.hm.data, cam)
            frames.append(renderer.render())
            next_frame_t += args.speed / fps
        if term:
            print(f"episode ended: {info['terminated_reason']}")
            break
    imageio.mimsave(args.record, frames, fps=fps) if not args.record.endswith(".png") else \
        imageio.imwrite(args.record, frames[-1])
    print(f"saved {len(frames)} frames to {args.record}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="training run folder; drives the exo with its policy")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. exo.alpha=0.5")
    ap.add_argument("--mode", choices=["policy", "off", "feedforward"])
    ap.add_argument("--task", default="squat",
                    help="squat | high_knees | quick_steps | backward_walk | walk | jog | high_knees_travel | circuit")
    ap.add_argument("--camera", choices=["track", "fixed"], default="track",
                    help="track follows the pelvis; fixed is a static side view showing forward travel")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (0.25 = slow motion)")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="random subject instead of nominal")
    ap.add_argument("--no-exo", action="store_true", help="hide/remove the CAD exo (human + payload only)")
    ap.add_argument("--record", help="write a video (.mp4/.gif) or last frame (.png) instead of opening a window")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=528)
    ap.add_argument("--save-xml", help="write the generated MJCF and exit")
    args = ap.parse_args()

    global CAMERA
    CAMERA = args.camera
    env, cfg, mode = make_env(args)
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
    print(f"task={args.task} mode={mode} policy={'yes' if args.run else 'no'}")
    policy = make_policy(args) if mode == "policy" else None
    stats = run_record(env, policy, args) if args.record else run_live(env, policy, args)
    if stats.effort:
        print(f"session mean human effort: {np.mean(stats.effort):.3f}")


if __name__ == "__main__":
    main()
