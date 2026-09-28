"""Shared helpers for scripts/view.py and scripts/watch_training.py: env setup for viewing,
camera, policy loading (incl. a checkpoint watcher), live viewer, and video recording."""
from __future__ import annotations

import io
import os
import platform
import sys
import time

import mujoco
import numpy as np

from .envs.exo_env import ExoAssistEnv
from .sim.reference import ReferenceTrajectory
from .utils import load_config


def macos_viewer_hint(script: str):
    if platform.system() == "Darwin" and "mjpython" not in os.path.basename(sys.executable):
        print(f"On macOS the live viewer must be started with mjpython:\n    mjpython {script} ...\n"
              "(or record a video instead)")


def view_config(run: str | None, config: str, overrides=(), no_exo: bool = False) -> dict:
    cfg = load_config(os.path.join(run, "config.yaml") if run else config, list(overrides))
    if no_exo:
        cfg.setdefault("exo_hardware", {})["enabled"] = False
    return cfg


def make_view_env(cfg: dict, task: str, mode: str, seed: int = 0, randomize: bool = False,
                  endless: bool = True) -> ExoAssistEnv:
    """Env with detailed meshes, reset to `task` ("circuit" = ACE course, repeated).
    endless: never truncate (view.py); otherwise episodes keep their configured length."""
    env = ExoAssistEnv(cfg, seed=seed)
    env.exo_quality = "hi"
    reset_view_env(env, task, mode, seed, randomize)
    if endless:
        env.max_steps = 10 ** 9
    return env


def reset_view_env(env: ExoAssistEnv, task: str, mode: str, seed: int = 0, randomize: bool = False):
    circuit = task == "circuit"
    obs, _ = env.reset(seed=seed, options={"task": None if circuit else task, "exo_mode": mode,
                                           "nominal": not randomize})
    if circuit:                  # long, repeating ACE course so it never ends mid-view
        tcfg = env.cfg["tasks"]
        names = [n for n in tcfg.get("ace_order", tcfg["names"]) if n in env.motions]
        env.ref = ReferenceTrajectory([(env.motions[n], 4.0) for n in names] * 20,
                                      blend=float(tcfg.get("blend_seconds", 0.5)))
        env.speeds = {m.name: env.hm.implied_speed(m) for m, _ in env.ref.segments}
        q, qd, _, tr, _ = env.ref.sample(0.0)
        env.hm.set_pose(q, tr, qd, root_vx=env._v_ref(0.0))
        env.human.reset()
    return obs


def setup_camera(cam, env: ExoAssistEnv, mode: str = "track"):
    if mode == "fixed":          # static side view: the pilot visibly travels across the frame
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        x0 = float(env.hm.data.qpos[env.hm.root_qpos[0]])
        cam.lookat[:] = [x0 + 2.0, 0.0, 0.8]
        cam.distance, cam.azimuth, cam.elevation = 6.0, 90.0, -8.0
        return
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.hm.model.body("pelvis").id
    cam.distance, cam.azimuth, cam.elevation = 2.8, 110.0, -12.0


def load_policy(path_or_file, device: str = "cpu"):
    """Deterministic policy from a TD3Agent checkpoint -> (policy, agent)."""
    from .agents.td3 import TD3Agent
    agent = TD3Agent.load(path_or_file, device)
    return (lambda o: agent.act(o, 0.0)), agent


class CheckpointWatcher:
    """Reloads a checkpoint when its file changes. A missing, half-written, or unreadable file
    never raises: the previous policy (or None) is kept and the next poll tries again."""

    def __init__(self, path: str):
        self.path = path
        self.key = None              # (mtime_ns, size) of the loaded file
        self.policy = None
        self.step = None             # env steps at save time (None for old checkpoints)

    def poll(self) -> bool:
        """True if a new policy was loaded."""
        try:
            st = os.stat(self.path)
            key = (st.st_mtime_ns, st.st_size)
            if key == self.key:
                return False
            with open(self.path, "rb") as f:       # read fast; the trainer replaces the file atomically
                blob = f.read()
            policy, agent = load_policy(io.BytesIO(blob))
        except Exception:                          # missing, being replaced, or corrupt
            return False
        self.key, self.policy, self.step = key, policy, agent.step
        return True

    def label(self) -> str:
        return "no checkpoint yet" if self.policy is None else \
            f"step {self.step}" if self.step is not None else "step ?"


class Stats:
    def __init__(self):
        self.effort, self.last_print = [], -1.0

    def update(self, env, info, force=False):
        self.effort.append(info["reward_terms"]["effort"])
        if force or env.t - self.last_print >= 1.0:
            self.last_print = env.t
            print(f"t={env.t:6.1f}s  {info['task']:14s} phase {info['phase']:.2f}  effort {self.effort[-1]:.3f} "
                  f"(mean {np.mean(self.effort):.3f})  exo tau [hipL kneeL hipR kneeR] = {np.round(info['tau_exo_cmd'], 1)}")


def exit_after_viewer(code: int = 0):
    """End the process after a live viewer closed. MuJoCo's passive viewer sometimes segfaults
    during normal interpreter teardown (seen ~1 in 10 exits on Windows, MuJoCo 3.4); skipping
    the teardown avoids it. Everything worth keeping must be written before calling this."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


class LiveView:
    """MuJoCo passive viewer, paced to `speed` x real time. It displays a copy of the env's
    first model and mirrors the state into it, so the window survives env.reset() rebuilding
    the model (the body must stay the same: nominal subjects, or a single episode)."""

    def __init__(self, env: ExoAssistEnv, camera: str = "track", speed: float = 1.0):
        import mujoco.viewer
        self.model = env.hm.model
        self.data = mujoco.MjData(self.model)
        self.camera, self.dt = camera, env.control_dt / speed
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.has_text = hasattr(self.handle, "set_texts")
        self.text = None
        self.t_next = time.perf_counter()
        self.show(env)
        self.reset_camera(env)

    def reset_camera(self, env):
        setup_camera(self.handle.cam, env, self.camera)      # same body ids in the display model

    def running(self) -> bool:
        return self.handle.is_running()

    def show(self, env, text: tuple[str, str] | None = None):
        """Mirror env state, draw, and sleep to keep real-time pace. text = (labels, values)."""
        src = env.hm.data
        self.data.qpos[:], self.data.qvel[:], self.data.time = src.qpos, src.qvel, src.time
        mujoco.mj_forward(self.model, self.data)
        if text is not None and text != self.text and self.has_text:   # set_texts is slow: only on change
            self.text = text
            self.handle.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, text[0], text[1]))
        self.handle.sync()
        self.t_next += self.dt
        time.sleep(max(0.0, self.t_next - time.perf_counter()))
        if self.t_next < time.perf_counter() - 1.0:          # fell behind (e.g. after a reload)
            self.t_next = time.perf_counter()

    def close(self):
        self.handle.close()


def record(env: ExoAssistEnv, policy, path: str, seconds: float = 8.0, speed: float = 1.0,
           width: int = 720, height: int = 528, camera: str = "track", on_step=None, fps: int = 30) -> int:
    """Run the env from its current state and save .mp4/.gif (or the last frame as .png)."""
    import imageio
    vis = env.hm.model.vis.global_          # offscreen buffer must fit the requested size
    vis.offwidth, vis.offheight = max(vis.offwidth, width), max(vis.offheight, height)
    renderer = mujoco.Renderer(env.hm.model, height, width)
    cam = mujoco.MjvCamera()
    setup_camera(cam, env, camera)
    obs = np.concatenate(env.obs_builder.frames)
    frames, next_frame_t, t_end = [], env.t, env.t + seconds
    while env.t < t_end:
        a = policy(obs) if policy else np.zeros(4)
        obs, _, term, trunc, info = env.step(a)
        if on_step:
            on_step(env, info)
        if env.t >= next_frame_t:                 # one frame per 1/fps of video time
            renderer.update_scene(env.hm.data, cam)
            frames.append(renderer.render())
            next_frame_t += speed / fps
        if term or trunc:
            if term:
                print(f"episode ended: {info['terminated_reason']}")
            break
    renderer.close()
    if path.endswith(".png"):
        imageio.imwrite(path, frames[-1])
    else:
        imageio.mimsave(path, frames, fps=fps)
    return len(frames)
