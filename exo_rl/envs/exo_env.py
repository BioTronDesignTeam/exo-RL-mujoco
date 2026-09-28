"""Exoskeleton assistance environment.

One control step (control_dt, default 10 ms = 100 Hz):
  1. Pipeline turns the policy action into an exo torque command, using only the
     (noisy) sensor readings the hardware would have.
  2. Command passes through a randomized actuator latency + torque tracking noise.
  3. The sim runs frame_skip physics substeps. Each substep, the human model computes
     its own torque to follow the reference, offloading the exo torque it felt on the
     previous control step (scaled by beta).
  4. Reward = alive - human effort - tracking - smoothness - magnitude.

exo_mode (reset option or constructor arg) is for evaluation baselines:
  "policy"       normal operation
  "off"          exo applies zero torque (human-alone baseline)
  "feedforward"  pipeline with action = 0 (hand-designed baseline only)

lean_info=True (training) returns only the small info fields training needs
(reward_terms, task, terminated_reason, tau_needed), so vector-env workers pickle little.

info["tau_needed"] (also self.tau_needed) is the torque the human needed at hip_l, knee_l,
hip_r, knee_r before offloading any exo torque, averaged over the control step. It is
privileged (sim-only) information for the warm-start teacher (exo_rl/warmstart.py).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import mujoco
import numpy as np

from ..control.observation import ObservationBuilder
from ..control.torque_pipeline import ExoTorquePipeline, LimbParams
from ..sim.exo_hardware import ExoHardware
from ..sim.human import HumanController
from ..sim.model_builder import HumanExoModel, SubjectParams
from ..sim.reference import ReferenceTrajectory, load_motions
from . import randomization
from .rewards import EffortReward

EXO_IN_HUMAN = [0, 1, 3, 4]      # exo joints (hip_l, knee_l, hip_r, knee_r) in human joint order


class ExoAssistEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None, exo_mode: str = "policy", lean_info: bool = False):
        super().__init__()
        self.cfg = cfg
        self.lean_info = lean_info
        self.rng = np.random.default_rng(seed)
        self.exo_mode = exo_mode
        sim = cfg["sim"]
        self.control_dt = float(sim["control_dt"])
        self.frame_skip = int(round(self.control_dt / float(sim["timestep"])))
        self.max_steps = int(round(float(sim["episode_seconds"]) / self.control_dt))
        tasks = cfg["tasks"]
        self.task_names = list(tasks["names"])
        extra = [n for n in tasks.get("extra_motions", []) if n not in self.task_names]
        self.motions = load_motions(self.task_names + extra, tasks.get("data_dir"))
        self.obs_builder = ObservationBuilder(cfg["observation"], self.control_dt)
        hw = cfg.get("exo_hardware") or {}
        self.exo_hw = ExoHardware(hw["spec"]) if hw.get("enabled") else None
        self.exo_quality = hw.get("quality", "lo")       # set to "hi" before reset() for viewing
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_builder.dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._build(randomization.nominal(cfg))

    # ------------------------------------------------------------------ setup
    def _build(self, ep: randomization.EpisodeParams):
        self.ep = ep
        sp = SubjectParams(height=ep.height, mass=ep.mass, payload_mass=ep.payload_mass,
                           payload_offset=ep.payload_offset, floor_friction=ep.floor_friction,
                           joint_damping=ep.joint_damping, timestep=float(self.cfg["sim"]["timestep"]),
                           exo=self.exo_hw, exo_mass_scale=ep.exo_mass_scale, exo_quality=self.exo_quality)
        self.hm = HumanExoModel(sp)
        self.human = HumanController(self.cfg["human"], ep.mass)
        self.human.set_randomization(ep.human_gain_scale, ep.exo_adaptation)
        if self.exo_hw is not None:     # feedforward compensates human leg + exo leg links
            self.limb = LimbParams(**self.exo_hw.limb_params(self.hm.segments, ep.exo_mass_scale))
        else:
            self.limb = LimbParams.from_segment_info(self.hm.segments)
        self.pipeline = ExoTorquePipeline(self.cfg["exo"], self.limb, self.control_dt)
        self.reward_fn = EffortReward(self.cfg["reward"], self.human.tau_max, self.pipeline.tau_max)

    def _make_reference(self, task: str | None) -> ReferenceTrajectory:
        tcfg = self.cfg["tasks"]
        blend = float(tcfg.get("blend_seconds", 0.5))
        if task is None and self.rng.random() < float(tcfg.get("circuit_prob", 0.0)):
            order = self._circuit_order(tcfg)
            lo, hi = tcfg.get("circuit_segment_seconds", [3.0, 5.0])
            segs = [(self.motions[n], float(self.rng.uniform(lo, hi))) for n in order]
            return ReferenceTrajectory(segs, blend=blend)
        if task is not None and task not in self.motions:
            self.motions.update(load_motions([task], tcfg.get("data_dir")))
        name = task or str(self.rng.choice(self.task_names))
        m = self.motions[name]
        return ReferenceTrajectory([(m, 1e9)], blend=blend, t_offset=float(self.rng.uniform(0, m.period)))

    def _circuit_order(self, tcfg) -> list:
        """Task order for a chained episode: the ACE course order, or a random permutation
        that avoids forbidden direct transitions (e.g. instant backward->forward reversal)."""
        if tcfg.get("circuit_order", "random") == "ace":
            return [n for n in tcfg.get("ace_order", self.task_names) if n in self.task_names]
        forbidden = {tuple(p) for p in tcfg.get("forbidden_transitions", [])}
        for _ in range(100):
            order = list(self.rng.permutation(self.task_names))
            if not any((a, b) in forbidden for a, b in zip(order, order[1:])):
                return order
        raise ValueError("could not build a circuit without forbidden transitions")

    def _v_ref(self, t: float) -> float:
        w = self.ref.weights(t)
        return float(sum(wi * self.speeds[m.name] for wi, (m, _) in zip(w, self.ref.segments)))

    def _read_sensors(self) -> dict:
        s = self.hm.sensors()
        n = self.ep.sensor_noise
        if any(v > 0 for v in n.values()):
            s["q_exo"] = s["q_exo"] + self.rng.normal(0, n["angle"], 4)
            s["qd_exo"] = s["qd_exo"] + self.rng.normal(0, n["velocity"], 4)
            s["trunk_pitch"] = s["trunk_pitch"] + self.rng.normal(0, n["angle"])
            for k in list(s):
                if k.startswith("gyro_"):
                    s[k] = s[k] + self.rng.normal(0, n["gyro"], 3)
                elif k.startswith("acc_"):
                    s[k] = s[k] + self.rng.normal(0, n["accel"], 3)
        return s

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.exo_mode = options.get("exo_mode", self.exo_mode)
        ep = randomization.nominal(self.cfg) if options.get("nominal") else randomization.sample(self.cfg, self.rng)
        self._build(ep)
        self.ref = self._make_reference(options.get("task"))
        overrides = {}
        for m, _ in self.ref.segments:
            overrides.update(getattr(m, "residual_overrides", {}) or {})
        self.human.residual = {**self.cfg["human"]["residual"], **overrides}
        self.speeds = {m.name: self.hm.implied_speed(m) for m, _ in self.ref.segments}

        self.t, self.steps = 0.0, 0
        q, qd, _, trunk, _ = self.ref.sample(0.0)
        self.hm.set_pose(q, trunk, qd, root_vx=self._v_ref(0.0))
        self.human.reset()
        self.pipeline.reset()
        self.obs_builder.reset()
        self.prev_action = np.zeros(4)
        self.tau_cmd_prev = np.zeros(4)
        self.human_felt_exo = np.zeros(6)
        self.latency = deque([np.zeros(4)] * ep.latency_steps)
        self.tau_needed = np.zeros(4)
        self.sensors = self._read_sensors()
        obs = self.obs_builder(self.sensors, self.prev_action)
        return obs, {"task": self.ref.name, "params": ep}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.exo_mode == "off":
            tau_cmd = np.zeros(4)
        else:
            a = np.zeros(4) if self.exo_mode == "feedforward" else action
            tau_cmd = self.pipeline(a, self.sensors["q_exo"], self.sensors["trunk_pitch"])

        # Actuator latency and torque tracking error (hardware effects, not in the pipeline).
        if self.ep.latency_steps > 0:
            self.latency.append(tau_cmd)
            tau_applied = self.latency.popleft()
        else:
            tau_applied = tau_cmd
        if self.ep.torque_noise_frac > 0:
            tau_applied = tau_applied * (1.0 + self.rng.normal(0, self.ep.torque_noise_frac, 4))
        tau_exo_h = np.zeros(6)
        tau_exo_h[EXO_IN_HUMAN] = tau_applied

        m, d, hm = self.hm.model, self.hm.data, self.hm
        dt = m.opt.timestep
        effort_sq = np.zeros(6)
        tau_h_sum = np.zeros(6)
        tau_needed_sum = np.zeros(6)
        resid_sum = np.zeros(3)
        for k in range(self.frame_skip):
            ts = self.t + k * dt
            q, qd, qdd, trunk, trunk_d = self.ref.sample(ts)
            tau_h, tau_res = self.human.compute(hm, q, qd, qdd, trunk, trunk_d, self._v_ref(ts),
                                                self.human_felt_exo, dt)
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[hm.human_dof] += tau_h + tau_exo_h
            d.qfrc_applied[hm.root_dof] += tau_res
            mujoco.mj_step(m, d)
            effort_sq += (tau_h / self.human.tau_max) ** 2
            tau_h_sum += tau_h
            tau_needed_sum += self.human.tau_needed
            resid_sum += np.abs(tau_res)
        self.human_felt_exo = tau_exo_h          # human reacts one control step later
        self.tau_needed = tau_needed_sum[EXO_IN_HUMAN] / self.frame_skip
        self.t += self.control_dt
        self.steps += 1

        q_ref, _, _, _, _ = self.ref.sample(self.t)
        q_err = d.qpos[hm.human_qpos] - q_ref
        reward, terms = self.reward_fn(effort_sq / self.frame_skip, q_err, tau_cmd, self.tau_cmd_prev)
        self.tau_cmd_prev = tau_cmd
        self.prev_action = action

        term_cfg = self.cfg["termination"]
        unstable = not np.all(np.isfinite(d.qpos))
        fell = d.qpos[hm.root_qpos[1]] < term_cfg["min_pelvis_height_frac"] * hm.segments.leg_len
        lost = np.max(np.abs(q_err)) > np.radians(term_cfg["max_joint_error_deg"])
        terminated = bool(unstable or fell or lost)
        truncated = self.steps >= self.max_steps
        if unstable:
            reward = 0.0

        self.sensors = self._read_sensors() if not unstable else self.sensors
        obs = self.obs_builder(self.sensors, self.prev_action)
        task, phase = self.ref.active(self.t)
        reason = "unstable" if unstable else "fell" if fell else "tracking" if lost else ""
        if self.lean_info:
            return obs, float(reward), terminated, truncated, {
                "reward_terms": terms, "task": task, "terminated_reason": reason, "tau_needed": self.tau_needed}
        info = {
            "reward_terms": terms,
            "tau_human": tau_h_sum / self.frame_skip,
            "tau_exo_cmd": tau_cmd,
            "q": d.qpos[hm.human_qpos].copy(),
            "q_ref": q_ref,
            "residual_abs": resid_sum / self.frame_skip,
            "task": task,
            "phase": phase,
            "raw_sensors": self.sensors,
            "tau_needed": self.tau_needed,
            "terminated_reason": reason,
        }
        return obs, float(reward), terminated, truncated, info
