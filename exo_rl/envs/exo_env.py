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
  "teacher"      privileged simulation-only torque-sharing baseline

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
from ..sim.reference import TRAVELING_MOTIONS, ReferenceTrajectory, load_motions
from . import randomization
from .rewards import EffortReward, muscle_joint_demand

EXO_IN_HUMAN = [0, 1, 3, 4]      # exo joints (hip_l, knee_l, hip_r, knee_r) in human joint order


class ExoAssistEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None, exo_mode: str = "policy", lean_info: bool = False):
        super().__init__()
        self.cfg = cfg
        human_model = cfg["human"].get("model", "torque")
        if human_model not in ("torque", "myolegs26", "myolegs", "myofullbody"):
            raise ValueError(f"Unsupported human.model: {human_model!r}")
        self.muscle_model = human_model != "torque"
        if not self.muscle_model and float(cfg["reward"].get("w_assisted_joint", 0.0)) > 0:
            raise ValueError("Assisted-joint muscle demand requires a Myo model")
        if self.muscle_model:
            if cfg.get("randomization", {}).get("enabled", False):
                raise ValueError("MyoLeg26 currently requires randomization.enabled=false (fixed anatomy)")
            if cfg.get("train", {}).get("sim_backend", "cpu") != "cpu":
                raise ValueError("MyoLeg26 currently requires train.sim_backend=cpu; Warp does not transfer muscle activation state")
        self.lean_info = lean_info
        self.rng = np.random.default_rng(seed)
        self.exo_mode = exo_mode
        sim = cfg["sim"]
        self.control_dt = float(sim["control_dt"])
        self.frame_skip = int(round(self.control_dt / float(sim["timestep"])))
        self.max_steps = int(round(float(sim["episode_seconds"]) / self.control_dt))
        self.diagnostics_enabled = bool(sim.get("diagnostics", False))
        self._settle_reference: tuple[np.ndarray, float] | None = None
        reset_cfg = cfg["human"].get("myo", {}).get("reset", {})
        self.start_phase = reset_cfg.get("start_phase")
        self.settle_seconds = float(reset_cfg.get("settle_seconds", 0.0))
        self.height_jitter = float(reset_cfg.get("height_jitter_m", 0.0))
        if (not np.isfinite(self.settle_seconds) or self.settle_seconds < 0
                or not np.isfinite(self.height_jitter) or self.height_jitter < 0):
            raise ValueError("Myo reset settling time and height jitter must be finite and nonnegative")
        if self.start_phase is not None:
            self.start_phase = float(self.start_phase)
            if not np.isfinite(self.start_phase) or not 0 <= self.start_phase < 1:
                raise ValueError("Myo reset.start_phase must be in [0, 1)")
        if self.settle_seconds and (not self.muscle_model or self.start_phase != 0.0):
            raise ValueError("Settled Myo reset requires start_phase=0 and a muscle model")
        if not np.isclose(self.settle_seconds / self.control_dt, round(self.settle_seconds / self.control_dt)):
            raise ValueError("Myo reset.settle_seconds must be a whole number of control steps")
        tasks = cfg["tasks"]
        self.task_names = list(tasks["names"])
        extra = [n for n in tasks.get("extra_motions", []) if n not in self.task_names]
        self.motions = load_motions(self.task_names + extra, tasks.get("data_dir"), tasks.get("periods"))
        self.obs_builder = ObservationBuilder(cfg["observation"], self.control_dt)
        hw = cfg.get("exo_hardware") or {}
        self.exo_hw = ExoHardware(hw["spec"]) if hw.get("enabled") else None
        self.exo_quality = hw.get("quality", "lo")       # set to "hi" before reset() for viewing
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_builder.dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._build(randomization.nominal(cfg))

    # ------------------------------------------------------------------ setup
    def _build(self, ep: randomization.EpisodeParams, flat_ground: bool = False):
        self.ep = ep
        sp = SubjectParams(height=ep.height, mass=ep.mass, payload_mass=ep.payload_mass,
                           payload_offset=ep.payload_offset, floor_friction=ep.floor_friction,
                           joint_damping=ep.joint_damping, timestep=float(self.cfg["sim"]["timestep"]),
                           exo=self.exo_hw, exo_mass_scale=ep.exo_mass_scale, exo_quality=self.exo_quality)
        if self.muscle_model:
            from ..sim.myoleg import MuscleController, MyoLegModel
            myo = self.cfg["human"]["myo"]
            self.hm = MyoLegModel(sp, myo["source_dir"], float(myo.get("scale", 1.0)),
                                  self.cfg["human"]["model"], bool(myo.get("lock_upper_body", False)),
                                  myo.get("ankle_dorsiflexion_deg"), flat_ground=flat_ground,
                                  torso_control=myo.get("torso_control"))
            self.human = MuscleController(self.cfg["human"], ep.mass)
            self.human.diagnostics_enabled = self.diagnostics_enabled
        else:
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
            self.motions.update(load_motions([task], tcfg.get("data_dir"), tcfg.get("periods")))
        name = task or str(self.rng.choice(self.task_names))
        m = self.motions[name]
        return ReferenceTrajectory([(m, 1e9)], blend=blend, t_offset=float(self.rng.uniform(0, m.period)),
                                   startup_seconds=tcfg.get("startup_seconds", 0.0))

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
        if self._settle_reference is not None:
            return 0.0
        w = self.ref.weights(t)
        return float(sum(wi * self.speeds[m.name] for wi, (m, _) in zip(w, self.ref.segments))) * self.ref.clock_rate(t)

    def _sample_reference(self, t: float):
        if self._settle_reference is not None:
            q, trunk = self._settle_reference
            return q, np.zeros(6), np.zeros(6), trunk, 0.0
        return self.ref.sample(t)

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
        if self.muscle_model:
            self.ref = self._make_reference(options.get("task"))
            flat_ground = any(m.name in TRAVELING_MOTIONS for m, _ in self.ref.segments)
            self._build(ep, flat_ground=flat_ground or options.get("flat_ground", False)
                        or self.cfg["sim"].get("flat_ground", False))
        else:
            self._build(ep)
            self.ref = self._make_reference(options.get("task"))
        if self.start_phase is not None:
            if len(self.ref.segments) != 1:
                raise ValueError("A fixed Myo start_phase requires a single motion")
            self.ref.t_offset = self.start_phase * self.ref.segments[0][0].period
        if self.settle_seconds and self.ref.name != "squat" and not self.ref.startup_seconds:
            raise ValueError("Settled Myo reset requires a single squat or a single motion with startup_seconds")
        overrides = {}
        for m, _ in self.ref.segments:
            overrides.update(getattr(m, "residual_overrides", {}) or {})
        self.human.residual = {**self.cfg["human"]["residual"], **overrides}
        self.speeds = {m.name: self.hm.implied_speed(m) for m, _ in self.ref.segments}

        self.t, self.steps = 0.0, 0
        q, qd, _, trunk, trunk_d = self.ref.sample(0.0)
        if self.muscle_model and self.cfg["human"]["myo"].get("initialize_reference_velocity", False):
            self.hm.set_pose(q, trunk, qd, root_vx=self._v_ref(0.0), trunk_velocity=trunk_d)
        else:
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
        if self.settle_seconds:
            mode, horizon = self.exo_mode, self.max_steps
            self._settle_reference = (q.copy(), float(trunk))
            self.exo_mode = "off"
            settle_steps = int(round(self.settle_seconds / self.control_dt))
            self.max_steps = max(horizon, settle_steps + 1)
            try:
                for _ in range(settle_steps):
                    _, _, terminated, _, info = self.step(np.zeros(4))
                    if terminated:
                        raise RuntimeError(f"Myo settling failed: {info['terminated_reason']}")
            finally:
                self._settle_reference = None
                self.exo_mode, self.max_steps = mode, horizon
            # Preserve the physically settled activations, integrators and
            # contact state; only the episode clock and command history restart.
            self.t, self.steps, self.hm.data.time = 0.0, 0, 0.0
            self.pipeline.reset()
            self.prev_action = np.zeros(4)
            self.tau_cmd_prev = np.zeros(4)
            self.human_felt_exo = np.zeros(6)
            self.tau_needed = np.zeros(4)
        height_offset = self.rng.uniform(-self.height_jitter, self.height_jitter) if self.height_jitter else 0.0
        if height_offset:
            self.hm.data.qpos[self.hm.root_qpos[1]] += height_offset
            mujoco.mj_forward(self.hm.model, self.hm.data)
        if self.settle_seconds or height_offset:
            self.sensors = self._read_sensors()
            self.obs_builder.reset()
            obs = self.obs_builder(self.sensors, self.prev_action)
        reset_info = {"task": self.ref.name, "params": ep}
        if self.settle_seconds:
            reset_info["reset_diagnostics"] = {
                "settle_seconds": self.settle_seconds, "initial_phase": self.ref.active(0)[1],
                "height_offset_m": height_offset, "activation_rms": float(np.sqrt(np.mean(self.hm.data.act ** 2))),
                "contacts": int(self.hm.data.ncon),
            }
        return obs, reset_info

    def step(self, action):
        # Drive the same control-step generator with standard CPU physics.
        step = self.step_substeps(action)
        try:
            while True:
                model, data = next(step)
                mujoco.mj_step(model, data)
        except StopIteration as done:
            return done.value

    def step_substeps(self, action):
        """Yield before each physics step; the runner supplies CPU or batched GPU physics."""
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.exo_mode == "off":
            tau_cmd = np.zeros(4)
        else:
            if self.exo_mode == "teacher":
                fraction = float(self.cfg.get("warmstart", {}).get("teacher_fraction", 0.2))
                a = np.clip(np.divide(fraction * self.tau_needed, self.pipeline.action_scale,
                                      out=np.zeros(4), where=self.pipeline.action_scale != 0), -1, 1)
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
        diagnostics = {}
        assisted_demand = 0.0
        for k in range(self.frame_skip):
            ts = self.t + k * dt
            q, qd, qdd, trunk, trunk_d = self._sample_reference(ts)
            tau_h, tau_res = self.human.compute(hm, q, qd, qdd, trunk, trunk_d, self._v_ref(ts),
                                                self.human_felt_exo, dt)
            lumbar_dof = getattr(hm, "lumbar_dof", None)
            lumbar_torque = float(getattr(self.human, "lumbar_torque", 0.))
            if self.muscle_model and (self.diagnostics_enabled or self.reward_fn.w_assisted_joint > 0):
                demand = muscle_joint_demand(hm.coordinate_map().T @ hm.moment.T,
                                             d.actuator_force, self.human.tau_max, EXO_IN_HUMAN)
                assisted_demand += demand / self.frame_skip
            if self.diagnostics_enabled:
                for name, value in {
                    "helper_x_force_sq_N2": tau_res[0] ** 2,
                    "helper_pitch_torque_sq_Nm2": tau_res[2] ** 2 + lumbar_torque ** 2,
                }.items():
                    diagnostics[name] = diagnostics.get(name, 0.0) + float(value) / self.frame_skip
                helper_power = tau_res * d.qvel[hm.root_dof]
                if lumbar_dof is not None:
                    helper_power = np.r_[helper_power, lumbar_torque * d.qvel[lumbar_dof]]
                    diagnostics["helper_lumbar_torque_sq_Nm2"] = diagnostics.get("helper_lumbar_torque_sq_Nm2", 0.) + lumbar_torque ** 2 / self.frame_skip
                velocity = hm.signs * d.qvel[hm.human_dof] if self.muscle_model else d.qvel[hm.human_dof]
                exo_power = tau_exo_h * velocity
                human_power = d.actuator_force * d.actuator_velocity if self.muscle_model else tau_h * velocity
                work = {
                    "helper_abs_work_J": float(np.abs(helper_power).sum() * dt),
                    "helper_positive_work_J": float(np.maximum(helper_power, 0).sum() * dt),
                    "exo_abs_work_J": float(np.abs(exo_power).sum() * dt),
                    "exo_positive_work_J": float(np.maximum(exo_power, 0).sum() * dt),
                    "human_positive_work_J": float(np.maximum(human_power, 0).sum() * dt),
                }
                for name, value in work.items():
                    diagnostics[name] = diagnostics.get(name, 0.0) + value
                if self.muscle_model:
                    for name, value in self.human.last_recruitment_diagnostics.items():
                        diagnostics[name] = diagnostics.get(name, 0.0) + value / self.frame_skip
            d.qfrc_applied[:] = 0.0
            if self.muscle_model:
                # Only exo torque is applied directly; human forces come from muscles.
                d.qfrc_applied[hm.human_dof] += hm.signs * tau_exo_h
            else:
                d.qfrc_applied[hm.human_dof] += tau_h + tau_exo_h
            d.qfrc_applied[hm.root_dof] += tau_res
            if lumbar_dof is not None:
                d.qfrc_applied[lumbar_dof] += lumbar_torque
            yield m, d
            if self.muscle_model:
                effort_sq += float(np.mean(d.act ** 2))
            else:
                effort_sq += (tau_h / self.human.tau_max) ** 2
            tau_h_sum += tau_h
            tau_needed_sum += self.human.tau_needed
            resid_sum += np.abs(tau_res)
        self.human_felt_exo = tau_exo_h          # human reacts one control step later
        self.tau_needed = tau_needed_sum[EXO_IN_HUMAN] / self.frame_skip
        self.t += self.control_dt
        self.steps += 1

        q_ref, _, _, _, _ = self._sample_reference(self.t)
        human_q = hm.joint_positions() if self.muscle_model else d.qpos[hm.human_qpos]
        q_err = human_q - q_ref
        if self.diagnostics_enabled and self.muscle_model:
            diagnostics["assisted_joint_demand"] = assisted_demand
        if self.reward_fn.w_assisted_joint > 0:
            reward, terms = self.reward_fn(effort_sq / self.frame_skip, q_err, tau_cmd, self.tau_cmd_prev,
                                           assisted_joint_demand=assisted_demand)
        else:
            reward, terms = self.reward_fn(effort_sq / self.frame_skip, q_err, tau_cmd, self.tau_cmd_prev)
        self.tau_cmd_prev = tau_cmd
        self.prev_action = self.pipeline.observation_action(a if self.exo_mode == "teacher" else action)

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
            info = {
                "reward_terms": terms, "task": task, "terminated_reason": reason, "tau_needed": self.tau_needed}
            if self.diagnostics_enabled:
                info["diagnostics"] = diagnostics
            return obs, float(reward), terminated, truncated, info
        info = {
            "reward_terms": terms,
            "tau_human": tau_h_sum / self.frame_skip,
            "tau_exo_cmd": tau_cmd,
            "q": human_q.copy(),
            "q_ref": q_ref,
            "residual_abs": resid_sum / self.frame_skip,
            "task": task,
            "phase": phase,
            "raw_sensors": self.sensors,
            "tau_needed": self.tau_needed,
            "terminated_reason": reason,
        }
        if self.muscle_model:
            info["muscle_activation"] = d.act.copy()
            info["muscle_excitation"] = d.ctrl.copy()
            info["effort_metric"] = "mean_squared_muscle_activation"
            if getattr(hm, "articulated_torso", False):
                info["posture"] = hm.posture()
                info["effort_metric"] = "mean_squared_leg_muscle_activation"
        if self.diagnostics_enabled:
            info["diagnostics"] = diagnostics
        return obs, float(reward), terminated, truncated, info
