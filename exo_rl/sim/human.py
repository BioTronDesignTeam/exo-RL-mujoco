"""Active human model, Level 1 fidelity (TODO 2).

The simulated human produces its own joint torques to follow the reference motion:

    tau_needed = M(q)·qdd_ref + Kp*(q_ref - q) + Kd*(qd_ref - qd) + Ki*∫e + bias(q, qd)
    tau_human  = clip(tau_needed - beta * tau_exo_prev, ±tau_max)

M(q)·qdd_ref and bias() (gravity + Coriolis) are inertial/gravity feedforward: the human
"knows" its own body, which removes the phase lag a pure PD tracker would have;
the integral term carries quasi-static loads such as body weight in a squat.

beta (exo_adaptation) models how well the human lets the exo help:
    beta = 1.0  -> human instantly offloads exo torque: kinematics unchanged, effort drops.
    beta = 0.0  -> human ignores the exo: exo torque perturbs motion, feedback fights it.
Randomizing beta in training (see randomization.py) keeps the policy from relying on a
perfectly cooperative human. The exo torque the human reacts to is one control step old
(reaction delay).

Root residuals (NOT human effort, logged separately):
    root_x      gentle velocity helper toward the speed implied by the reference; stands in
                for the balance/foot-placement strategy a planar leg model lacks
    root_pitch  trunk-orientation helper standing in for upper-body balance
    root_z      is left unactuated so the legs must carry body weight + payload.
Keep an eye on residual magnitudes in the logs: large residuals mean the reference and the
model disagree and the effort numbers are less trustworthy.

Upgrade path: replace this class with a musculoskeletal model (MyoSuite) and report
muscle activations as the effort signal; the env/reward interfaces stay the same.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .model_builder import HUMAN_JOINTS, HumanExoModel


def _per_joint(cfg_dict: dict) -> np.ndarray:
    return np.array([cfg_dict[j.split("_")[0]] for j in HUMAN_JOINTS], dtype=float)


class HumanController:
    def __init__(self, cfg: dict, subject_mass: float):
        mass_scale = subject_mass / 75.0
        self.kp0 = _per_joint(cfg["kp"]) * mass_scale
        self.kd0 = _per_joint(cfg["kd"]) * mass_scale
        self.ki0 = _per_joint(cfg["ki"]) * mass_scale
        self.tau_max = _per_joint(cfg["max_torque"]) * mass_scale
        self.i_limit = cfg.get("integral_limit_frac", 0.5) * self.tau_max
        self.residual = cfg["residual"]
        self.mass_scale = mass_scale
        self.set_randomization(gain_scale=1.0, beta=cfg.get("exo_adaptation", 1.0))
        self.reset()

    def set_randomization(self, gain_scale: float, beta: float):
        self.kp = self.kp0 * gain_scale
        self.kd = self.kd0 * gain_scale
        self.ki = self.ki0 * gain_scale
        self.beta = beta

    def reset(self):
        self.integral = np.zeros(len(HUMAN_JOINTS))

    def compute(self, hm: HumanExoModel, q_ref, qd_ref, qdd_ref, trunk_ref, trunk_d_ref,
                v_ref: float, tau_exo_human_order: np.ndarray, dt: float):
        """Returns (tau_human[6], tau_residual[3]) as generalized forces."""
        d = hm.data
        q = d.qpos[hm.human_qpos]
        qd = d.qvel[hm.human_dof]
        e = q_ref - q
        ed = qd_ref - qd

        self.integral = np.clip(self.integral + e * dt, -self.i_limit / np.maximum(self.ki, 1e-9),
                                self.i_limit / np.maximum(self.ki, 1e-9))
        bias = d.qfrc_bias[hm.human_dof]
        qdd_des = np.zeros(hm.model.nv)
        qdd_des[hm.human_dof] = qdd_ref
        m_qdd = np.zeros(hm.model.nv)
        mujoco.mj_mulM(hm.model, d, m_qdd, qdd_des)
        tau_ff = m_qdd[hm.human_dof] + bias
        tau_needed = tau_ff + self.kp * e + self.kd * ed + self.ki * self.integral
        tau_h = np.clip(tau_needed - self.beta * tau_exo_human_order, -self.tau_max, self.tau_max)
        self.tau_needed = tau_needed        # read-only export (before exo offloading); not used here

        # Root residuals (x: velocity helper; pitch: trunk helper; z: none).
        r = self.residual
        s = self.mass_scale
        pitch = d.qpos[hm.root_qpos[2]]
        xd, pitchd = d.qvel[hm.root_dof[0]], d.qvel[hm.root_dof[2]]
        f_x = s * r["x_kv"] * (v_ref - xd)
        t_pitch = s * (r["pitch_kp"] * (trunk_ref - pitch) + r["pitch_kd"] * (trunk_d_ref - pitchd))
        return tau_h, np.array([f_x, 0.0, t_pitch])
