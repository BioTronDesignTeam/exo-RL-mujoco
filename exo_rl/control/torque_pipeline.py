"""Exo torque pipeline, shared by the simulator and the real hardware.

    tau_cmd = alpha * tau_ff(runtime signals) + action_scale * action
    -> direction constraint (e.g. cable can only pull)
    -> saturation |tau| <= tau_max
    -> rate limit |d tau / dt| <= rate_limit

Rules that keep training and deployment identical:
  * Only signals available at runtime are used (measured joint angles, trunk pitch
    estimate). No recorded/reference motion data enters this pipeline (the hardware has none).
  * The env and deploy/controller.py both construct this same class from the same config.
  * Hardware must still enforce its own independent torque/rate limits in firmware.

Exo joint order: hip_l, knee_l, hip_r, knee_r. Sign: + = flexion torque.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class LimbParams:
    """Leg inertial parameters for the analytic gravity feedforward (per leg)."""
    thigh_mass: float
    thigh_len: float
    thigh_com: float
    shank_foot_mass: float
    shank_foot_com: float

    @classmethod
    def from_segment_info(cls, seg) -> "LimbParams":
        return cls(seg.thigh_mass, seg.thigh_len, seg.thigh_com, seg.shank_foot_mass, seg.shank_foot_com)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def limb_gravity_torque(hip, knee, trunk_pitch, lp: LimbParams):
    """Torques (hip, knee) that cancel the gravity load of a free-swinging leg.

    Absolute segment angles from vertical (+ = forward): thigh = hip - pitch,
    shank = hip - pitch - knee. Valid for swing; in stance the leg is supported, which
    is why this term is weighted by alpha and not trusted on its own.
    """
    th_t = hip - trunk_pitch
    th_s = th_t - knee
    m_sf = lp.shank_foot_mass
    tau_hip = G * ((lp.thigh_mass * lp.thigh_com + m_sf * lp.thigh_len) * np.sin(th_t)
                   + m_sf * lp.shank_foot_com * np.sin(th_s))
    tau_knee = -G * m_sf * lp.shank_foot_com * np.sin(th_s)
    return tau_hip, tau_knee


class ExoTorquePipeline:
    DIRECTIONS = ("both", "flexion_only", "extension_only")

    def __init__(self, cfg: dict, limb: LimbParams | None, control_dt: float):
        self.alpha = float(cfg.get("alpha", 0.0))
        self.feedforward = cfg.get("feedforward", "none")
        self.tau_max = np.asarray(cfg["max_torque"], dtype=float)
        self.action_scale = np.asarray(cfg.get("action_scale", cfg["max_torque"]), dtype=float)
        self.rate_limit = float(cfg["rate_limit"])
        self.directions = list(cfg.get("direction", ["both"] * 4))
        for d in self.directions:
            if d not in self.DIRECTIONS:
                raise ValueError(f"direction must be one of {self.DIRECTIONS}, got {d}")
        self.limb = limb
        self.dt = control_dt
        if self.feedforward == "limb_gravity" and limb is None:
            raise ValueError("limb_gravity feedforward needs LimbParams")
        self.reset()

    def reset(self):
        self.prev = np.zeros(4)

    def feedforward_torque(self, q_exo: np.ndarray, trunk_pitch: float) -> np.ndarray:
        if self.feedforward == "none" or self.alpha == 0.0:
            return np.zeros(4)
        if self.feedforward == "limb_gravity":
            hl, kl = limb_gravity_torque(q_exo[0], q_exo[1], trunk_pitch, self.limb)
            hr, kr = limb_gravity_torque(q_exo[2], q_exo[3], trunk_pitch, self.limb)
            return np.array([hl, kl, hr, kr])
        raise ValueError(f"unknown feedforward '{self.feedforward}'")

    def __call__(self, action: np.ndarray, q_exo: np.ndarray, trunk_pitch: float) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        tau = self.alpha * self.feedforward_torque(q_exo, trunk_pitch) + self.action_scale * action
        for i, d in enumerate(self.directions):
            if d == "flexion_only":
                tau[i] = max(tau[i], 0.0)
            elif d == "extension_only":
                tau[i] = min(tau[i], 0.0)
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        max_step = self.rate_limit * self.dt
        tau = self.prev + np.clip(tau - self.prev, -max_step, max_step)
        self.prev = tau.copy()
        return tau
