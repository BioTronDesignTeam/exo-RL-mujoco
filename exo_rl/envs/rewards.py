"""Effort-based reward.

    r = alive
        - w_effort    * mean((tau_human / tau_human_max)^2)     primary: human effort
        - w_tracking  * mean((q - q_ref)^2) / sigma^2           keep the motion intact
        - w_smooth    * mean((d tau_exo / tau_exo_max)^2)       no jerky torque
        - w_magnitude * mean((tau_exo / tau_exo_max)^2)         don't push harder than needed

Notes
  * There is no term rewarding a match to a hand-designed target torque
    (-|T_target - T_user|): it would only teach the policy to copy the feedforward.
  * Tracking is secondary. The human model already tracks the reference; this term
    penalises exo torque that knocks the human off their motion (matters when the
    randomized human adaptation beta < 1).
  * `alive` keeps per-step reward positive so early termination is never attractive.
  * Every term is returned separately so it can be logged (see exo_rl/training.py, episodes.csv).
"""
from __future__ import annotations

import numpy as np


class EffortReward:
    def __init__(self, cfg: dict, tau_human_max: np.ndarray, tau_exo_max: np.ndarray):
        self.alive = float(cfg.get("alive", 1.0))
        self.w = {k: float(cfg[f"w_{k}"]) for k in ("effort", "tracking", "smooth", "magnitude")}
        self.sigma = np.radians(float(cfg.get("tracking_sigma_deg", 10.0)))
        self.tau_human_max = np.asarray(tau_human_max, dtype=float)
        self.tau_exo_max = np.asarray(tau_exo_max, dtype=float)

    def __call__(self, effort_sq: np.ndarray, q_err: np.ndarray, tau_exo: np.ndarray,
                 tau_exo_prev: np.ndarray):
        """effort_sq: per-joint mean over sim substeps of (tau_human / tau_human_max)^2."""
        terms = {
            "effort": float(np.mean(effort_sq)),
            "tracking": float(np.mean(q_err ** 2) / self.sigma ** 2),
            "smooth": float(np.mean(((tau_exo - tau_exo_prev) / self.tau_exo_max) ** 2)),
            "magnitude": float(np.mean((tau_exo / self.tau_exo_max) ** 2)),
        }
        reward = self.alive - sum(self.w[k] * v for k, v in terms.items())
        return reward, terms
