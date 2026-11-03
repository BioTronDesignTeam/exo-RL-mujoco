"""Effort-based reward (TODO 1).

    r = alive
        - w_effort    * mean((tau_human / tau_human_max)^2)     primary: human effort
        - w_tracking  * mean((q - q_ref)^2) / sigma^2           keep the motion intact
        - w_smooth    * mean((d tau_exo / tau_exo_max)^2)       no jerky torque
        - w_magnitude * mean((tau_exo / tau_exo_max)^2)         don't push harder than needed

Notes
  * The old r_torque = -|T_target - T_user| term is intentionally gone: it only rewarded
    reproducing the hand-designed feedforward.
  * Tracking is secondary. The human model already tracks the reference; this term
    penalises exo torque that knocks the human off their motion (matters when the
    randomized human adaptation beta < 1).
  * `alive` keeps per-step reward positive so early termination is never attractive.
  * Every term is returned separately so it can be logged (see scripts/train.py).
"""
from __future__ import annotations

import numpy as np


def muscle_joint_demand(moment_arms: np.ndarray, muscle_force: np.ndarray,
                        torque_scale: np.ndarray, joints: list[int]) -> float:
    """Measure absolute muscular moment demand at selected assisted joints.

    Antagonist moments count separately instead of cancelling. The transmission
    moment arms already include the adapter's coupled attachment coordinates.
    This is a mechanical-demand proxy, not joint contact load or metabolic cost.
    """
    moments = np.abs(moment_arms[joints] * muscle_force[None, :]).sum(axis=1)
    return float(np.mean(moments / torque_scale[joints]))


class EffortReward:
    def __init__(self, cfg: dict, tau_human_max: np.ndarray, tau_exo_max: np.ndarray):
        self.alive = float(cfg.get("alive", 1.0))
        self.w = {k: float(cfg[f"w_{k}"]) for k in ("effort", "tracking", "smooth", "magnitude")}
        self.w_assisted_joint = float(cfg.get("w_assisted_joint", 0.0))
        if not np.isfinite(self.w_assisted_joint) or self.w_assisted_joint < 0:
            raise ValueError("reward.w_assisted_joint must be finite and nonnegative")
        self.sigma = np.radians(float(cfg.get("tracking_sigma_deg", 10.0)))
        self.tau_human_max = np.asarray(tau_human_max, dtype=float)
        self.tau_exo_max = np.asarray(cfg.get("exo_torque_reference", tau_exo_max), dtype=float)
        if (self.tau_exo_max.shape != np.asarray(tau_exo_max).shape
                or not np.isfinite(self.tau_exo_max).all() or np.any(self.tau_exo_max <= 0)):
            raise ValueError("reward.exo_torque_reference must contain one finite positive value per exo joint")

    def __call__(self, effort_sq: np.ndarray, q_err: np.ndarray, tau_exo: np.ndarray,
                 tau_exo_prev: np.ndarray, *, assisted_joint_demand: float | None = None):
        """effort_sq: per-joint mean over sim substeps of (tau_human / tau_human_max)^2."""
        terms = {
            "effort": float(np.mean(effort_sq)),
            "tracking": float(np.mean(q_err ** 2) / self.sigma ** 2),
            "smooth": float(np.mean(((tau_exo - tau_exo_prev) / self.tau_exo_max) ** 2)),
            "magnitude": float(np.mean((tau_exo / self.tau_exo_max) ** 2)),
        }
        reward = self.alive - sum(self.w[k] * v for k, v in terms.items())
        if self.w_assisted_joint > 0:
            if assisted_joint_demand is None or not np.isfinite(assisted_joint_demand) or assisted_joint_demand < 0:
                raise ValueError("Assisted-joint reward requires a finite nonnegative muscle demand")
            terms["assisted_joint_demand"] = float(assisted_joint_demand)
            reward -= self.w_assisted_joint * assisted_joint_demand
        return reward, terms
