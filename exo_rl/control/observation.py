"""Observation construction, shared by the simulator and the hardware (TODO 3, TODO 4).

Input is a raw sensor dict with the same keys in sim (HumanExoModel.sensors()) and on
hardware (deploy/hardware_interface.py):
    q_exo[4], qd_exo[4]           encoder angles / velocities, exo joint order
    trunk_pitch                   trunk pitch estimate from the pelvis IMU
    gyro_{pelvis,thigh_l,thigh_r}[3], acc_{pelvis,thigh_l,thigh_r}[3]

Per frame:  scaled sensors (+ previous action) (+ gait-phase features)
Output:     the last `history` frames stacked, oldest first.

The history window gives the policy temporal context so it can tell a squat from a
high knee or backward step from a single ambiguous instant.
"""
from __future__ import annotations

from collections import deque

import numpy as np

IMU_KEYS = ["pelvis", "thigh_l", "thigh_r"]

# Fixed scales (not running statistics) so the exact same numbers are used on hardware.
SCALE = {"q": 1.0 / np.pi, "qd": 1.0 / 10.0, "pitch": 1.0 / np.pi, "gyro": 1.0 / 10.0, "acc": 1.0 / 20.0}


class PhaseEstimator:
    """Gait phase per leg from the hip-angle phase portrait: phi = atan2(-v_n, q_n).

    Center and amplitude are tracked with slow exponential averages so the estimate adapts
    to each task and user without any reference data.
    """

    def __init__(self, dt: float, tau: float = 2.0):
        self.a = dt / tau
        self.reset()

    def reset(self):
        self.mean_q = None
        self.amp_q = np.full(2, 0.3)
        self.amp_v = np.full(2, 2.0)

    def __call__(self, hip: np.ndarray, hip_vel: np.ndarray) -> np.ndarray:
        if self.mean_q is None:
            self.mean_q = hip.copy()
        a = self.a
        self.mean_q += a * (hip - self.mean_q)
        dq = hip - self.mean_q
        self.amp_q += a * (np.abs(dq) * np.sqrt(2) - self.amp_q)
        self.amp_v += a * (np.abs(hip_vel) * np.sqrt(2) - self.amp_v)
        qn = dq / np.maximum(self.amp_q, 0.05)
        vn = hip_vel / np.maximum(self.amp_v, 0.5)
        phi = np.arctan2(-vn, qn)
        return np.concatenate([np.sin(phi), np.cos(phi)])


class ObservationBuilder:
    def __init__(self, cfg: dict, control_dt: float):
        self.history = int(cfg.get("history", 8))
        self.use_phase = bool(cfg.get("use_phase", True))
        self.include_prev_action = bool(cfg.get("include_prev_action", True))
        self.phase = PhaseEstimator(control_dt) if self.use_phase else None
        self.frame_dim = 4 + 4 + 1 + 9 + 9 + (4 if self.include_prev_action else 0) + (4 if self.use_phase else 0)
        self.dim = self.frame_dim * self.history
        self.frames: deque = deque(maxlen=self.history)

    def reset(self):
        self.frames.clear()
        if self.phase is not None:
            self.phase.reset()

    def frame(self, s: dict, prev_action: np.ndarray) -> np.ndarray:
        parts = [
            np.asarray(s["q_exo"]) * SCALE["q"],
            np.asarray(s["qd_exo"]) * SCALE["qd"],
            [s["trunk_pitch"] * SCALE["pitch"]],
        ]
        parts += [np.asarray(s[f"gyro_{k}"]) * SCALE["gyro"] for k in IMU_KEYS]
        parts += [np.asarray(s[f"acc_{k}"]) * SCALE["acc"] for k in IMU_KEYS]
        if self.include_prev_action:
            parts.append(prev_action)
        if self.use_phase:
            q, qd = np.asarray(s["q_exo"]), np.asarray(s["qd_exo"])
            parts.append(self.phase(q[[0, 2]], qd[[0, 2]]))
        return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)

    def __call__(self, s: dict, prev_action: np.ndarray) -> np.ndarray:
        f = self.frame(s, prev_action)
        if not self.frames:
            for _ in range(self.history):
                self.frames.append(f)
        else:
            self.frames.append(f)
        return np.concatenate(self.frames)
