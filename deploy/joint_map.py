"""Motor <-> human joint convention mapping for the real exo.

The policy, pipeline, and sensors all use the HUMAN convention:
    order hip_l, knee_l, hip_r, knee_r;  + = flexion;  0 = standing straight
Your motors use their own direction and encoder zero. MotorMap converts between them:
    q_human   = sign * (q_motor - zero_offset)
    qd_human  = sign * qd_motor
    tau_motor = sign * tau_human

Signs are derived from the URDF joint axes (exo_rl/sim/exo_hardware.py) and exported in
deploy_config.json["motor_map"]. Zero offsets come from configs/exo_mk1_fa900.yaml
(motor_zero_offsets): stand the pilot straight, read the encoders, and write them there.

Before powering on, verify each joint by hand: flex the pilot's left hip with the exo
unpowered and confirm q_human[0] increases. Then command a small (+2 Nm) torque with the
exo off the pilot and confirm the left thigh link swings forward.
"""
from __future__ import annotations

import numpy as np


class MotorMap:
    def __init__(self, signs, zero_offsets=None, order=("hip_l", "knee_l", "hip_r", "knee_r")):
        self.order = list(order)
        self.signs = np.asarray(signs, dtype=float)
        self.zero = np.zeros(4) if zero_offsets is None else np.asarray(zero_offsets, dtype=float)
        if not np.all(np.isin(self.signs, [-1.0, 1.0])):
            raise ValueError("motor signs must be +1 or -1")

    @classmethod
    def from_config(cls, deploy_cfg: dict) -> "MotorMap":
        mm = deploy_cfg.get("motor_map")
        if mm is None:
            raise KeyError("deploy_config.json has no motor_map; export with exo_hardware enabled")
        return cls(mm["signs"], mm["zero_offsets"], mm["order"])

    def q_to_human(self, q_motor):
        return self.signs * (np.asarray(q_motor, dtype=float) - self.zero)

    def qd_to_human(self, qd_motor):
        return self.signs * np.asarray(qd_motor, dtype=float)

    def tau_to_motor(self, tau_human):
        return self.signs * np.asarray(tau_human, dtype=float)
