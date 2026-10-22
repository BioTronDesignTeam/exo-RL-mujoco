"""Domain randomization, including the payload.

Sampled once per episode. Ranges live in configs/*.yaml under `randomization`.
Center the actuator/sensor ranges on values measured from your hardware (system ID)
before trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeParams:
    height: float
    mass: float
    payload_mass: float
    payload_offset: tuple
    floor_friction: float
    joint_damping: float
    human_gain_scale: float
    exo_adaptation: float        # beta in human.py
    latency_steps: int           # actuator delay in control steps
    torque_noise_frac: float
    sensor_noise: dict
    exo_mass_scale: float = 1.0  # CAD exo link masses/inertias (if exo_hardware is enabled)


def nominal(cfg: dict) -> EpisodeParams:
    subj, pay, sim, hum = cfg["subject"], cfg["payload"], cfg["sim"], cfg["human"]
    return EpisodeParams(
        height=subj["height"], mass=subj["mass"],
        payload_mass=pay["mass"] if pay["enabled"] else 0.0,
        payload_offset=tuple(pay["offset"]),
        floor_friction=sim["floor_friction"], joint_damping=sim.get("joint_damping", 0.5),
        human_gain_scale=1.0, exo_adaptation=hum.get("exo_adaptation", 1.0),
        latency_steps=0, torque_noise_frac=0.0,
        sensor_noise={"angle": 0.0, "velocity": 0.0, "gyro": 0.0, "accel": 0.0},
        exo_mass_scale=float((cfg.get("exo_hardware") or {}).get("mass_scale", 1.0)),
    )


def sample(cfg: dict, rng: np.random.Generator) -> EpisodeParams:
    p = nominal(cfg)
    r = cfg.get("randomization", {})
    if not r.get("enabled", False):
        return p

    def u(key, default):
        lo, hi = r.get(key, default)
        return float(rng.uniform(lo, hi))

    p.height = u("height", [p.height, p.height])
    p.mass = u("mass", [p.mass, p.mass])
    if p.payload_mass > 0:
        p.payload_mass *= u("payload_mass_scale", [1.0, 1.0])
        off = float(r.get("payload_offset_m", 0.0))
        p.payload_offset = tuple(np.asarray(p.payload_offset) + rng.uniform(-off, off, 3) * [1, 0, 1])
    p.floor_friction = u("floor_friction", [p.floor_friction, p.floor_friction])
    p.joint_damping *= u("joint_damping_scale", [1.0, 1.0])
    p.human_gain_scale = u("human_gain_scale", [1.0, 1.0])
    p.exo_adaptation = u("exo_adaptation", [p.exo_adaptation, p.exo_adaptation])
    lo, hi = r.get("latency_steps", [0, 0])
    p.latency_steps = int(rng.integers(lo, hi + 1))
    p.torque_noise_frac = float(r.get("torque_noise_frac", 0.0))
    p.sensor_noise = dict(r.get("sensor_noise", p.sensor_noise))
    p.exo_mass_scale *= u("exo_mass_scale", [1.0, 1.0])     # CAD uncertainty around the design mass
    return p
