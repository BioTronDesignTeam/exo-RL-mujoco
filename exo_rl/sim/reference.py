"""Reference motions for the simulated human to perform.

The synthetic motions below are PLACEHOLDERS so the pipeline runs end-to-end before
real data exists. Replace them with recorded data (TODO 6) by dropping CSV files into
data/motions/ (format documented in TabulatedMotion) and setting tasks.data_dir.

Conventions (degrees in CSV / generators, radians everywhere else):
    hip   + = flexion        knee  + = flexion
    ankle + = dorsiflexion   trunk + = forward lean
Joint order: hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r
"""
from __future__ import annotations

import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .model_builder import HUMAN_JOINTS


def resolve_motion_path(path: str | Path) -> Path:
    """Resolve historical CSV paths without changing the data used by old runs."""
    target = Path(path)
    if target.is_file():
        return target
    root = Path(__file__).resolve().parents[2] / "data" / "motions"
    try:
        relative = target.resolve().relative_to(root).as_posix()
    except ValueError:
        return target
    manifest = root / "archive" / "manifest.json"
    if not manifest.is_file():
        return target
    archived = json.loads(manifest.read_text(encoding="utf-8-sig"))["legacy_paths"].get(relative)
    return root / archived if archived is not None else target


def _pg(phi, mu, sigma):
    """Periodic gaussian bump on the unit circle."""
    d = (phi - mu + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (d / sigma) ** 2)


# --- Synthetic single-leg patterns: phi in [0, 1) -> (hip, knee, ankle) in degrees -----------

_LT, _LS = 0.245, 0.246           # Winter thigh/shank length ratios (height cancels out)


def _reach(theta_t, knee):
    """Vertical hip->ankle distance (units of height) for absolute thigh angle and knee flexion."""
    return _LT * np.cos(theta_t) + _LS * np.cos(theta_t - knee)


def _knee_for_reach(theta_t, h):
    """Smallest non-negative knee flexion giving vertical reach h (clipped if unreachable)."""
    c = np.clip((h - _LT * np.cos(theta_t)) / _LS, -1.0, 1.0)
    a = np.arccos(c)
    k1, k2 = theta_t - a, theta_t + a
    return np.where(k1 >= 0, k1, k2)


def _circ_smooth(x, sigma):
    n = len(x)
    k = np.arange(n)
    k = np.minimum(k, n - k)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(w / w.sum())))


def make_gait_table(hip_mean=10.0, hip_amp=22.0, trunk_deg=3.0, stance=0.62,
                    swing_knee=60.0, clearance=0.02, n=400, hip_waypoints=None, stance_knee=5.0):
    """Kinematically consistent in-place gait for the low-friction (treadmill) floor.

    Hip follows a cosine; the pelvis height is set by the stance leg(s) with a nearly
    straight knee; the other stance leg bends to meet it; the swing leg bends enough
    to clear the floor by `clearance` (fraction of body height). Ankles keep the foot
    flat. Returns an (n, 6) array in degrees: hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r.
    """
    phi = np.arange(n) / n
    if hip_waypoints is None:
        hip_l = hip_mean + hip_amp * np.cos(2 * np.pi * (phi - 0.02))
    else:   # [(phase, deg), ...] -> periodic piecewise-linear, then smoothed
        wp = sorted(hip_waypoints)
        xs = np.array([p for p, _ in wp] + [wp[0][0] + 1.0])
        ys = np.array([v for _, v in wp] + [wp[0][1]])
        hip_l = _circ_smooth(np.interp(phi, xs, ys, period=None) if xs[0] == 0 else
                             np.interp(phi, xs, ys), sigma=n * 0.025)
    hip_r = np.roll(hip_l, n // 2)
    th_l, th_r = np.radians(hip_l - trunk_deg), np.radians(hip_r - trunk_deg)
    st_l = phi < stance
    st_r = np.roll(st_l, n // 2)
    straight = np.radians(stance_knee)
    reach_l = np.where(st_l, _reach(th_l, straight), np.inf)
    reach_r = np.where(st_r, _reach(th_r, straight), np.inf)
    h = np.minimum(reach_l, reach_r)
    if np.any(~np.isfinite(h)):          # flight phases (running): interpolate across them
        ok = np.isfinite(h)
        h = np.interp(phi, phi[ok], h[ok], period=1.0)
    h = _circ_smooth(h, sigma=n * 0.015)

    def leg(theta_t, st, phase_shift):
        knee_st = _knee_for_reach(theta_t, h)
        ph = (phi - phase_shift) % 1.0
        sw_phase = np.clip((ph - stance) / (1 - stance), 0, 1)
        bump = np.radians(swing_knee) * np.sin(np.pi * sw_phase) ** 1.5
        knee_clear = _knee_for_reach(theta_t, h - clearance)
        knee = np.where(st, knee_st, np.maximum(bump, knee_clear))
        return np.degrees(_circ_smooth(knee, sigma=n * 0.01))

    knee_l = leg(th_l, st_l, 0.0)
    knee_r = leg(th_r, st_r, 0.5)
    ankle_l = np.clip(trunk_deg - hip_l + knee_l, -40, 40)
    ankle_r = np.clip(trunk_deg - hip_r + knee_r, -40, 40)
    return np.stack([hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r], axis=1)


def _high_knees(phi):
    hip = 5.0 + 85.0 * _pg(phi, 0.25, 0.11)
    knee = 8.0 + 97.0 * _pg(phi, 0.27, 0.12)
    ankle = -12.0 * _pg(phi, 0.27, 0.12)
    return hip, knee, ankle


def _quick_steps(phi):
    hip = 5.0 + 45.0 * _pg(phi, 0.25, 0.10)
    knee = 8.0 + 65.0 * _pg(phi, 0.26, 0.11)
    ankle = 5.0 * _pg(phi, 0.26, 0.11)
    return hip, knee, ankle


def _squat_depth(phi):
    # Down-up with a short hold at the bottom; smoothstep keeps velocity continuous.
    s = np.clip(1.15 * 0.5 * (1.0 - np.cos(2 * np.pi * phi)), 0.0, 1.0)
    return s * s * (3 - 2 * s)


@dataclass
class PeriodicMotion:
    name: str
    period: float
    leg_fn: Callable | None = None      # phi -> (hip, knee, ankle) deg
    trunk_deg: float = 0.0
    right_offset: float = 0.5
    squat: bool = False                 # squat: both legs together, flat-foot ankle

    def angles(self, t: float):
        phi = (t / self.period) % 1.0
        if self.squat:
            s = _squat_depth(phi)
            hip, knee, trunk = 105.0 * s, 100.0 * s, 45.0 * s
            ankle = trunk - hip + knee      # keeps the foot flat on the floor
            q = [hip, knee, ankle] * 2
        else:
            left = self.leg_fn(phi)
            right = self.leg_fn((phi + self.right_offset) % 1.0)
            q = [*left, *right]
            trunk = self.trunk_deg
        return np.radians(np.array(q, dtype=float)), float(np.radians(trunk))

    def phase(self, t: float) -> float:
        return (t / self.period) % 1.0


class TableMotion:
    """Periodic motion from an (n, 6) degree table sampled uniformly over one cycle."""

    def __init__(self, name, period, table_deg, trunk_deg=0.0, reverse=False, residual_overrides=None):
        self.name, self.period = name, period
        self.residual_overrides = residual_overrides or {}   # per-motion human.residual gains
        tbl = table_deg[::-1] if reverse else table_deg
        self.table = np.radians(np.vstack([tbl, tbl[:1]]))
        self.grid = np.linspace(0.0, 1.0, len(self.table))
        self.trunk = float(np.radians(trunk_deg))

    def angles(self, t):
        phi = (t / self.period) % 1.0
        q = np.array([np.interp(phi, self.grid, self.table[:, i]) for i in range(self.table.shape[1])])
        return q, self.trunk

    def phase(self, t):
        return (t / self.period) % 1.0


class TabulatedMotion:
    """Periodic motion loaded from a CSV of one (or more) full cycles.

    Required columns: time (s), hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r (deg)
    Optional column:  trunk (deg, forward lean +)
    The time span of the file is treated as one period and looped.
    """

    def __init__(self, path: str | Path, name: str | None = None):
        with open(resolve_motion_path(path), newline="") as f:
            rows = list(csv.DictReader(f))
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.t = np.array([float(r["time"]) for r in rows])
        self.t -= self.t[0]
        self.period = float(self.t[-1])
        self.q = np.radians(np.array([[float(r[j]) for j in HUMAN_JOINTS] for r in rows]))
        self.trunk = np.radians(np.array([float(r.get("trunk", 0.0) or 0.0) for r in rows]))

    def angles(self, t: float):
        tt = t % self.period
        q = np.array([np.interp(tt, self.t, self.q[:, i]) for i in range(len(HUMAN_JOINTS))])
        return q, float(np.interp(tt, self.t, self.trunk))

    def phase(self, t: float) -> float:
        return (t % self.period) / self.period


_WALK = make_gait_table(trunk_deg=3.0)
_FAST_WALK = make_gait_table(hip_amp=24.0, trunk_deg=5.0, stance=0.60)
_BACK_WALK = make_gait_table(hip_mean=5.0, hip_amp=18.0, trunk_deg=0.0)
# Extra test motions (not in the default training task list):
# Grounded running: bent knees, fast cadence, one foot always down (no flight phase; this
# simplified human can't generate the push-off for true flight).
_JOG = make_gait_table(trunk_deg=8.0, stance=0.55, swing_knee=85.0, stance_knee=30.0, clearance=0.03,
                       hip_waypoints=[(0.0, 22), (0.55, -10), (0.62, -10), (0.88, 28)])
_HIGH_KNEES_TRAVEL = make_gait_table(trunk_deg=5.0, stance=0.55, swing_knee=105.0, clearance=0.03,
                                     hip_waypoints=[(0.0, 15), (0.55, -10), (0.78, 85)])

TRAVELING_MOTIONS = frozenset({"walk", "fast_walk", "backward_walk", "jog", "high_knees_travel"})

SYNTHETIC = {
    "walk": TableMotion("walk", 1.10, _WALK, trunk_deg=3.0),
    "fast_walk": TableMotion("fast_walk", 0.85, _FAST_WALK, trunk_deg=5.0),
    "backward_walk": TableMotion("backward_walk", 1.30, _BACK_WALK, trunk_deg=0.0, reverse=True),
    # Grounded running has large trunk pitching moments; a firmer trunk helper keeps it stable.
    "jog": TableMotion("jog", 0.75, _JOG, trunk_deg=8.0, residual_overrides={"pitch_kp": 4000, "pitch_kd": 400}),
    "high_knees_travel": TableMotion("high_knees_travel", 0.90, _HIGH_KNEES_TRAVEL, trunk_deg=5.0),
    "squat": PeriodicMotion("squat", 2.50, squat=True),
    "high_knees": PeriodicMotion("high_knees", 0.70, _high_knees, trunk_deg=5.0),
    "quick_steps": PeriodicMotion("quick_steps", 0.45, _quick_steps, trunk_deg=3.0),
}


def load_motions(names, data_dir: str | None = None, periods: dict | None = None) -> dict:
    """Load requested motions, preferring CSVs in data_dir over synthetic placeholders."""
    motions = {}
    for n in names:
        path = resolve_motion_path(os.path.join(data_dir, f"{n}.csv")) if data_dir else None
        if path and os.path.exists(path):
            motions[n] = TabulatedMotion(path, n)
        elif n in SYNTHETIC:
            motions[n] = SYNTHETIC[n]
        else:
            raise ValueError(f"No motion '{n}' in {data_dir} and no synthetic fallback.")
        if periods and n in periods:
            period = float(periods[n])
            if not np.isfinite(period) or period <= 0:
                raise ValueError(f"Motion period for {n!r} must be finite and positive")
            motion = copy.deepcopy(motions[n])
            if isinstance(motion, TabulatedMotion):
                motion.t *= period / motion.period
            motion.period = period
            motions[n] = motion
    return motions


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class ReferenceTrajectory:
    """A time-indexed reference: one task, or a circuit of tasks with blended transitions."""

    def __init__(self, segments, blend: float = 0.5, t_offset: float = 0.0, startup_seconds: float = 0.0):
        # segments: list of (motion, duration)
        self.segments = segments
        self.blend = blend
        self.t_offset = t_offset
        self.startup_seconds = float(startup_seconds)
        if not np.isfinite(self.startup_seconds) or self.startup_seconds < 0:
            raise ValueError("Reference startup_seconds must be finite and nonnegative")
        if self.startup_seconds and len(segments) != 1:
            raise ValueError("Reference startup currently supports a single motion")
        self.starts = np.cumsum([0.0] + [d for _, d in segments[:-1]])

    @property
    def name(self) -> str:
        return "+".join(m.name for m, _ in self.segments)

    def clock(self, t):
        """Integrate a smooth zero-to-full cadence ramp without a pose jump."""
        duration = self.startup_seconds
        if not duration:
            return t
        if t <= 0:
            return 0.0
        if t >= duration:
            return t - duration / 2
        u = t / duration
        return duration * (u ** 3 - .5 * u ** 4)

    def clock_rate(self, t):
        if not self.startup_seconds:
            return 1.0
        u = float(np.clip(t / self.startup_seconds, 0., 1.))
        return u * u * (3 - 2 * u)

    def weights(self, t):
        if len(self.segments) == 1:
            return np.array([1.0])
        w = np.zeros(len(self.segments))
        for i in range(len(self.segments)):
            start = self.starts[i]
            end = start + self.segments[i][1]
            rise = 1.0 if i == 0 else _smoothstep((t - start + self.blend / 2) / self.blend)
            fall = 1.0 if i == len(self.segments) - 1 else 1.0 - _smoothstep((t - end + self.blend / 2) / self.blend)
            w[i] = rise * fall
        return w / max(w.sum(), 1e-9)

    def angles(self, t):
        t = self.clock(t) + self.t_offset
        w = self.weights(t)
        q = np.zeros(len(HUMAN_JOINTS))
        trunk = 0.0
        for wi, (m, _), s in zip(w, self.segments, self.starts):
            if wi > 0:
                qi, ti = m.angles(t - s if len(self.segments) > 1 else t)
                q += wi * qi
                trunk += wi * ti
        return q, trunk

    def sample(self, t, dt=2e-3):
        """Returns q, qd, qdd, trunk, trunk_d (central differences)."""
        q0, tr0 = self.angles(t - dt)
        q1, tr1 = self.angles(t)
        q2, tr2 = self.angles(t + dt)
        qd = (q2 - q0) / (2 * dt)
        qdd = (q2 - 2 * q1 + q0) / (dt * dt)
        return q1, qd, qdd, tr1, (tr2 - tr0) / (2 * dt)

    def active(self, t):
        """(task name, phase in [0,1)) of the dominant segment, for logging/plots only."""
        t = self.clock(t) + self.t_offset
        i = int(np.argmax(self.weights(t)))
        m = self.segments[i][0]
        local = t - self.starts[i] if len(self.segments) > 1 else t
        return m.name, m.phase(local)
