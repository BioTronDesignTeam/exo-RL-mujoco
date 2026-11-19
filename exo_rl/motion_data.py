"""Turn continuous joint-angle recordings into reference cycles the human model can loop.

The human model (exo_rl/sim/human.py) loops one cycle per task (format: data/motions/README.md)
and differentiates it twice for its inertial feedforward, so a usable reference must be
    seamless   no jump in angle or velocity where the cycle wraps
    smooth     a second derivative free of measurement noise
    in range   inside the model's joint limits
    complete   with the root pitch ("trunk") the legs hang from
    performable by the human model (simulate_task); the planar model's box feet scuff on a
               plantarflexed toe-off and its ankle can't carry the body on the forefoot.
Recorded trials are many repeats of a movement and have no pelvis channel. prepare_cycle():
    1. finds repeats from maxima of a drive signal (left knee flexion for locomotion: one sharp
       peak per stride; mean knee flexion for squats),
    2. resamples every repeat over phase (locomotion: piecewise, so the right knee peak lines up
       too), drops repeats with atypical duration or shape and averages the rest; the period is
       the mean duration of the kept repeats (seconds),
    3. rotates the cycle so phase 0 matches the synthetic motions: left hip flexion peak (late
       swing, about heel strike) for locomotion, standing (lowest mean knee flexion) for squats,
    4. keeps each joint inside the model's range with a smooth soft limit,
    5. low-passes with a truncated Fourier series, which is periodic by construction,
    6. adds the trunk: for squats the pitch that keeps both feet flat, ankle + hip - knee (the
       identity the synthetic squat uses); for locomotion a constant, 0 by default,
    7. with foot_flat, replaces the ankles with the angle that keeps each foot parallel to the
       floor, as the synthetic motions do; hip and knee (the exo's joints) stay as recorded.
       Planar box feet need it for every task; on MyoLeg, recorded walking ankles fail but
       squat, high knees and backward walk work as recorded (scripts/prepare_motions.py tries
       the recorded ankles first).
info["amplitude_retained"] flags trials whose repeats are too irregular to average, and
simulate_task() checks that the human model can actually perform the result.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .sim.model_builder import HUMAN_JOINTS

KINDS = ("locomotion", "squat")
LEG = {"hip": 0, "knee": 1, "ankle": 2}


def read_angles_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(time s, angles deg (n, 6) in HUMAN_JOINTS order) from a CSV with those column names."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ["time", *HUMAN_JOINTS] if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    t = np.array([float(r["time"]) for r in rows])
    q = np.array([[float(r[j]) for j in HUMAN_JOINTS] for r in rows])
    if not (np.all(np.isfinite(q)) and np.all(np.diff(t) > 0)):
        raise ValueError(f"{path}: non-finite angles or non-increasing time")
    return t, q


def model_joint_ranges(human_model: str = "torque") -> dict:
    """{joint: (lo, hi) deg} for the six tracked joints and root_pitch (trunk), read from the human
    model the config selects (human.model: torque = planar, myo = MyoLeg). MyoLeg's root is a free
    joint, so the trunk keeps the planar model's root_pitch range as its sanity bound."""
    from .sim.model_builder import HumanExoModel, SubjectParams

    def rng(model, name):
        return tuple(float(v) for v in np.degrees(model.jnt_range[model.joint(name).id]))
    planar = HumanExoModel(SubjectParams(payload_mass=0.0)).model
    out = {"root_pitch": rng(planar, "root_pitch")}
    if human_model == "myo":
        from .sim.muscle_human import TRACKED
        from .sim.myo_model import load_myolegs
        myo = load_myolegs()
        out.update({j: rng(myo, TRACKED[j]) for j in HUMAN_JOINTS})
    elif human_model == "torque":
        out.update({j: rng(planar, j) for j in HUMAN_JOINTS})
    else:
        raise ValueError(f"unknown human model {human_model!r}")
    return out


# ---------------------------------------------------------------- repeat detection

def _gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    r = int(np.ceil(4 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    return np.convolve(np.pad(x, r, mode="reflect"), k / k.sum(), mode="valid")


def dominant_period(x: np.ndarray, fs: float, lo: float = 0.3, hi: float = 4.0) -> float:
    """Repeat period (s) of x from its autocorrelation; the shortest strong peak wins, so
    a two-cycle lag is not mistaken for the period."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    a, b = int(lo * fs), min(int(hi * fs), n - 2)
    peaks = [i for i in range(max(a, 1), b) if ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]]
    if not peaks:
        raise ValueError("no repeating pattern found")
    best = max(ac[i] for i in peaks)
    return min(i for i in peaks if ac[i] >= 0.8 * best) / fs


def find_repeats(t: np.ndarray, drive: np.ndarray) -> np.ndarray:
    """Times of the maxima of `drive` that start each repeat."""
    fs = 1.0 / np.median(np.diff(t))
    s = _gaussian_smooth(drive, 0.02 * fs)
    period = dominant_period(s, fs)
    p5, p95 = np.percentile(s, [5, 95])
    thresh = p5 + 0.5 * (p95 - p5)
    w = max(1, int(0.3 * period * fs))
    idx = [i for i in range(len(s)) if s[i] > thresh and s[i] == s[max(0, i - w):i + w + 1].max()]
    return t[np.array(idx, dtype=int)]


# ---------------------------------------------------------------- averaging

@dataclass
class Cycle:
    """One averaged, smoothed cycle. coefs: complex Fourier coefficients (K+1, 7) for the six
    joints then the trunk, in degrees over phase in [0, 1)."""
    period: float
    coefs: np.ndarray
    info: dict = field(default_factory=dict)

    def angles_deg(self, phase, deriv: int = 0) -> np.ndarray:
        """(len(phase), 7) degrees (deriv=1: deg per unit phase, etc.)."""
        return fourier_eval(self.coefs, np.atleast_1d(phase), deriv)


def fourier_coefs(samples: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Least-squares Fourier series through samples taken uniformly over phase [0, 1)."""
    return np.fft.rfft(samples, axis=0)[: n_harmonics + 1] / len(samples)


def fourier_eval(coefs: np.ndarray, phase: np.ndarray, deriv: int = 0) -> np.ndarray:
    k = np.arange(len(coefs))
    w = (2j * np.pi * k) ** deriv * np.where(k == 0, 1.0, 2.0)
    return np.real(np.exp(2j * np.pi * np.outer(phase, k)) @ (coefs * w[:, None]))


def soft_limit(x: np.ndarray, lo: float, hi: float, margin: float) -> np.ndarray:
    """Identity well inside [lo, hi]; within `margin` of a bound, bends smoothly (C1) toward it."""
    y = np.array(x, dtype=float)
    a, b = lo + margin, hi - margin
    low, high = y < a, y > b
    y[low] = lo + margin * np.exp((y[low] - a) / margin)
    y[high] = hi - margin * np.exp((b - y[high]) / margin)
    return y


def _resample(t, q, t0, t1, n, mid=None, mid_phase=None):
    """Repeat [t0, t1) sampled at n phases; with a landmark, time `mid` lands on `mid_phase`
    (piecewise-linear time warp)."""
    ph = np.arange(n) / n
    if mid is None:
        tt = t0 + (t1 - t0) * ph
    else:
        tt = np.where(ph < mid_phase, t0 + (mid - t0) * ph / mid_phase,
                      mid + (t1 - mid) * (ph - mid_phase) / (1.0 - mid_phase))
    return np.stack([np.interp(tt, t, q[:, j]) for j in range(q.shape[1])], axis=1)


def prepare_cycle(t: np.ndarray, q: np.ndarray, kind: str, *, trunk_deg: float = 0.0,
                  cutoff_hz: float = 8.0, min_harmonics: int = 6, n_phase: int = 400,
                  duration_tol: float = 0.25, shape_tol: float = 2.5,
                  ranges: dict | None = None, limit_margin: float = 3.0, limit_inset: float = 1.0,
                  foot_flat: bool = True) -> Cycle:
    """Average the repeats of a continuous trial into one loopable cycle (see module doc).

    t, q:         time (s) and joint angles (deg, HUMAN_JOINTS order, repository signs)
    kind:         "locomotion" or "squat"
    trunk_deg:    constant root pitch for locomotion (ignored for squats)
    cutoff_hz:    Fourier harmonics kept up to this frequency (at least min_harmonics)
    duration_tol: keep repeats within this fraction of the median duration
    shape_tol:    keep repeats whose RMS distance to the median repeat is within this many
                  times the median distance
    ranges:       {joint: (lo, hi) deg}; default model_joint_ranges(). Joints are soft-limited
                  to limit_inset deg inside each bound.
    foot_flat:    replace the recorded ankles with the angle that keeps each foot parallel to
                  the floor (hip and knee, the exo's joints, stay as recorded)
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ranges = ranges or model_joint_ranges()
    hl, kl, kr = HUMAN_JOINTS.index("hip_l"), HUMAN_JOINTS.index("knee_l"), HUMAN_JOINTS.index("knee_r")
    drive = q[:, kl] if kind == "locomotion" else 0.5 * (q[:, kl] + q[:, kr])
    events = find_repeats(t, drive)
    if len(events) < 3:
        raise ValueError(f"found {len(events)} repeat markers; need at least 3")

    starts, ends = events[:-1], events[1:]
    dur = ends - starts
    keep = np.abs(dur / np.median(dur) - 1.0) <= duration_tol
    mids, mid_phase = [None] * len(starts), None
    if kind == "locomotion":    # second landmark: right knee peak, so both legs' swings line up
        fs = 1.0 / np.median(np.diff(t))
        kr_s = _gaussian_smooth(q[:, kr], 0.02 * fs)
        mids = []
        for a, b in zip(starts, ends):
            m = (t > a) & (t < b)
            mids.append(t[m][np.argmax(kr_s[m])] if m.sum() > 2 else 0.5 * (a + b))
        rel = (np.array(mids) - starts) / dur
        mid_phase = float(np.median(rel[keep]))
        keep &= np.abs(rel - mid_phase) <= 0.15
    reps = np.stack([_resample(t, q, a, b, n_phase, mid, mid_phase) for a, b, mid in zip(starts, ends, mids)])
    dist = np.full(len(reps), np.inf)
    for _ in range(2):                          # median shape of the kept repeats, then re-test
        med = np.median(reps[keep], axis=0)
        dist = np.sqrt(np.mean((reps - med) ** 2, axis=(1, 2)))
        keep &= dist <= shape_tol * np.median(dist[keep])
    if keep.sum() < 2:
        raise ValueError("fewer than 2 consistent repeats")
    mean = reps[keep].mean(axis=0)
    period = float(dur[keep].mean())

    # Phase 0: standing for squats, left hip flexion peak (late swing) for locomotion.
    shift = int(np.argmin(0.5 * (mean[:, kl] + mean[:, kr]))) if kind == "squat" else int(np.argmax(mean[:, hl]))
    mean = np.roll(mean, -shift, axis=0)
    p2p = np.ptp(mean, axis=0) / np.maximum(np.ptp(reps[keep], axis=1).mean(axis=0), 1e-9)

    def limit(x, name):
        lo, hi = ranges[name]
        return soft_limit(x, lo + limit_inset, hi - limit_inset, limit_margin)

    k = max(min_harmonics, int(np.floor(cutoff_hz * period)))
    phase = np.arange(n_phase) / n_phase
    limited = np.stack([limit(mean[:, j], n) for j, n in enumerate(HUMAN_JOINTS)], axis=1)
    smooth = fourier_eval(fourier_coefs(limited, k), phase)

    legs = [{p: s + i for p, i in LEG.items()} for s in (0, 3)]
    if kind == "squat":
        trunk = np.mean([smooth[:, g["ankle"]] + smooth[:, g["hip"]] - smooth[:, g["knee"]] for g in legs], axis=0)
    else:
        trunk = np.full(n_phase, float(trunk_deg))
    model = smooth.copy()
    if foot_flat:
        for g, side in zip(legs, "lr"):
            model[:, g["ankle"]] = limit(trunk - smooth[:, g["hip"]] + smooth[:, g["knee"]], f"ankle_{side}")
    coefs = fourier_coefs(np.concatenate([model, trunk[:, None]], axis=1), k)

    out = fourier_eval(coefs, phase)
    names = [*HUMAN_JOINTS, "trunk"]
    lo = np.array([ranges[n][0] for n in [*HUMAN_JOINTS, "root_pitch"]])
    hi = np.array([ranges[n][1] for n in [*HUMAN_JOINTS, "root_pitch"]])
    clearance = np.minimum(out.min(axis=0) - lo, hi - out.max(axis=0))
    if np.any(clearance < 0):
        raise ValueError(f"reference leaves the model's joint range: {[n for n, c in zip(names, clearance) if c < 0]}")

    rnd = lambda v, d=2: round(float(v), d)  # noqa: E731
    info = {
        "kind": kind,
        "repeats_found": int(len(reps)),
        "repeats_used": int(keep.sum()),
        "period_s": rnd(period, 4),
        "period_sd_s": rnd(dur[keep].std(), 4),
        "harmonics": int(k),
        "phase0_shift": rnd(shift / n_phase, 4),
        "trunk": "feet flat: mean over legs of ankle + hip - knee" if kind == "squat"
                 else f"constant {trunk_deg:g} deg",
        "ankle": "feet parallel to the floor: trunk - hip + knee" if foot_flat else "recorded",
        "between_repeat_sd_deg": {n: rnd(v) for n, v in zip(HUMAN_JOINTS, reps[keep].std(axis=0).mean(axis=0))},
        # range of motion of the average / mean range of motion of the repeats; well below 1
        # means the repeats are out of step with each other and averaging flattened them
        "amplitude_retained": {n: rnd(v) for n, v in zip(HUMAN_JOINTS, p2p)},
        # how far the reference is from the (averaged) recording, per joint: RMS and max
        "change_from_recording_deg": {n: [rnd(np.sqrt(np.mean(d ** 2))), rnd(np.max(np.abs(d)))]
                                      for n, d in zip(HUMAN_JOINTS, (out[:, :6] - mean).T)},
        "range_deg": {n: [rnd(a, 1), rnd(b, 1)] for n, a, b in zip(names, out.min(axis=0), out.max(axis=0))},
        "min_clearance_to_joint_limit_deg": rnd(clearance.min()),
    }
    return Cycle(period, coefs, info)


def write_motion_csv(path: str, cycle: Cycle, rate_hz: float = 500.0) -> None:
    """data/motions format: one cycle, last row repeats the first (t = period)."""
    n = max(2, int(round(cycle.period * rate_hz)))
    phase = np.arange(n + 1) / n
    vals = cycle.angles_deg(phase)
    vals[-1] = vals[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", *HUMAN_JOINTS, "trunk"])
        for p, row in zip(phase, vals):
            w.writerow([f"{p * cycle.period:.6f}", *(f"{v:.5f}" for v in row)])


# ---------------------------------------------------------------- can the human model do it?

def simulate_task(cfg: dict, task: str, seeds, nominal: bool) -> dict:
    """Run the human model alone (exo off) on `task` for whole episodes, as the training env
    would (random start phase). cfg selects the motion source via tasks.data_dir."""
    from .envs.exo_env import ExoAssistEnv
    cfg = {**cfg, "randomization": {**cfg["randomization"], "enabled": not nominal}}
    env = ExoAssistEnv(cfg, seed=0)
    fails, reasons, track, resid, effort, v_ref, v = 0, [], [], [], [], [], []
    for s in seeds:
        env.reset(seed=int(s), options={"task": task, "exo_mode": "off", "nominal": nominal})
        x0 = env.hm.data.qpos[env.hm.root_qpos[0]]
        v_ref.append(env.speeds[task])
        err = []
        while True:
            _, _, term, trunc, info = env.step(np.zeros(4))
            err.append(np.degrees(np.sqrt(np.mean((info["q"] - info["q_ref"]) ** 2))))
            resid.append(info["residual_abs"])
            effort.append(info["reward_terms"]["effort"])
            if term or trunc:
                break
        if term:
            fails += 1
            reasons.append(info["terminated_reason"])
        track.append(float(np.mean(err)))
        v.append((env.hm.data.qpos[env.hm.root_qpos[0]] - x0) / env.t)
    r = np.mean(resid, axis=0)
    return {"episodes": len(track), "failed": fails, "fail_reasons": sorted(set(reasons)),
            "tracking_rms_deg": round(float(np.mean(track)), 2), "effort": round(float(np.mean(effort)), 4),
            "residual_x_N": round(float(r[0]), 1), "residual_pitch_Nm": round(float(r[2]), 1),
            "implied_speed_m_s": round(float(np.mean(v_ref)), 2), "speed_m_s": round(float(np.mean(v)), 2)}
