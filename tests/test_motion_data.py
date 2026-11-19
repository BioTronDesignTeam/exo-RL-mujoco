"""Recorded motions: cycle preparation (exo_rl/motion_data.py) and the prepared set the human uses."""
import json
import os

import numpy as np
import pytest

from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.motion_data import model_joint_ranges, prepare_cycle, soft_limit, write_motion_csv
from exo_rl.sim.reference import SYNTHETIC, ReferenceTrajectory, TabulatedMotion, load_motions

PREPARED = "data/motions/gatech_ab01"


def _recording(motion, seconds=20.0, fs=200.0, jitter=0.03, noise=0.5, seed=0):
    """A continuous 'recording' of a synthetic motion: repeats of varying duration, plus noise."""
    rng = np.random.default_rng(seed)
    ts, qs, t0 = [], [], 0.0
    while t0 < seconds:
        dur = motion.period * (1 + rng.uniform(-jitter, jitter))
        ph = np.arange(int(round(dur * fs))) / round(dur * fs)
        ts.append(t0 + ph * dur)
        qs.append([np.degrees(motion.angles(p * motion.period)[0]) for p in ph])
        t0 += dur
    q = np.concatenate(qs)
    return np.concatenate(ts), q + rng.normal(0, noise, q.shape)


def _best_shift_rms(a, b):
    return min(np.sqrt(np.mean((np.roll(a, k, axis=0) - b) ** 2)) for k in range(len(a)))


def test_cycle_recovers_period_and_shape():
    src = SYNTHETIC["walk"]
    t, q = _recording(src)
    cyc = prepare_cycle(t, q, "locomotion", foot_flat=False, ranges=model_joint_ranges())
    assert abs(cyc.period / src.period - 1) < 0.01
    assert min(cyc.info["amplitude_retained"].values()) > 0.95
    ph = np.arange(200) / 200
    ref = np.array([np.degrees(src.angles(p * src.period)[0]) for p in ph])
    assert _best_shift_rms(cyc.angles_deg(ph)[:, :6], ref) < 2.0


def test_cycle_loops_without_a_jump(tmp_path):
    t, q = _recording(SYNTHETIC["walk"])
    cyc = prepare_cycle(t, q, "locomotion")
    path = tmp_path / "walk.csv"
    write_motion_csv(str(path), cyc)
    ref = ReferenceTrajectory([(TabulatedMotion(str(path), "walk"), 1e9)])
    qdd = np.array([ref.sample(x)[2] for x in np.linspace(0, cyc.period, 400)])
    near_wrap = np.array([ref.sample(cyc.period + x)[2] for x in np.linspace(-0.01, 0.01, 21)])
    assert np.max(np.abs(near_wrap)) < 1.2 * np.max(np.abs(qdd))


def test_squat_trunk_keeps_feet_flat():
    """The synthetic squat obeys ankle = trunk - hip + knee, so its trunk must be recovered."""
    src = SYNTHETIC["squat"]
    t, q = _recording(src, noise=0.2)
    cyc = prepare_cycle(t, q, "squat")
    ph = np.arange(200) / 200
    trunk = cyc.angles_deg(ph)[:, 6]
    ref = np.array([np.degrees(src.angles(p * src.period)[1]) for p in ph])
    assert _best_shift_rms(trunk, ref) < 2.5     # 0..45 deg; the 8 Hz fit rounds the clipped bottom hold
    assert max(rms for rms, _ in cyc.info["change_from_recording_deg"].values()) < 1.0


def test_soft_limit():
    x = np.linspace(-20, 20, 401)
    y = soft_limit(x, -10, 10, 3)
    assert np.all((y > -10) & (y < 10))
    assert np.all(np.diff(y) > 0)
    inner = np.abs(x) <= 7
    assert np.allclose(y[inner], x[inner])


def _manifest():
    path = os.path.join(PREPARED, "manifest.json")
    if not os.path.exists(path):
        return {"tasks": {}, "checks": {}}
    with open(path) as f:
        return json.load(f)


def _prepared_tasks():
    return [t for t in _manifest()["tasks"] if os.path.exists(os.path.join(PREPARED, f"{t}.csv"))]


def _use_prepared(cfg):
    """cfg on the prepared motions and the human model they were checked on."""
    cfg["tasks"]["data_dir"] = PREPARED
    cfg["human"]["model"] = _manifest()["checks"].get("human_model", "torque")
    cfg["randomization"]["enabled"] = False


@pytest.mark.parametrize("task", _prepared_tasks())
def test_human_tracks_prepared_motion(cfg, task):
    _use_prepared(cfg)
    env = ExoAssistEnv(cfg, seed=0)
    assert isinstance(env.motions[task], TabulatedMotion)
    env.reset(options={"task": task, "exo_mode": "off"})
    errs = []
    for _ in range(400):
        *_, term, _, info = env.step(np.zeros(4))
        assert not term, f"{task} terminated: {info['terminated_reason']}"
        errs.append(np.degrees(np.sqrt(np.mean((info["q"] - info["q_ref"]) ** 2))))
    assert np.mean(errs[100:]) < 8.0


def test_unprepared_tasks_fall_back_to_synthetic(cfg):
    names = cfg["tasks"]["names"]
    m = load_motions(names, PREPARED)
    for n in names:
        assert isinstance(m[n], TabulatedMotion) == (n in _prepared_tasks())


@pytest.mark.parametrize("human", ["torque", "myo"])
def test_fall_line_unchanged_for_synthetic_motions(cfg, human):
    cfg["human"]["model"] = human
    cfg["randomization"]["enabled"] = False
    env = ExoAssistEnv(cfg, seed=0)
    for task in SYNTHETIC:
        env.reset(options={"task": task})
        assert env._fall_height() == pytest.approx(0.5 * env.hm.segments.leg_len)


@pytest.mark.skipif("squat" not in _prepared_tasks(), reason="no prepared squat")
def test_deep_squat_started_mid_descent_keeps_feet_down(cfg):
    """Regression: a start mid-descent used to leave the pelvis at rest while the legs folded,
    so the feet left the floor and the human dropped, then collapsed on landing."""
    _use_prepared(cfg)
    env = ExoAssistEnv(cfg, seed=0)
    env.reset(options={"task": "squat", "exo_mode": "off"})
    m = env.motions["squat"]
    env.ref = ReferenceTrajectory([(m, 1e9)], t_offset=0.26 * m.period)
    env.start_pose()
    env.human.reset()
    airborne = 0
    for i in range(60):
        *_, term, _, info = env.step(np.zeros(4))
        assert not term, info["terminated_reason"]
        airborne += i < 20 and env.hm.data.ncon == 0
    assert airborne <= 2, f"feet off the floor for {airborne} of the first 20 steps"   # old start: 19-39
