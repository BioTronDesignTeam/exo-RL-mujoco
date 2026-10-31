"""MyoLeg musculoskeletal model (optional dependency: pip install -r requirements-myo.txt)."""
import mujoco
import numpy as np
import pytest

pytest.importorskip("myo_sim")
from exo_rl.sim.myo_model import load_myolegs, summary  # noqa: E402


def test_myolegs_loads_with_exo_joints():
    s = summary(load_myolegs())
    assert len(s["muscles"]) == 80
    for joint in ("hip_flexion_l", "knee_angle_l", "hip_flexion_r", "knee_angle_r"):
        assert joint in s["joints"]                 # the exo's four joints exist in the muscle model


def test_myolegs_simulates_stably():
    m = load_myolegs()
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.2                                  # light, constant muscle activation
    for _ in range(500):                             # 0.5 s
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.act))
