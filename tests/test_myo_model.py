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


def test_exo_attaches_to_myolegs():
    from exo_rl.sim.exo_hardware import ExoHardware
    from exo_rl.sim.myo_model import exo_actuator_ids
    exo = ExoHardware("configs/exo_mk1.yaml")
    bare, worn = load_myolegs(), load_myolegs(exo)
    assert np.isclose(worn.body_mass.sum() - bare.body_mass.sum(), exo.total_mass, atol=1e-3)
    d = mujoco.MjData(worn)
    d.ctrl[exo_actuator_ids(worn)] = [10.0, 20.0, 30.0, 40.0]
    mujoco.mj_forward(worn, d)
    with_exo = [d.qfrc_actuator[worn.joint(j).dofadr[0]] for j in ("hip_flexion_l", "knee_angle_l",
                                                                   "hip_flexion_r", "knee_angle_r")]
    d.ctrl[:] = 0.0
    mujoco.mj_forward(worn, d)
    passive = [d.qfrc_actuator[worn.joint(j).dofadr[0]] for j in ("hip_flexion_l", "knee_angle_l",
                                                                  "hip_flexion_r", "knee_angle_r")]
    assert np.allclose(np.subtract(with_exo, passive), [10.0, 20.0, 30.0, 40.0])
    for _ in range(500):
        mujoco.mj_step(worn, d)
    assert np.all(np.isfinite(d.qpos))
