"""MyoLeg musculoskeletal model (optional dependency: pip install -r requirements-myo.txt)."""
import mujoco
import numpy as np
import pytest

pytest.importorskip("myo_sim")
from exo_rl.sim.model_builder import SubjectParams  # noqa: E402
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


def test_upper_body_brings_subject_to_full_mass():
    m = load_myolegs(SubjectParams(mass=75.0, payload_mass=0.0))
    assert np.isclose(m.body_mass.sum(), 75.0, atol=1e-3)
    bare = load_myolegs()
    com = []
    for model in (bare, m):
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        com.append(d.subtree_com[0][2])
    assert com[1] > com[0] + 0.1                    # the trunk raises the whole-body COM


def test_exo_attaches_to_myolegs():
    from exo_rl.sim.exo_hardware import ExoHardware
    from exo_rl.sim.myo_model import exo_actuator_ids
    exo = ExoHardware("configs/exo_mk1.yaml")
    p = SubjectParams(mass=80.0, payload_mass=11.34, exo=exo)
    worn = load_myolegs(p)
    assert np.isclose(worn.body_mass.sum(), 80.0 + 11.34 + exo.total_mass, atol=1e-3)
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


def _track(motion: str, seconds: float):
    import yaml
    from exo_rl.sim.muscle_human import MuscleHumanController, stand_on_floor
    from exo_rl.sim.reference import ReferenceTrajectory, load_motions
    cfg = yaml.safe_load(open("configs/default.yaml"))
    m = load_myolegs(SubjectParams(payload_mass=0.0))
    d = mujoco.MjData(m)
    ref = ReferenceTrajectory([(load_motions([motion])[motion], 1e9)])
    q, _, _, trunk, _ = ref.sample(0.0)
    stand_on_floor(m, d, q, trunk)
    hc = MuscleHumanController(m, cfg["human"], 75.0)
    errs, effort = [], []
    for k in range(int(seconds / m.opt.timestep)):
        q, qd, qdd, trunk, trunk_d = ref.sample(k * m.opt.timestep)
        a, _ = hc.compute(d, q, qd, qdd, trunk, trunk_d, 0.0, np.zeros(4), m.opt.timestep)
        mujoco.mj_step(m, d)
        errs.append(np.clip(q, hc.q_lo, hc.q_hi) - d.qpos[hc.qadr[:6]])
        effort.append(hc.effort(a))
    return np.degrees(np.sqrt(np.mean(np.square(errs)))), np.array(effort), hc


def test_muscles_hold_quiet_standing():
    rms, effort, hc = _track("squat", 0.2)          # the squat starts from standing
    assert rms < 0.5 and effort.max() < 0.02 and np.all((hc.a >= 0) & (hc.a <= 1))


def test_muscles_track_a_squat():
    rms, effort, _ = _track("squat", 2.5)            # one full squat
    assert rms < 3.0 and effort.mean() > 0.02        # tracked, and the muscles did the work
