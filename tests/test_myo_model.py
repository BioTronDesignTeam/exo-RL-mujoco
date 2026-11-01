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
    exo = ExoHardware("configs/exo_mk1_fa900.yaml")
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
        errs.append(hc.q_target - d.qpos[hc.qadr[:6]])
        effort.append(hc.effort(a))
    return np.degrees(np.sqrt(np.mean(np.square(errs)))), np.array(effort), hc


def test_muscles_hold_quiet_standing():
    rms, effort, hc = _track("squat", 0.2)          # the squat starts from standing
    assert rms < 1.0 and effort.max() < 0.02 and np.all((hc.a >= 0) & (hc.a <= 1))


def test_muscles_track_a_squat():
    rms, effort, hc = _track("squat", 2.5)           # one full squat
    assert rms < 3.0 and effort.mean() > 0.02        # tracked, and the muscles did the work
    assert np.all(np.abs(np.degrees(hc.lean)) < 15)  # balance shift stays small


def test_balance_shift_decays_while_moving():
    import yaml
    from exo_rl.sim.muscle_human import MuscleHumanController
    cfg = yaml.safe_load(open("configs/default.yaml"))["human"]
    m = load_myolegs(SubjectParams(payload_mass=0.0))
    d = mujoco.MjData(m)
    hc = MuscleHumanController(m, cfg, 75.0)
    hc.lean[:] = [0.1, 0.05]
    for _ in range(500):                             # 1 s of walking reference (balance_tau_s = 1)
        hc.update_balance(d, m.opt.timestep, v_ref=1.2)
    assert np.all(np.abs(hc.lean) < 0.4 * np.array([0.1, 0.05]))   # in gait the shift would drift


def _myo_env():
    import yaml
    from exo_rl.envs.exo_env import ExoAssistEnv
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["human"]["model"] = "myo"
    return ExoAssistEnv(cfg, seed=0), cfg


def test_env_runs_on_myolegs_with_same_spaces():
    env, cfg = _myo_env()
    import yaml
    from exo_rl.envs.exo_env import ExoAssistEnv
    planar = ExoAssistEnv(yaml.safe_load(open("configs/default.yaml")), seed=0)
    assert env.observation_space.shape == planar.observation_space.shape     # policies are interchangeable
    obs, _ = env.reset(seed=0, options={"nominal": True, "task": "squat"})
    for _ in range(30):
        obs, r, term, trunc, info = env.step(np.zeros(4))
    assert np.all(np.isfinite(obs)) and not term
    assert 0 < info["reward_terms"]["effort"] < 1                         # mean muscle activation^2
    assert info["tau_needed"].shape == (4,)


def test_env_exo_torque_reaches_myolegs_joints():
    env, _ = _myo_env()
    env.reset(seed=0, options={"nominal": True, "task": "squat"})
    env.step(np.array([1.0, 0.0, 0.0, 0.0]))                               # full hip_l flexion command
    force = env.hm.data.actuator_force[env.exo_act]
    assert force[0] > 1.0 and np.allclose(force[1:], 0.0)                  # hip_l motor pushes (rate-limited)


def test_myo_env_rejects_partial_effort_joints():
    import yaml
    from exo_rl.envs.exo_env import ExoAssistEnv
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["human"]["model"] = "myo"
    cfg["reward"]["effort_joints"] = ["hip", "knee"]
    with pytest.raises(ValueError):
        ExoAssistEnv(cfg, seed=0)


def test_effort_metrics_and_muscle_groups():
    import yaml
    from exo_rl.sim.muscle_human import MUSCLE_GROUPS, MuscleHumanController
    cfg = yaml.safe_load(open("configs/default.yaml"))["human"]
    m = load_myolegs(SubjectParams(payload_mass=0.0))
    mean = MuscleHumanController(m, cfg, 75.0)
    weighted = MuscleHumanController(m, {**cfg, "muscle": {"effort": "strength_weighted"}}, 75.0)
    a = np.zeros(len(mean.muscles))
    a[mean.groups["quads"]] = 1.0                    # only the (big) quads active
    assert np.isclose(mean.effort(a), len(mean.groups["quads"]) / len(a))
    assert weighted.effort(a) > mean.effort(a)       # strength weighting counts big muscles more
    g = mean.group_activation(a)
    assert g[list(MUSCLE_GROUPS).index("quads")] == 1.0 and g.sum() == 1.0
    with pytest.raises(ValueError):
        MuscleHumanController(m, {**cfg, "muscle": {"effort": "nope"}}, 75.0)


def test_env_reports_muscle_groups():
    from exo_rl.sim.muscle_human import MUSCLE_GROUPS
    env, _ = _myo_env()
    env.reset(seed=0, options={"nominal": True, "task": "squat"})
    _, _, _, _, info = env.step(np.zeros(4))
    assert list(info["muscle_groups"]) == list(MUSCLE_GROUPS)
    assert all(0 <= v <= 1 for v in info["muscle_groups"].values())
