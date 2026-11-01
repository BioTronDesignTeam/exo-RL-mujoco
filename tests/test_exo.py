"""CAD exoskeleton integration: mass bookkeeping, motor signs, stability, burden."""
import mujoco
import numpy as np
import pytest

from deploy.joint_map import MotorMap
from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.sim.exo_hardware import ExoHardware
from exo_rl.sim.model_builder import HumanExoModel, SubjectParams


@pytest.fixture(scope="module", params=["configs/exo_mk1_fa900.yaml", "configs/exo_mk1.yaml"])
def exo(request):
    return ExoHardware(request.param)        # the current exo and the previous MK1 (older runs use it)


def test_mass_is_conserved(exo):
    assert abs(exo.total_mass - exo.model.body_mass.sum()) < 1e-6
    with_exo = HumanExoModel(SubjectParams(exo=exo)).model.body_mass.sum()
    without = HumanExoModel(SubjectParams()).model.body_mass.sum()
    assert abs(with_exo - without - exo.total_mass) < 1e-4


@pytest.mark.parametrize("joint", ["hip_l", "knee_l", "hip_r", "knee_r"])
def test_motor_sign_makes_flexion(exo, joint):
    """Command +human flexion through the MotorMap on the standalone CAD model (no gravity)
    and check the limb moves the right way: hip flexion -> thigh swings forward;
    knee flexion -> shank rotates backward relative to the thigh (its own center of mass can sit
    on the knee axis, and the free thigh swings the other way)."""
    m = exo.model
    d = mujoco.MjData(m)
    d.qpos[:] = exo.data.qpos                # the worn pose (cad_pose_deg)
    m.opt.gravity[:] = 0
    mm = exo.motor_map()
    i = mm["order"].index(joint)
    motor = MotorMap(mm["signs"]).tau_to_motor(np.eye(4)[i] * 5.0)[i]
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mm["urdf_joints"][i])
    legs = exo.spec["legs"][joint[-1]]
    thigh, shank = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, legs[k][0]) for k in ("thigh_links", "shank_links"))

    def orientation(b, ref):
        """Orientation of body b in world (ref None) or relative to body ref."""
        Rb = d.xmat[b].reshape(3, 3).copy()
        return Rb if ref is None else d.xmat[ref].reshape(3, 3).T @ Rb

    def pitch_back(R0, R1, ref):
        """Rotation from R0 to R1 about the human left axis, rad (+ = lower end backward)."""
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, _mat2quat(R1 @ R0.T), 1.0)        # axis in world, or in ref's frame
        if ref is not None:
            v = d.xmat[ref].reshape(3, 3) @ v
        return float((exo.R @ v)[1])

    body, ref = (thigh, None) if joint.startswith("hip") else (shank, thigh)
    mujoco.mj_forward(m, d)
    R0 = orientation(body, ref)
    for _ in range(50):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[jid]] = motor
        mujoco.mj_step(m, d)
    turned = pitch_back(R0, orientation(body, ref), ref)
    if joint.startswith("hip"):
        assert turned < -1e-3                # thigh's lower end swings forward
    else:
        assert turned > 1e-3                 # shank's lower end swings backward
    m.opt.gravity[:] = [0, 0, -9.81]


def _mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def test_env_with_exo_is_stable_and_costs_effort(cfg):
    cfg["randomization"]["enabled"] = False
    effort = {}
    for enabled in (False, True):
        cfg["exo_hardware"]["enabled"] = enabled
        env = ExoAssistEnv(cfg, seed=0)
        env.reset(seed=1, options={"task": "squat", "exo_mode": "off"})
        e = []
        for _ in range(400):
            *_, term, _, info = env.step(np.zeros(4))
            assert not term
            e.append(info["reward_terms"]["effort"])
        effort[enabled] = np.mean(e[50:])
    assert effort[True] > 1.1 * effort[False]      # wearing the exo unpowered must cost effort


def test_exo_feedforward_includes_exo_legs(exo):
    from exo_rl.sim.model_builder import segment_info
    seg = segment_info(SubjectParams())
    lp = exo.limb_params(seg)
    assert np.isclose(lp["thigh_mass"], seg.thigh_mass + exo.group_mass("thigh", "l"))
    assert np.isclose(lp["shank_foot_mass"], seg.shank_foot_mass + exo.group_mass("shank", "l"))
    assert exo.group_mass("thigh", "l") > 1.0 and exo.group_mass("shank", "l") > 1.0


def test_exo_mass_scale_option(cfg):
    from exo_rl.envs.randomization import nominal, sample
    assert nominal(cfg).exo_mass_scale == 1.0
    cfg["exo_hardware"]["mass_scale"] = 0.7
    assert nominal(cfg).exo_mass_scale == 0.7
    s = sample(cfg, np.random.default_rng(0)).exo_mass_scale
    assert 0.7 * 0.95 <= s <= 0.7 * 1.05                       # randomization multiplies on top
    total = {}
    for scale, enabled in ((0.7, True), (1.0, True), (1.0, False)):
        cfg["exo_hardware"].update(mass_scale=scale, enabled=enabled)
        total[(scale, enabled)] = ExoAssistEnv(cfg, seed=0).hm.model.body_mass.sum()
    no_exo = total[(1.0, False)]
    ratio = (total[(0.7, True)] - no_exo) / (total[(1.0, True)] - no_exo)
    assert abs(ratio - 0.7) < 0.01
