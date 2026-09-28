"""CAD exoskeleton integration: mass bookkeeping, motor signs, stability, burden."""
import mujoco
import numpy as np
import pytest

from deploy.joint_map import MotorMap
from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.sim.exo_hardware import ExoHardware
from exo_rl.sim.model_builder import HumanExoModel, SubjectParams


@pytest.fixture(scope="module")
def exo():
    return ExoHardware("configs/exo_mk1.yaml")


def test_mass_is_conserved(exo):
    assert abs(exo.total_mass - exo.model.body_mass.sum()) < 1e-6
    with_exo = HumanExoModel(SubjectParams(exo=exo)).model.body_mass.sum()
    without = HumanExoModel(SubjectParams()).model.body_mass.sum()
    assert abs(with_exo - without - exo.total_mass) < 1e-4


@pytest.mark.parametrize("joint", ["hip_l", "knee_l", "hip_r", "knee_r"])
def test_motor_sign_makes_flexion(exo, joint):
    """Command +human flexion through the MotorMap on the standalone CAD model (no gravity)
    and check the limb moves the right way: hip flexion -> thigh swings forward;
    knee flexion -> shank swings backward."""
    m = exo.model
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    mm = exo.motor_map()
    i = mm["order"].index(joint)
    motor = MotorMap(mm["signs"]).tau_to_motor(np.eye(4)[i] * 5.0)[i]
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mm["urdf_joints"][i])
    side = joint[-1]
    link = exo.spec["legs"][side]["thigh_links" if joint.startswith("hip") else "shank_links"][0]
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
    mujoco.mj_forward(m, d)
    start = exo.R @ d.xipos[b]
    for _ in range(50):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[jid]] = motor
        mujoco.mj_step(m, d)
    moved_forward = (exo.R @ d.xipos[b])[0] - start[0]
    if joint.startswith("hip"):
        assert moved_forward > 1e-3
    else:
        assert moved_forward < -1e-3
    m.opt.gravity[:] = [0, 0, -9.81]


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
    assert effort[True] > 1.1 * effort[False]      # wearing 19 kg unpowered must cost effort


def test_exo_feedforward_includes_exo_legs(exo):
    from exo_rl.sim.model_builder import segment_info
    seg = segment_info(SubjectParams())
    lp = exo.limb_params(seg)
    assert lp["thigh_mass"] > seg.thigh_mass + 3.0 and lp["shank_foot_mass"] > seg.shank_foot_mass + 2.0
