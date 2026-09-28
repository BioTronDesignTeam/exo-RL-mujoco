"""Physics sanity: the human model performs every task, alone and in circuits."""
import itertools

import numpy as np
import pytest

from exo_rl.envs.exo_env import ExoAssistEnv
from exo_rl.sim.reference import ReferenceTrajectory


@pytest.mark.parametrize("task", ["squat", "high_knees", "quick_steps", "backward_walk", "walk"])
def test_human_tracks_task_without_exo(cfg, task):
    cfg["randomization"]["enabled"] = False
    env = ExoAssistEnv(cfg, seed=0)
    env.reset(options={"task": task, "exo_mode": "off"})
    errs = []
    for _ in range(400):
        *_, term, _, info = env.step(np.zeros(4))
        assert not term, f"{task} terminated: {info['terminated_reason']}"
        errs.append(np.degrees(np.sqrt(np.mean((info["q"] - info["q_ref"]) ** 2))))
    assert np.mean(errs[100:]) < 8.0


def test_all_task_transitions(cfg):
    cfg["randomization"]["enabled"] = False
    env = ExoAssistEnv(cfg, seed=0)
    names = cfg["tasks"]["names"]
    forbidden = {tuple(x) for x in cfg["tasks"].get("forbidden_transitions", [])}
    for a, b in itertools.permutations(names, 2):
        if (a, b) in forbidden:
            continue
        env.reset(options={"task": a, "exo_mode": "off"})
        env.ref = ReferenceTrajectory([(env.motions[a], 2.0), (env.motions[b], 2.0)], blend=0.5)
        env.speeds = {m.name: env.hm.implied_speed(m) for m, _ in env.ref.segments}
        q, qd, _, tr, _ = env.ref.sample(0.0)
        env.hm.set_pose(q, tr, qd, root_vx=env._v_ref(0.0))
        env.human.reset()
        for _ in range(380):
            *_, term, _, info = env.step(np.zeros(4))
            assert not term, f"{a}->{b} failed: {info['terminated_reason']}"


def test_circuits_respect_forbidden_transitions(cfg):
    env = ExoAssistEnv(cfg, seed=0)
    forbidden = {tuple(x) for x in cfg["tasks"]["forbidden_transitions"]}
    for _ in range(200):
        order = env._circuit_order(cfg["tasks"])
        assert not any((a, b) in forbidden for a, b in zip(order, order[1:]))
    cfg["tasks"]["circuit_order"] = "ace"
    assert env._circuit_order(cfg["tasks"]) == ["squat", "high_knees", "quick_steps", "backward_walk"]


def test_randomized_subjects_stay_stable(cfg):
    env = ExoAssistEnv(cfg, seed=123)
    fails = 0
    for _ in range(10):
        env.reset(options={"exo_mode": "off"})
        for _ in range(300):
            *_, term, _, _ = env.step(np.zeros(4))
            if term:
                fails += 1
                break
    assert fails <= 1


def test_csv_motion_matches_source():
    from exo_rl.sim.reference import SYNTHETIC, TabulatedMotion
    csv_m = TabulatedMotion("data/motions/example_walk.csv", "walk")
    src = SYNTHETIC["walk"]
    for t in np.linspace(0, 2 * src.period, 37):
        q1, tr1 = csv_m.angles(t)
        q2, tr2 = src.angles(t)
        assert np.max(np.abs(np.degrees(q1 - q2))) < 1.5
        assert abs(np.degrees(tr1 - tr2)) < 0.1


def test_data_dir_prefers_csv(cfg):
    from exo_rl.sim.reference import TabulatedMotion, load_motions
    m = load_motions(["example_walk", "squat"], "data/motions")
    assert isinstance(m["example_walk"], TabulatedMotion)
    assert not isinstance(m["squat"], TabulatedMotion)


@pytest.mark.parametrize("task,exo", [("high_knees_travel", False), ("high_knees_travel", True), ("jog", False)])
def test_extra_motions_travel_forward(cfg, task, exo):
    cfg["randomization"]["enabled"] = False
    cfg["exo_hardware"]["enabled"] = exo
    env = ExoAssistEnv(cfg, seed=0)
    env.reset(seed=1, options={"task": task, "exo_mode": "off", "nominal": True})
    x0 = env.hm.data.qpos[env.hm.root_qpos[0]]
    for _ in range(500):
        *_, term, _, info = env.step(np.zeros(4))
        assert not term, f"{task} (exo={exo}) failed: {info['terminated_reason']}"
    assert (env.hm.data.qpos[env.hm.root_qpos[0]] - x0) / env.t > 0.3     # actually moves forward
