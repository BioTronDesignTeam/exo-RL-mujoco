import numpy as np
import pytest

from exo_rl.agents.td3 import ReplayBuffer, TD3Agent
from exo_rl.utils import atomic_write


def test_td3_and_ddpg_update_and_save(tmp_path, cfg):
    for algo in ("td3", "ddpg"):
        cfg["agent"]["algo"] = algo
        agent = TD3Agent(10, 4, cfg["agent"])
        buf = ReplayBuffer(10, 4, 1000)
        for _ in range(300):
            buf.add(np.random.randn(10), np.random.uniform(-1, 1, 4), 1.0, np.random.randn(10), False)
        for _ in range(4):
            info = agent.update(buf, 32)
        assert np.isfinite(info["critic_loss"])
        path = tmp_path / f"{algo}.pt"
        agent.save(path)
        agent2 = TD3Agent.load(path)
        o = np.random.randn(10).astype(np.float32)
        assert np.allclose(agent.act(o), agent2.act(o))


def test_init_from_copies_nets_and_rejects_mismatch(tmp_path, cfg):
    src = TD3Agent(10, 4, cfg["agent"])
    path = tmp_path / "src.pt"
    src.save(path, step=123)
    dst = TD3Agent(10, 4, cfg["agent"])
    dst.init_from(path)
    o = np.random.randn(10).astype(np.float32)
    assert np.allclose(src.act(o), dst.act(o)) and dst.updates == 0
    with pytest.raises(ValueError, match="obs 10"):
        TD3Agent(12, 4, cfg["agent"]).init_from(path)
    with pytest.raises(ValueError, match="hidden"):
        TD3Agent(10, 4, {**cfg["agent"], "hidden": [32, 32]}).init_from(path)


def test_atomic_write_replaces_whole_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old")
    atomic_write(str(p), lambda tmp: open(tmp, "w").write("new"))
    assert p.read_text() == "new" and not (tmp_path / "f.txt.tmp").exists()


def test_replay_buffer_save_load_and_prefill(tmp_path):
    buf = ReplayBuffer(6, 4, 50)
    for i in range(30):
        buf.add(np.full(6, i), np.zeros(4), float(i), np.full(6, i + 1), terminated=(i == 29))
    buf.save(tmp_path / "b.npz")
    same = ReplayBuffer(6, 4, 50)
    same.load(tmp_path / "b.npz")
    assert len(same) == 30 and np.array_equal(same.obs, buf.obs) and same.not_done[29] == 0.0
    other = ReplayBuffer(6, 4, 100)
    other.add(np.zeros(6), np.zeros(4), 0.0, np.zeros(6), False)
    assert other.add_from_file(tmp_path / "b.npz") == 30 and len(other) == 31
    assert other.rew[30] == 29.0 and other.not_done[30] == 0.0
    with pytest.raises(ValueError, match="obs size"):
        ReplayBuffer(8, 4, 10).add_from_file(tmp_path / "b.npz")
