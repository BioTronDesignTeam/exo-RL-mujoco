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
