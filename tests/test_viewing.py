import os

import numpy as np

from exo_rl.agents.td3 import TD3Agent
from exo_rl.viewing import CheckpointWatcher


def test_checkpoint_watcher_keeps_previous_and_picks_up_replacement(tmp_path, cfg):
    path = tmp_path / "latest.pt"
    w = CheckpointWatcher(str(path))
    assert not w.poll() and w.policy is None                     # missing: no crash, no policy

    a1 = TD3Agent(10, 4, cfg["agent"])
    a1.save(path, step=5000)
    assert w.poll() and w.step == 5000
    o = np.random.randn(10).astype(np.float32)
    assert np.allclose(w.policy(o), a1.act(o))
    assert not w.poll()                                          # unchanged file: no reload

    a2 = TD3Agent(10, 4, cfg["agent"])
    a2.save(path, step=10000)                                    # replaced (atomic rename)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))   # coarse-mtime filesystems
    assert w.poll() and w.step == 10000
    assert np.allclose(w.policy(o), a2.act(o))

    os.remove(path)                                              # missing again: keep previous
    assert not w.poll() and w.step == 10000
    path.write_bytes(b"not a checkpoint")                        # corrupt: keep previous
    assert not w.poll() and np.allclose(w.policy(o), a2.act(o))
