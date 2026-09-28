import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


@pytest.fixture
def cfg():
    from exo_rl.utils import load_config
    return load_config(os.path.join(ROOT, "configs", "default.yaml"))
