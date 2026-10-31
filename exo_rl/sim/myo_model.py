"""Musculoskeletal human model (MyoLeg from MyoSuite's myo-sim package).

First step of the musculoskeletal integration: load the MyoLeg model (bilateral legs + pelvis,
80 muscle-tendon units, derived from Rajagopal et al.'s gait model) so it can later replace the
torque-driven human in exo_rl/sim/human.py. The exo is not attached yet.

The myo-sim package ships the model's parts (bone chain, muscles, tendons, mesh assets) but no
top-level file; myolegs_xml() assembles them the same way MyoSuite's myolegs_with_torso.xml
does, on our own floor and without MyoSuite's rigid torso (not part of the pip package).

Install: pip install -r requirements-myo.txt   (myo-sim, Apache-2.0, needs mujoco >= 3.4)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def myo_models_dir() -> Path:
    """Folder of the installed myo-sim models (raises ImportError if myo-sim is missing)."""
    import myo_sim
    return Path(myo_sim.__file__).parent / "models"


def myolegs_xml() -> str:
    """Top-level MJCF for MyoLeg (legs + pelvis) with a free-floating pelvis on a flat floor."""
    root = myo_models_dir().as_posix()
    leg = f"{root}/leg/assets"
    return f"""<mujoco model="myoleg_exo_rl">
  <compiler meshdir="{root}" texturedir="{root}" angle="radian"/>
  <include file="{leg}/myolegs_tendon.xml"/>
  <include file="{leg}/myolegs_muscle.xml"/>
  <include file="{leg}/myolegs_assets.xml"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba=".9 .9 .9 1" conaffinity="1" contype="1"/>
    <body name="root" pos="0 0 1">
      <freejoint name="root"/>
      <include file="{leg}/myolegs_chain.xml"/>
    </body>
  </worldbody>
</mujoco>"""


def load_myolegs() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(myolegs_xml())


def summary(model: mujoco.MjModel) -> dict:
    """Muscles, joints, and mass of a loaded MyoLeg model."""
    name = lambda kind, i: mujoco.mj_id2name(model, kind, i)  # noqa: E731
    muscles = [name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
               if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE]
    joints = [name(mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    return {"muscles": muscles, "joints": joints, "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "mass_kg": float(np.sum(model.body_mass))}


if __name__ == "__main__":
    m = load_myolegs()
    s = summary(m)
    print(f"MyoLeg: {len(s['muscles'])} muscles, {len(s['joints'])} joints, nq={s['nq']} nv={s['nv']}, "
          f"mass {s['mass_kg']:.1f} kg")
    print("joints:", ", ".join(s["joints"]))
