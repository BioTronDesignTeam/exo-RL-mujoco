"""Musculoskeletal human model (MyoLeg from MyoSuite's myo-sim package).

First step of the musculoskeletal integration: load the MyoLeg model (bilateral legs + pelvis,
80 muscle-tendon units, derived from Rajagopal et al.'s gait model) so it can later replace the
torque-driven human in exo_rl/sim/human.py. Pass an ExoHardware to attach the CAD exo: its links
ride on the pelvis, femurs, and tibias, and four motors drive hip flexion and knee flexion.

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


# Exo motor order matches ExoHardware.motor_map(); + = flexion in both models.
EXO_JOINTS = {"hip_l": "hip_flexion_l", "knee_l": "knee_angle_l", "hip_r": "hip_flexion_r", "knee_r": "knee_angle_r"}
# MyoLeg bodies use OpenSim axes (x forward, y up, z right); the exo snippets use x forward, y left, z up.
TO_OPENSIM = "0.70711 -0.70711 0 0"
# Hip centers in the MyoLeg pelvis frame (femur_l/r body pos) and femur length (tibia body pos).
HIP_CENTER = (-0.056276, -0.07849, 0.07726)
FEMUR_LEN = 0.404425


def _exo_frame(name: str, pos: str, children: str) -> str:
    return f'\n<body name="{name}" pos="{pos}" quat="{TO_OPENSIM}">{children}</body>'


def _with_exo(chain: str, exo, mass_scale: float, quality: str) -> str:
    """MyoLeg bone chain with the exo links inserted as the first children of the matching bones."""
    _, bodies = exo.mjcf_snippets(FEMUR_LEN, mass_scale, quality, hip_half_width=HIP_CENTER[2])
    hx, hy, _ = HIP_CENTER
    attach = {"pelvis": ("pelvis", f"{hx} {hy} 0"),
              "femur_l": ("thigh_l", "0 0 0"), "femur_r": ("thigh_r", "0 0 0"),
              "tibia_l": ("shank_l", "0 0 0"), "tibia_r": ("shank_r", "0 0 0")}
    for bone, (part, pos) in attach.items():
        start = chain.index(f'<body name="{bone}"')
        end = chain.index(">", start) + 1
        chain = chain[:end] + _exo_frame(f"exo_frame_{part}", pos, bodies.get(part, "")) + chain[end:]
    return chain


def myolegs_xml(exo=None, exo_mass_scale: float = 1.0, exo_quality: str = "lo") -> str:
    """Top-level MJCF for MyoLeg (legs + pelvis) with a free-floating pelvis on a flat floor,
    optionally wearing the exo (an ExoHardware)."""
    root = myo_models_dir().as_posix()
    leg = f"{root}/leg/assets"
    chain = f'<include file="{leg}/myolegs_chain.xml"/>'
    exo_assets = exo_motors = ""
    if exo is not None:
        text = (myo_models_dir() / "leg/assets/myolegs_chain.xml").read_text(encoding="utf-8")
        inner = text[text.index(">", text.index("<mujocoinclude")) + 1:text.rindex("</mujocoinclude>")]
        chain = _with_exo(inner, exo, exo_mass_scale, exo_quality)
        exo_assets = "<asset>\n" + exo.mjcf_snippets(FEMUR_LEN, exo_mass_scale, exo_quality)[0] + "\n  </asset>"
        exo_motors = "<actuator>\n" + "\n".join(
            f'    <motor name="exo_{j}" joint="{joint}" gear="1"/>' for j, joint in EXO_JOINTS.items()) + "\n  </actuator>"
    return f"""<mujoco model="myoleg_exo_rl">
  <compiler meshdir="{root}" texturedir="{root}" angle="radian"/>
  <include file="{leg}/myolegs_tendon.xml"/>
  <include file="{leg}/myolegs_muscle.xml"/>
  <include file="{leg}/myolegs_assets.xml"/>
  {exo_assets}
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba=".9 .9 .9 1" conaffinity="1" contype="1"/>
    <body name="root" pos="0 0 1">
      <freejoint name="root"/>
      {chain}
    </body>
  </worldbody>
  {exo_motors}
</mujoco>"""


def load_myolegs(exo=None, exo_mass_scale: float = 1.0, exo_quality: str = "lo") -> mujoco.MjModel:
    xml = myolegs_xml(exo, exo_mass_scale, exo_quality)
    if exo is None:
        return mujoco.MjModel.from_xml_string(xml)
    root = myo_models_dir().as_posix()           # exo meshes are looked up under MyoLeg's meshdir
    assets = {f"{root}/{k}": v for k, v in exo.meshes(exo_quality).items()}
    return mujoco.MjModel.from_xml_string(xml, assets)


def exo_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    """Actuator ids of the exo motors, in EXO_JOINTS order (hip_l, knee_l, hip_r, knee_r)."""
    return np.array([model.actuator(f"exo_{j}").id for j in EXO_JOINTS])


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
