"""Musculoskeletal human model (MyoLeg from MyoSuite's myo-sim package).

First step of the musculoskeletal integration: load the MyoLeg model (bilateral legs + pelvis,
80 muscle-tendon units, derived from Rajagopal et al.'s gait model) so it can later replace the
torque-driven human in exo_rl/sim/human.py. Given SubjectParams it adds a rigid upper body and the
payload on the pelvis, and the CAD exo if p.exo is set: its links ride on the pelvis, femurs, and
tibias, and four motors drive hip flexion and knee flexion.

The myo-sim package ships the model's parts (bone chain, muscles, tendons, mesh assets) but no
top-level file; myolegs_xml() assembles them the same way MyoSuite's myolegs_with_torso.xml
does, on our own floor; MyoSuite's torso is not part of the pip package, so ours is a capsule
sized like the planar model's trunk.

Install: pip install -r requirements-myo.txt   (myo-sim, Apache-2.0, needs mujoco >= 3.4)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from exo_rl.sim.model_builder import TRUNK_LEN_FRAC, SubjectParams


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
HIP_TO_PELVIS = 0.07849              # pelvis origin height above the hip centers
FEMUR_LEN = 0.404425
MYOLEGS_MASS = 37.8227               # legs + pelvis, kg


def _frame(name: str, pos: str, children: str) -> str:
    """Child body in the planar model's axes (x forward, y left, z up)."""
    return f'\n<body name="{name}" pos="{pos}" quat="{TO_OPENSIM}">{children}</body>'


def _hip_frame(p: SubjectParams, exo_pelvis: str) -> str:
    """Rigid upper body, payload, and exo pelvis parts, in a frame at the hip centers' midpoint
    (the planar model's pelvis origin, so payload offsets and exo parts carry over unchanged)."""
    torso_mass = p.mass - MYOLEGS_MASS
    if torso_mass <= 0:
        raise ValueError(f"subject mass {p.mass} kg must exceed the MyoLeg legs + pelvis ({MYOLEGS_MASS:.1f} kg)")
    l_trunk = TRUNK_LEN_FRAC * p.height
    torso = (f'<geom name="torso" type="capsule" fromto="0 0 {HIP_TO_PELVIS:.4f} 0 0 {l_trunk:.4f}" '
             f'size="{0.12 * p.height / 1.75:.4f}" mass="{torso_mass:.4f}" contype="0" conaffinity="0" '
             f'rgba=".5 .6 .8 1"/>')
    payload = ""
    if p.payload_mass > 0:
        ox, oy, oz = p.payload_offset
        payload = (f'<body name="payload" pos="{ox:.4f} {oy:.4f} {oz:.4f}"><geom type="box" size="0.08 0.12 0.10" '
                   f'mass="{p.payload_mass:.4f}" contype="0" conaffinity="0" rgba=".9 .2 .2 1"/></body>')
    hx, hy, _ = HIP_CENTER
    return _frame("hip_frame", f"{hx} {hy} 0", torso + payload + exo_pelvis)


def _with_subject(chain: str, p: SubjectParams) -> str:
    """MyoLeg bone chain with the upper body and exo inserted as the first children of the matching bones."""
    bodies = {}
    if p.exo is not None:
        _, bodies = p.exo.mjcf_snippets(FEMUR_LEN, p.exo_mass_scale, p.exo_quality, hip_half_width=HIP_CENTER[2])
    inserts = {"pelvis": _hip_frame(p, bodies.get("pelvis", ""))}
    if p.exo is not None:
        for bone, part in (("femur_l", "thigh_l"), ("femur_r", "thigh_r"), ("tibia_l", "shank_l"), ("tibia_r", "shank_r")):
            inserts[bone] = _frame(f"exo_frame_{part}", "0 0 0", bodies.get(part, ""))
    for bone, xml in inserts.items():
        start = chain.index(f'<body name="{bone}"')
        end = chain.index(">", start) + 1
        chain = chain[:end] + xml + chain[end:]
    return chain


def myolegs_xml(p: SubjectParams | None = None) -> str:
    """Top-level MJCF for MyoLeg with a free-floating pelvis on a flat floor. With SubjectParams it also
    gets a rigid upper body (subject mass minus the legs), the payload, and the exo if p.exo is set;
    without, it is the bare legs + pelvis."""
    root = myo_models_dir().as_posix()
    leg = f"{root}/leg/assets"
    chain = f'<include file="{leg}/myolegs_chain.xml"/>'
    exo_assets = exo_motors = ""
    if p is not None:
        text = (myo_models_dir() / "leg/assets/myolegs_chain.xml").read_text(encoding="utf-8")
        inner = text[text.index(">", text.index("<mujocoinclude")) + 1:text.rindex("</mujocoinclude>")]
        chain = _with_subject(inner, p)
    if p is not None and p.exo is not None:
        exo_assets = "<asset>\n" + p.exo.mjcf_snippets(FEMUR_LEN, p.exo_mass_scale, p.exo_quality)[0] + "\n  </asset>"
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


def load_myolegs(p: SubjectParams | None = None) -> mujoco.MjModel:
    xml = myolegs_xml(p)
    if p is None or p.exo is None:
        return mujoco.MjModel.from_xml_string(xml)
    root = myo_models_dir().as_posix()           # exo meshes are looked up under MyoLeg's meshdir
    assets = {f"{root}/{k}": v for k, v in p.exo.meshes(p.exo_quality).items()}
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
