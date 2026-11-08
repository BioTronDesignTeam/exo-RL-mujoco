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

Install: myo-sim is in requirements.txt (Apache-2.0, needs mujoco >= 3.4)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from exo_rl.sim.model_builder import HUMAN_JOINTS, IMU_SITES, TRUNK_LEN_FRAC, SegmentInfo, SubjectParams


def myo_models_dir() -> Path:
    """Folder of the installed myo-sim models (raises ImportError if myo-sim is missing)."""
    import myo_sim
    return Path(myo_sim.__file__).parent / "models"


# Exo motor order matches ExoHardware.motor_map(); + = flexion in both models.
EXO_JOINTS = {"hip_l": "hip_flexion_l", "knee_l": "knee_angle_l", "hip_r": "hip_flexion_r", "knee_r": "knee_angle_r"}
# MyoLeg bodies use OpenSim axes (x forward, y up, z right); the exo snippets use x forward, y left, z up.
TO_OPENSIM = "0.70711 -0.70711 0 0"
# MyoLeg faces world -y; turned 90 deg so it faces +x like the planar model (the root frame is then
# x forward, y left, z up, and root orientation errors read as pitch about y, roll about x).
FACE_X = "0.70711 0 0 0.70711"
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
    imu = '<site name="imu_pelvis" pos="-0.1 0 0.05" size="0.01"/>'      # as in the planar model
    hx, hy, _ = HIP_CENTER
    return _frame("hip_frame", f"{hx} {hy} 0", torso + payload + imu + exo_pelvis)


def _with_subject(chain: str, p: SubjectParams) -> str:
    """MyoLeg bone chain with the upper body and exo inserted as the first children of the matching bones."""
    bodies = {}
    if p.exo is not None:
        _, bodies = p.exo.mjcf_snippets(FEMUR_LEN, p.exo_mass_scale, p.exo_quality, hip_half_width=HIP_CENTER[2])
    inserts = {"pelvis": _hip_frame(p, bodies.get("pelvis", ""))}
    for s in "lr":                     # thigh IMU mid-femur, front, in the planar model's axes
        inserts[f"femur_{s}"] = (f'\n<site name="imu_thigh_{s}" pos="0.05 {-0.5 * FEMUR_LEN:.4f} 0" '
                                 f'quat="{TO_OPENSIM}" size="0.01"/>')
    if p.exo is not None:
        for bone, part in (("femur_l", "thigh_l"), ("femur_r", "thigh_r"), ("tibia_l", "shank_l"), ("tibia_r", "shank_r")):
            inserts[bone] = inserts.get(bone, "") + _frame(f"exo_frame_{part}", "0 0 0", bodies.get(part, ""))
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
    exo_assets = exo_motors = option = sensors = ""
    friction = "1 0.005 0.0001"
    if p is not None:
        option = f'<option timestep="{p.timestep}" integrator="implicitfast"/>'
        friction = f"{p.floor_friction} 0.005 0.0001"
        sensors = "<sensor>\n" + "\n".join(f'    <gyro name="gyro_{n}" site="imu_{n}"/>\n    <accelerometer name="acc_{n}" '
                                          f'site="imu_{n}"/>' for n in IMU_SITES) + "\n  </sensor>"
        text =(myo_models_dir() / "leg/assets/myolegs_chain.xml").read_text(encoding="utf-8")
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
  <size memory="16M"/>  <!-- the assets' legacy nconmax/njmax reserve 1.3 GB per MjData; ~0.1 MB is used -->
  {option}
  {exo_assets}
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="50 50 0.1" rgba=".9 .9 .9 1" conaffinity="1" contype="1"
          friction="{friction}"/>
    <body name="root" pos="0 0 1">
      <freejoint name="root"/>
      <body name="facing" quat="{FACE_X}">
        {chain}
      </body>
    </body>
  </worldbody>
  {exo_motors}
  {sensors}
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


class MyoHumanExoModel:
    """MyoLeg counterpart of model_builder.HumanExoModel: same index lookups, set_pose, implied_speed,
    and sensors, so ExoAssistEnv can run either model (config human.model: torque | myo).

    human_qpos/human_dof are the six joints the planar model has (hip flexion, knee, ankle per side);
    root_qpos/root_dof are the free joint's (x, z) entries, as the planar model's root_x, root_z.
    The leg geometry is MyoLeg's fixed size: SubjectParams.height only sets the trunk length, and
    joint_damping is not used (MyoLeg has its own). Floor friction applies to the floor; MuJoCo uses
    the larger of floor and foot friction, so values below the feet's 1.0 have no effect."""

    def __init__(self, params: SubjectParams):
        from .muscle_human import TRACKED
        self.params = params
        self.model = load_myolegs(params)
        self.data = mujoco.MjData(self.model)
        m = self.model
        joint = lambda n: m.joint(n).id  # noqa: E731
        tracked = [joint(TRACKED[j]) for j in HUMAN_JOINTS]
        self.human_qpos = m.jnt_qposadr[tracked]
        self.human_dof = m.jnt_dofadr[tracked]
        exo = [joint(EXO_JOINTS[j]) for j in EXO_JOINTS]
        self.exo_qpos, self.exo_dof = m.jnt_qposadr[exo], m.jnt_dofadr[exo]
        root = joint("root")
        self.root_qpos = m.jnt_qposadr[root] + np.array([0, 2])
        self.root_dof = m.jnt_dofadr[root] + np.array([0, 2])
        self.root_body = m.jnt_bodyid[root]
        self.feet = [i for i in range(m.ngeom)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(("foot", "bofoot"))]
        self.segments = self._segment_info()

    def _segment_info(self) -> SegmentInfo:
        """Leg segment data for the exo's gravity feedforward (left leg, standing straight)."""
        m, d = self.model, mujoco.MjData(self.model)
        from .muscle_human import stand_on_floor
        stand_on_floor(m, d, np.zeros(len(HUMAN_JOINTS)), 0.0)
        hip, knee, ankle = (d.xanchor[m.joint(n).id] for n in ("hip_flexion_l", "knee_angle_l", "ankle_angle_l"))

        def group(bodies):
            ids = [m.body(b).id for b in bodies]
            mass = float(m.body_mass[ids].sum())
            return mass, (m.body_mass[ids, None] * d.xipos[ids]).sum(axis=0) / mass
        tm, tc = group(["femur_l", "patella_l"])
        sm, sc = group(["tibia_l", "talus_l", "calcn_l", "toes_l"])
        return SegmentInfo(thigh_mass=tm, thigh_len=float(np.linalg.norm(knee - hip)),
                           thigh_com=float(np.linalg.norm(tc - hip)), shank_foot_mass=sm,
                           shank_len=float(np.linalg.norm(ankle - knee)),
                           shank_foot_com=float(np.linalg.norm(sc - knee)), leg_len=float(hip[2]))

    def set_pose(self, q_human: np.ndarray, trunk_pitch: float, qd_human=None, root_vx: float = 0.0,
                 clearance: float = 0.0):
        """Pose with the lowest foot touching the floor (see HumanExoModel.set_pose)."""
        from .muscle_human import stand_on_floor
        m, d = self.model, self.data
        stand_on_floor(m, d, q_human, trunk_pitch)
        d.qpos[self.root_qpos[1]] += clearance
        if qd_human is not None:
            d.qvel[self.human_dof] = qd_human
        d.qvel[self.root_dof[0]] = root_vx
        mujoco.mj_forward(m, d)

    def implied_speed(self, motion, n: int = 200) -> float:
        """Average forward speed implied by a motion if the lowest foot does not slip."""
        from .muscle_human import satisfy_couplings
        m, d = self.model, mujoco.MjData(self.model)
        xs, ids = [], []
        for i in range(n):
            q, trunk = motion.angles(motion.period * i / n)
            d.qpos[:] = m.qpos0
            d.qpos[self.human_qpos] = q
            satisfy_couplings(m, d)
            adr = m.jnt_qposadr[m.joint("root").id]
            mujoco.mju_axisAngle2Quat(d.qpos[adr + 3:adr + 7], np.array([0.0, 1.0, 0.0]), trunk)
            mujoco.mj_kinematics(m, d)
            k = self.feet[int(np.argmin(d.geom_xpos[self.feet, 2]))]
            ids.append(k)
            xs.append(d.geom_xpos[k, 0])
        dt = motion.period / n
        v = [-(xs[(i + 1) % n] - xs[i]) / dt for i in range(n) if ids[i] == ids[(i + 1) % n]]
        return float(np.mean(v)) if v else 0.0

    def trunk_pitch(self) -> float:
        R = self.data.xmat[self.root_body].reshape(3, 3)
        return float(np.arctan2(R[0, 2], R[2, 2]))

    def sensors(self) -> dict:
        """Raw sensor readings, in the planar model's layout."""
        d = self.data
        s = {"q_exo": d.qpos[self.exo_qpos].copy(), "qd_exo": d.qvel[self.exo_dof].copy(),
             "trunk_pitch": self.trunk_pitch()}
        for name in IMU_SITES:
            s[f"gyro_{name}"] = d.sensor(f"gyro_{name}").data.copy()
            s[f"acc_{name}"] = d.sensor(f"acc_{name}").data.copy()
        return s


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
