"""Builds a sagittal-plane human + exoskeleton MuJoCo model scaled to a subject.

Segment masses and lengths follow Winter's anthropometric ratios, so the model can be
regenerated for each subject's height and mass.

Model overview:
Root (pelvis/trunk, "HAT" = head-arms-trunk):
    root_x     slide  (free: the body progresses overground, driven by the legs)
    root_z     slide  (UNactuated: body weight must be carried by the legs)
    root_pitch hinge  (+ = trunk leans forward)
Each leg (l, r):
    hip   (+ = flexion)      human + exo torque
    knee  (+ = flexion)      human + exo torque
    ankle (+ = dorsiflexion) human torque only

All torques are applied through data.qfrc_applied so human, exo, and residual
contributions can be accounted for separately.

The payload (ACE: 25 lb dead weight for a soft exo) is a rigid body attached to the
trunk, i.e. to the suit, not to the limbs.

If SubjectParams.exo is set, the CAD exoskeleton's links (mass, inertia, mesh) are mounted
on the pelvis, thighs, and shanks (see exo_hardware.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

HUMAN_JOINTS = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]
EXO_JOINTS = ["hip_l", "knee_l", "hip_r", "knee_r"]
ROOT_JOINTS = ["root_x", "root_z", "root_pitch"]
IMU_SITES = ["pelvis", "thigh_l", "thigh_r"]

# Winter (2009): (mass fraction of body mass, length fraction of body height)
WINTER = {
    "thigh": (0.100, 0.245),
    "shank": (0.0465, 0.246),
    "foot": (0.0145, 0.152),
}
HAT_MASS_FRAC = 0.678
TRUNK_LEN_FRAC = 0.35
ANKLE_HEIGHT_FRAC = 0.039


@dataclass
class SubjectParams:
    height: float = 1.75            # m
    mass: float = 75.0              # kg (pilot only, without payload)
    payload_mass: float = 11.34     # kg (25 lb). 0 disables the payload.
    payload_offset: tuple = (-0.12, 0.0, 0.40)  # m, relative to pelvis origin (on the exo waist band)
    floor_friction: float = 1.0     # applied to floor AND feet (MuJoCo uses the max of the pair)
    joint_damping: float = 0.5      # Nm·s/rad, passive joint damping
    timestep: float = 0.002
    exo: object = None              # ExoHardware (exo_rl/sim/exo_hardware.py) or None
    exo_mass_scale: float = 1.0
    exo_quality: str = "lo"         # "lo" meshes for training, "hi" for viewing


@dataclass
class SegmentInfo:
    """Segment inertial data used by the analytic feedforward in control/torque_pipeline.py."""
    thigh_mass: float
    thigh_len: float
    thigh_com: float        # distance hip -> thigh COM
    shank_foot_mass: float
    shank_len: float
    shank_foot_com: float   # distance knee -> combined shank+foot COM
    leg_len: float
    extras: dict = field(default_factory=dict)


def segment_info(p: SubjectParams) -> SegmentInfo:
    m_t, l_t = WINTER["thigh"][0] * p.mass, WINTER["thigh"][1] * p.height
    m_s, l_s = WINTER["shank"][0] * p.mass, WINTER["shank"][1] * p.height
    m_f = WINTER["foot"][0] * p.mass
    ha = ANKLE_HEIGHT_FRAC * p.height
    # Winter COM locations: thigh 0.433 from proximal, shank 0.433 from proximal.
    com_s = 0.433 * l_s
    com_sf = (m_s * com_s + m_f * (l_s + 0.5 * ha)) / (m_s + m_f)
    return SegmentInfo(
        thigh_mass=m_t, thigh_len=l_t, thigh_com=0.433 * l_t,
        shank_foot_mass=m_s + m_f, shank_len=l_s, shank_foot_com=com_sf,
        leg_len=l_t + l_s + ha,
    )


def build_mjcf(p: SubjectParams) -> str:
    H, M = p.height, p.mass
    exo_assets, exo_bodies = "", {}
    if p.exo is not None:
        exo_assets, exo_bodies = p.exo.mjcf_snippets(WINTER["thigh"][1] * H, p.exo_mass_scale, p.exo_quality)
    m_hat = HAT_MASS_FRAC * M
    l_trunk = TRUNK_LEN_FRAC * H
    m_t, l_t = WINTER["thigh"][0] * M, WINTER["thigh"][1] * H
    m_s, l_s = WINTER["shank"][0] * M, WINTER["shank"][1] * H
    m_f, l_f = WINTER["foot"][0] * M, WINTER["foot"][1] * H
    ha = ANKLE_HEIGHT_FRAC * H
    r_leg = 0.045 * H / 1.75

    def leg(side: str, y: float) -> str:
        heel_x, toe_x = -0.25 * l_f, 0.75 * l_f
        return f"""
      <body name="thigh_{side}" pos="0 {y} 0">
        <joint name="hip_{side}" type="hinge" axis="0 -1 0" range="-40 125"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-l_t:.4f}" size="{r_leg:.4f}" mass="{m_t:.4f}" rgba=".8 .6 .5 1"/>
        <site name="imu_thigh_{side}" pos="0.05 0 {-0.5 * l_t:.4f}" size="0.01"/>{exo_bodies.get(f"thigh_{side}", "")}
        <body name="shank_{side}" pos="0 0 {-l_t:.4f}">
          <joint name="knee_{side}" type="hinge" axis="0 1 0" range="-5 150"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-l_s:.4f}" size="{0.8 * r_leg:.4f}" mass="{m_s:.4f}" rgba=".8 .6 .5 1"/>{exo_bodies.get(f"shank_{side}", "")}
          <body name="foot_{side}" pos="0 0 {-l_s:.4f}">
            <joint name="ankle_{side}" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot_{side}" type="box" pos="{0.25 * l_f:.4f} 0 {-0.5 * ha:.4f}"
                  size="{0.5 * l_f:.4f} 0.045 {0.5 * ha:.4f}" mass="{m_f:.4f}"
                  contype="1" conaffinity="1" friction="{p.floor_friction} 0.005 0.0001"
                  rgba=".3 .3 .3 1"/>
            <site name="heel_{side}" pos="{heel_x:.4f} 0 {-ha:.4f}" size="0.005"/>
            <site name="toe_{side}" pos="{toe_x:.4f} 0 {-ha:.4f}" size="0.005"/>
          </body>
        </body>
      </body>"""

    payload = ""
    if p.payload_mass > 0:
        ox, oy, oz = p.payload_offset
        payload = f"""
      <body name="payload" pos="{ox:.4f} {oy:.4f} {oz:.4f}">
        <geom type="box" size="0.08 0.12 0.10" mass="{p.payload_mass:.4f}" rgba=".9 .2 .2 1"/>
      </body>"""

    imu_sensors = "\n".join(
        f'    <gyro name="gyro_{s}" site="imu_{s}"/>\n    <accelerometer name="acc_{s}" site="imu_{s}"/>'
        for s in IMU_SITES
    )

    return f"""<mujoco model="human_exo_planar">
  <compiler angle="degree" inertiafromgeom="auto"/>
  <option timestep="{p.timestep}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><headlight ambient=".35 .35 .35" diffuse=".55 .55 .55" specular=".1 .1 .1"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .82" rgb2=".68 .68 .68" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>
{exo_assets}
  </asset>
  <default>
    <joint limited="true" damping="{p.joint_damping}" armature="0.01"/>
    <geom contype="0" conaffinity="0" condim="3" solref="0.005 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <light pos="0 -3 3" dir="0 1 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="1"
          friction="{p.floor_friction} 0.005 0.0001" material="grid"/>
    <body name="pelvis" pos="0 0 0">
      <joint name="root_x" type="slide" axis="1 0 0" limited="false" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" limited="false" damping="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" range="-30 80" damping="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 {l_trunk:.4f}" size="{0.12 * H / 1.75:.4f}"
            mass="{m_hat:.4f}" rgba=".5 .6 .8 1"/>
      <site name="imu_pelvis" pos="-0.1 0 0.05" size="0.01"/>{payload}{exo_bodies.get("pelvis", "")}
{leg("l", 0.09)}
{leg("r", -0.09)}
    </body>
  </worldbody>
  <sensor>
{imu_sensors}
  </sensor>
</mujoco>
"""


class HumanExoModel:
    """Thin wrapper holding the MjModel/MjData plus index lookups."""

    def __init__(self, params: SubjectParams):
        self.params = params
        self.xml = build_mjcf(params)
        assets = params.exo.meshes(params.exo_quality) if params.exo is not None else None
        self.model = mujoco.MjModel.from_xml_string(self.xml, assets)
        self.data = mujoco.MjData(self.model)
        self.segments = segment_info(params)

        def qadr(name):
            return self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]

        def dadr(name):
            return self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]

        self.human_qpos = np.array([qadr(j) for j in HUMAN_JOINTS])
        self.human_dof = np.array([dadr(j) for j in HUMAN_JOINTS])
        self.exo_qpos = np.array([qadr(j) for j in EXO_JOINTS])
        self.exo_dof = np.array([dadr(j) for j in EXO_JOINTS])
        self.root_qpos = np.array([qadr(j) for j in ROOT_JOINTS])
        self.root_dof = np.array([dadr(j) for j in ROOT_JOINTS])
        self.sole_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            for n in ("heel_l", "toe_l", "heel_r", "toe_r")
        ]

    def set_pose(self, q_human: np.ndarray, trunk_pitch: float, qd_human=None, root_vx: float = 0.0,
                 clearance: float = 0.0):
        """Place the model in a pose with the lowest sole point touching the floor.

        root_vx: initial forward pelvis velocity. Use the motion's implied speed so gait
        starts with consistent momentum instead of from rest.
        """
        m, d = self.model, self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[self.human_qpos] = q_human
        d.qpos[self.root_qpos[2]] = trunk_pitch
        mujoco.mj_kinematics(m, d)
        min_z = min(d.site_xpos[s][2] for s in self.sole_sites)
        d.qpos[self.root_qpos[1]] = -min_z + clearance
        if qd_human is not None:
            d.qvel[self.human_dof] = qd_human
        d.qvel[self.root_dof[0]] = root_vx
        mujoco.mj_forward(m, d)

    def implied_speed(self, motion, n: int = 200) -> float:
        """Average forward speed implied by a motion if the stance foot does not slip.

        The stance foot is taken to be the lowest sole point; its backward velocity relative
        to the pelvis equals the pelvis' forward speed. Used for the root velocity helper.
        """
        m, d = self.model, self.data
        saved_q, saved_v = d.qpos.copy(), d.qvel.copy()
        xs, ids = [], []
        for i in range(n):
            q, trunk = motion.angles(motion.period * i / n)
            d.qpos[:] = 0.0
            d.qpos[self.human_qpos] = q
            d.qpos[self.root_qpos[2]] = trunk
            mujoco.mj_kinematics(m, d)
            k = int(np.argmin([d.site_xpos[s][2] for s in self.sole_sites]))
            ids.append(k)
            xs.append(d.site_xpos[self.sole_sites[k]][0])
        dt = motion.period / n
        v = [-(xs[(i + 1) % n] - xs[i]) / dt for i in range(n) if ids[i] == ids[(i + 1) % n]]
        d.qpos[:], d.qvel[:] = saved_q, saved_v
        mujoco.mj_forward(m, d)
        return float(np.mean(v)) if v else 0.0

    def sensors(self) -> dict:
        """Raw sensor readings, in the same layout the hardware provides (see deploy/)."""
        d = self.data
        s = {
            "q_exo": d.qpos[self.exo_qpos].copy(),
            "qd_exo": d.qvel[self.exo_dof].copy(),
            "trunk_pitch": float(d.qpos[self.root_qpos[2]]),
        }
        for name in IMU_SITES:
            s[f"gyro_{name}"] = d.sensor(f"gyro_{name}").data.copy()
            s[f"acc_{name}"] = d.sensor(f"acc_{name}").data.copy()
        return s
