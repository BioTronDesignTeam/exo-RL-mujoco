"""Experimental MyoLeg26 adapter for the planar exoskeleton task.

The upstream muscle/tendon dynamics and coupled moving muscle attachment points
are retained. Root motion is planar and hip adduction/rotation are constrained.
The controller recruits muscles to track a reference; it is not a learned or
physiologically validated model of human motor control.
An optional lumbar hinge uses bounded ideal support, logged separately from
the 26 leg-muscle activations.
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import cast

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .human import HumanController
from .exo_hardware import ExoHardware
from .model_builder import (
    HumanExoModel,
    IMU_SITES,
    ROOT_JOINTS,
    SegmentInfo,
    SubjectParams,
)

ROOT = Path(__file__).resolve().parents[2]
JOINTS = [
    f"{joint}_{side}"
    for side in ("l", "r")
    for joint in ("hip_flexion", "knee_angle", "ankle_angle")
]
SIGNS = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
EXO_INDICES = [0, 1, 3, 4]
MOUNT_QUAT = "0.7071067811865476 -0.7071067811865476 0 0"


def _scale_anatomy(root: ET.Element, native: mujoco.MjModel, scale: float) -> None:
    """Uniformly resize anatomy while retaining body masses and muscle force capacity.

    Lengths, mesh coordinates, inertias, and coupled slide coordinates must agree.
    Muscle lengthrange is dimensional; gain/bias length parameters are normalized.
    The scene and exoskeleton are excluded from this geometric scaling.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("human.myo.scale must be finite and positive")
    body = root.find("worldbody/body[@name='Full Body']")
    assert body is not None
    # Use compiled inertias, including upstream compiler repairs, rather than
    # rounded XML values. Also freeze masses otherwise inferred from mesh volume.
    for element in body.iter("body"):
        native_body = native.body(element.attrib["name"])
        if native_body.mass[0] > 0:
            inertial = element.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(element, "inertial")
            inertial.attrib.clear()
            inertial.set("mass", str(native_body.mass[0]))
            inertial.set("pos", " ".join(map(str, native_body.ipos)))
            inertial.set("quat", " ".join(map(str, native_body.iquat)))
            inertia = native_body.inertia.copy() * scale**2
            largest = int(np.argmax(inertia))
            other_sum = float(np.delete(inertia, largest).sum())
            # Equality is physically valid, but needs a tiny margin for XML
            # round trips (e.g. lumbar5: 0.01219 + 0.00546 == 0.01765).
            inertia[largest] = min(inertia[largest], other_sum * (1.0 - 1e-12))
            inertial.set("diaginertia", " ".join(format(v, ".17g") for v in inertia))

    def multiply(element: ET.Element, key: str, factor: float) -> None:
        if key in element.attrib:
            values = np.fromstring(element.attrib[key], sep=" ") * factor
            element.set(key, " ".join(format(value, ".17g") for value in values))

    for element in body.iter():
        for key in ("pos", "fromto"):
            multiply(element, key, scale)
        if element.tag in ("geom", "site"):
            for key in ("size", "margin", "gap"):
                multiply(element, key, scale)
        if element.tag == "joint":
            joint = native.joint(element.attrib["name"])
            if joint.type[0] == mujoco.mjtJoint.mjJNT_SLIDE:
                for key in ("range", "ref", "springref", "margin"):
                    multiply(element, key, scale)
            elif joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE:
                # Write resolved values so inherited defaults also scale correctly.
                dof = int(joint.dofadr[0])
                element.set("armature", str(native.dof_armature[dof] * scale**2))
                element.set("damping", str(native.dof_damping[dof] * scale**2))
    for default in root.findall("default"):
        for element in default.iter():
            if element.tag in ("geom", "site"):
                multiply(element, "size", scale)
            if element.tag == "tendon":
                multiply(element, "width", scale)
    meshes = {element.get("mesh") for element in body.iter("geom")}
    for mesh in root.findall("asset/mesh"):
        if mesh.get("name") in meshes:
            mesh.set("scale", mesh.get("scale", "1 1 1"))
            multiply(mesh, "scale", scale)
    for tendon in root.findall("tendon/spatial"):
        for key in ("springlength", "range", "width", "margin"):
            multiply(tendon, key, scale)
    for actuator in root.findall("actuator/general"):
        multiply(actuator, "lengthrange", scale)
    for equality in root.findall("equality/joint"):
        dep = native.joint(equality.attrib["joint1"])
        driver_name = equality.get("joint2")
        dep_scale = scale if dep.type[0] == mujoco.mjtJoint.mjJNT_SLIDE else 1.0
        driver_scale = (
            scale
            if driver_name
            and native.joint(driver_name).type[0] == mujoco.mjtJoint.mjJNT_SLIDE
            else 1.0
        )
        coeff = np.fromstring(equality.get("polycoef", "0 1 0 0 0"), sep=" ")
        coeff *= dep_scale / driver_scale ** np.arange(len(coeff))
        equality.set("polycoef", " ".join(format(value, ".12g") for value in coeff))


def _model_xml(params: SubjectParams, source_dir: str, scale: float = 1.0,
               model_name: str = "myolegs26", lock_upper_body: bool = False,
               ankle_dorsiflexion_deg: float | None = None,
               flat_ground: bool = False, articulated_torso: bool = False) -> str:
    source = Path(source_dir)
    if not source.is_absolute():
        source = ROOT / source
    if not (source / "myo_sim/build/compose.py").is_file():
        raise ValueError(f"MyoSim clone missing at {source}; set human.myo.source_dir")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from myo_sim.build.compose import build_spec, sanitize_spec_xml

    if model_name not in ("myolegs26", "myolegs", "myofullbody"):
        raise ValueError(f"Unsupported MyoSim model: {model_name}")
    if model_name == "myofullbody" and not lock_upper_body:
        raise ValueError("The leg-only controller requires human.myo.lock_upper_body=true for myofullbody")
    spec = build_spec(model_name)
    native = spec.compile()
    if model_name == "myolegs26" and not np.isclose(params.mass, native.body_mass.sum(), atol=0.01):
        raise ValueError(
            f"MyoLeg26 has fixed anatomy: subject.mass must be {native.body_mass.sum():.6f}"
        )
    if not np.isclose(params.height, 1.75):
        raise ValueError(
            "Use human.myo.scale to resize MyoLeg26; subject.height=1.75 is metadata only"
        )
    root = ET.fromstring(sanitize_spec_xml(spec.to_xml(), model=native))
    _scale_anatomy(root, native, scale)
    if ankle_dorsiflexion_deg is not None:
        if not np.isfinite(ankle_dorsiflexion_deg) or ankle_dorsiflexion_deg <= 0:
            raise ValueError("human.myo.ankle_dorsiflexion_deg must be finite and positive")
        for side in ("l", "r"):
            name = f"ankle_angle_{side}"
            joint = root.find(f".//worldbody//joint[@name='{name}']")
            assert joint is not None
            lower = float(native.joint(name).range[0])
            joint.set("range", f"{lower:.17g} {np.radians(ankle_dorsiflexion_deg):.17g}")
    if model_name != "myolegs26":
        # New model variants use the configured total human mass, distributed
        # according to native body masses. Backpack and exo are added afterward.
        mass_ratio = params.mass / float(native.body_mass.sum())
        if not np.isfinite(mass_ratio) or mass_ratio <= 0:
            raise ValueError("subject.mass must be finite and positive")
        for inertial in root.findall(".//body/inertial"):
            for key in ("mass", "diaginertia"):
                values = np.fromstring(inertial.attrib[key], sep=" ") * mass_ratio
                if key == "diaginertia":
                    largest = int(np.argmax(values))
                    values[largest] = min(values[largest],
                                          np.delete(values, largest).sum() * (1 - 1e-12))
                inertial.set(key, " ".join(format(v, ".17g") for v in values))
    root.set("model", f"{model_name}_exo_planar")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(params.timestep))
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    assert world is not None
    body = world.find("body[@name='Full Body']")
    assert body is not None
    free = body.find("joint[@type='free']")
    assert free is not None
    body.remove(free)
    world.remove(body)
    frame = ET.SubElement(world, "body", name="reference_root")
    for name, axis, kind in zip(
        ROOT_JOINTS, ("1 0 0", "0 0 1", "0 1 0"), ("slide", "slide", "hinge")
    ):
        ET.SubElement(
            frame,
            "joint",
            name=name,
            type=kind,
            axis=axis,
            limited="false",
            damping="0",
        )
    # Native model faces -Y. Rotate it to +X, the reference and exo convention.
    body.set("pos", "0 0 0")
    body.set("quat", "0.7071067811865476 0 0 0.7071067811865476")
    frame.append(body)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)  # native free-root qpos layout no longer applies
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for side in ("l", "r"):
        for joint in ("hip_adduction", "hip_rotation"):
            ET.SubElement(
                equality,
                "joint",
                name=f"planar_{joint}_{side}",
                joint1=f"{joint}_{side}",
                polycoef="0 0 0 0 0",
            )
    bodies = {element.get("name"): element for element in world.iter("body")}
    if articulated_torso:
        if model_name != "myolegs26":
            raise ValueError("Independent lumbar support currently requires myolegs26")
        ET.SubElement(bodies["lumbar5"], "joint", name="exo_rl_lumbar", type="hinge",
                      axis="0 0 -1", limited="true", range=f"{np.radians(-20)} {np.radians(60)}",
                      damping="4", armature="0.02")
    if lock_upper_body:
        leg_joints = {joint.get("name") for joint in bodies["pelvis"].iter("joint")}
        dependent_joints = {joint.get("joint1") for joint in equality.findall("joint")}
        for joint in body.iter("joint"):
            name = joint.attrib["name"]
            if name in leg_joints or name in dependent_joints:
                continue
            joint_id = native.joint(name).id
            neutral = float(native.qpos0[native.jnt_qposadr[joint_id]])
            if native.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE:
                neutral *= scale
            ET.SubElement(equality, "joint", name=f"fixed_upper_{name}",
                          joint1=name, polycoef=f"{neutral} 0 0 0 0")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for name in IMU_SITES:
        parent = "pelvis" if name == "pelvis" else name.replace("thigh", "femur")
        ET.SubElement(
            bodies[parent], "site", name=f"imu_{name}", quat=MOUNT_QUAT, size="0.01"
        )
        ET.SubElement(sensor, "gyro", name=f"gyro_{name}", site=f"imu_{name}")
        ET.SubElement(sensor, "accelerometer", name=f"acc_{name}", site=f"imu_{name}")
    for side in ("l", "r"):
        for label, x in (("heel", "0"), ("toe", "0.18")):
            ET.SubElement(
                bodies[f"calcn_{side}"],
                "site",
                name=f"exo_rl_sole_{label}_{side}",
                pos=f"{float(x) * scale} {-0.03 * scale} 0",
                size="0.003",
            )
    if params.payload_mass > 0:
        mount = ET.SubElement(
            bodies["torso"], "body", name="payload_mount", quat=MOUNT_QUAT
        )
        payload = ET.SubElement(
            mount, "body", name="payload", pos=" ".join(map(str, params.payload_offset))
        )
        ET.SubElement(
            payload,
            "geom",
            type="box",
            size="0.08 0.15 0.20",
            mass=str(params.payload_mass),
            contype="0",
            conaffinity="0",
            rgba=".9 .2 .2 1",
        )
    if params.exo is not None:
        thigh_len = float(np.linalg.norm(native.body("tibia_l").pos)) * scale
        assets, snippets = cast(ExoHardware, params.exo).mjcf_snippets(
            thigh_len, params.exo_mass_scale, params.exo_quality
        )
        asset = root.find("asset")
        assert asset is not None
        asset.extend(ET.fromstring(f"<asset>{assets}</asset>"))
        for name, snippet in snippets.items():
            parent = name.replace("thigh", "femur").replace("shank", "tibia")
            mount = ET.SubElement(
                bodies[parent], "body", name=f"exo_mount_{name}", quat=MOUNT_QUAT
            )
            if name == "pelvis":
                # CAD pelvis parts are relative to the hip-axis midpoint.
                # Myo's pelvis origin is above and in front of those axes.
                hip_midpoint = scale * (
                    native.body("femur_l").pos + native.body("femur_r").pos
                ) / 2.0
                mount.set("pos", " ".join(map(str, hip_midpoint)))
            mount.extend(ET.fromstring(f"<body>{snippet}</body>"))
    floor = world.find("geom[@name='floor']")
    if floor is not None:
        floor.set("friction", f"{params.floor_friction} 0.005 0.0001")
        if flat_ground:
            # The native scene includes a finite pedestal and its decorative
            # mesh. Traveling episodes need one continuous surface at foot level.
            for geom in list(world.findall("geom")):
                if geom is not floor:
                    world.remove(geom)
            floor.set("pos", "0 0 0")
            floor.set("size", "0 0 0.1")
    return ET.tostring(root, encoding="unicode")


class MyoLegModel(HumanExoModel):
    is_muscle_model = True

    def __init__(self, params: SubjectParams, source_dir: str, scale: float = 1.0,
                 model_name: str = "myolegs26", lock_upper_body: bool = False,
                 ankle_dorsiflexion_deg: float | None = None,
                 flat_ground: bool = False, torso_control: dict | None = None):
        self.params = params
        self.torso_control = torso_control or {}
        self.articulated_torso = bool(self.torso_control.get("enabled", False))
        if self.articulated_torso:
            for name, default in [("kp", 600.), ("kd", 60.), ("max_torque_Nm", 160.)]:
                value = float(self.torso_control.get(name, default))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"human.myo.torso_control.{name} must be finite and positive")
            share = float(self.torso_control.get("pelvis_share", 0.))
            if not np.isfinite(share) or not 0 <= share <= 1:
                raise ValueError("human.myo.torso_control.pelvis_share must be in [0, 1]")
        # MyoLeg26 flexes the knee at negative native angles; standard MyoLeg
        # (also used by full-body) uses positive knee flexion and reversed axes.
        self.signs = SIGNS.copy() if model_name == "myolegs26" else np.ones(6)
        self.xml = _model_xml(params, source_dir, scale, model_name, lock_upper_body,
                              ankle_dorsiflexion_deg, flat_ground, self.articulated_torso)
        assets = (
            cast(ExoHardware, params.exo).meshes(params.exo_quality)
            if params.exo is not None
            else None
        )
        self.model = mujoco.MjModel.from_xml_string(self.xml, assets)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.human_qpos = np.array([int(m.joint(name).qposadr[0]) for name in JOINTS])
        self.human_dof = np.array([int(m.joint(name).dofadr[0]) for name in JOINTS])
        self.exo_qpos, self.exo_dof = (
            self.human_qpos[EXO_INDICES],
            self.human_dof[EXO_INDICES],
        )
        self.root_qpos = np.array(
            [int(m.joint(name).qposadr[0]) for name in ROOT_JOINTS]
        )
        self.root_dof = np.array([int(m.joint(name).dofadr[0]) for name in ROOT_JOINTS])
        self.sole_sites = [
            m.site(f"exo_rl_sole_{point}_{side}").id
            for side in ("l", "r")
            for point in ("heel", "toe")
        ]
        self.toe_qpos = np.array([int(m.joint(f"mtp_angle_{s}").qposadr[0]) for s in ("l", "r")])
        self.toe_dof = np.array([int(m.joint(f"mtp_angle_{s}").dofadr[0]) for s in ("l", "r")])
        self.foot_bodies = [
            {m.body(f"{part}_{s}").id for part in ("talus", "calcn", "toes")}
            for s in ("l", "r")
        ]
        self.couplings = [
            i for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
        ]
        self.moment = np.zeros((m.nu, m.nv))
        self.lumbar_qpos = self.lumbar_dof = None
        if self.articulated_torso:
            joint = m.joint("exo_rl_lumbar")
            self.lumbar_qpos = int(joint.qposadr[0])
            self.lumbar_dof = int(joint.dofadr[0])
            self.lumbar_angles = np.linspace(*joint.range, 161)
            head = m.body("head").id
            hips = [m.joint(f"hip_flexion_{s}").id for s in ("l", "r")]
            self.torso_angles = np.empty_like(self.lumbar_angles)
            saved = d.qpos.copy()
            for i, angle in enumerate(self.lumbar_angles):
                d.qpos[self.root_qpos[2]] = 0.
                d.qpos[self.lumbar_qpos] = angle
                mujoco.mj_kinematics(m, d)
                direction = d.xipos[head] - d.xanchor[hips].mean(0)
                self.torso_angles[i] = np.arctan2(direction[0], direction[2])
            if np.any(np.diff(self.torso_angles) <= 0):
                raise ValueError("Lumbar calibration must be monotonic")
            d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        # Derive the feedforward segment data from the imported anatomy, not Winter ratios.
        thigh = m.body("femur_l")
        shank_ids = [
            m.body(name).id for name in ("tibia_l", "talus_l", "calcn_l", "toes_l")
        ]
        sm = float(m.body_mass[shank_ids].sum())
        knee = d.body("tibia_l").xpos
        com = np.average(d.xipos[shank_ids], axis=0, weights=m.body_mass[shank_ids])
        tl = float(np.linalg.norm(m.body("tibia_l").pos))
        sl = float(np.linalg.norm(m.body("talus_l").pos))
        self.segments = SegmentInfo(
            float(thigh.mass[0]),
            tl,
            float(-thigh.ipos[1]),
            sm,
            sl,
            float(knee[2] - com[2]),
            tl + sl + 0.06 * scale,
        )

    def pelvis_reference(self, torso: float) -> float:
        if not self.articulated_torso:
            return torso
        return (np.radians(float(self.torso_control.get("pelvis_pitch_deg", 0.)))
                + float(self.torso_control.get("pelvis_share", 0.)) * torso)

    def lumbar_reference(self, torso: float, pelvis: float) -> float:
        return float(np.interp(torso - pelvis, self.torso_angles, self.lumbar_angles))

    def posture(self) -> dict[str, float]:
        pelvis = float(self.data.qpos[self.root_qpos[2]])
        if not self.articulated_torso:
            return dict(pelvis=pelvis, lumbar=0., torso=pelvis)
        lumbar = float(self.data.qpos[self.lumbar_qpos])
        torso = pelvis + float(np.interp(lumbar, self.lumbar_angles, self.torso_angles))
        return dict(pelvis=pelvis, lumbar=lumbar, torso=torso)

    def coordinate_map(self) -> np.ndarray:
        """Map six flexion velocities to native and equality-coupled coordinates.

        The dependent moving attachment points contribute to effective muscle
        moment arms; ignoring them would misrepresent knee muscle torque.
        """
        m, d = self.model, self.data
        mapping = np.zeros((m.nv, 6))
        mapping[self.human_dof, np.arange(6)] = self.signs
        for i in self.couplings:
            dependent, driver = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            if driver < 0:
                continue
            x = d.qpos[m.jnt_qposadr[driver]]
            derivative = np.polynomial.polynomial.polyval(
                x, np.arange(1, 5) * m.eq_data[i, 1:5]
            )
            mapping[m.jnt_dofadr[dependent]] = (
                derivative * mapping[m.jnt_dofadr[driver]]
            )
        return mapping

    def _set_configuration(self, q_human: np.ndarray, trunk: float) -> None:
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        d.qpos[self.human_qpos] = self.signs * q_human
        d.qpos[self.root_qpos[2]] = self.pelvis_reference(trunk)
        if self.articulated_torso:
            d.qpos[self.lumbar_qpos] = self.lumbar_reference(trunk, d.qpos[self.root_qpos[2]])
        for i in self.couplings:
            dependent, driver = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            x = d.qpos[m.jnt_qposadr[driver]] if driver >= 0 else 0.0
            d.qpos[m.jnt_qposadr[dependent]] = np.polynomial.polynomial.polyval(
                x, m.eq_data[i, :5]
            )

    def set_pose(
        self,
        q_human: np.ndarray,
        trunk_pitch: float,
        qd_human=None,
        root_vx: float = 0.0,
        clearance: float = 0.0,
        trunk_velocity: float | None = None,
    ) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._set_configuration(q_human, trunk_pitch)
        mujoco.mj_kinematics(m, d)
        d.qpos[self.root_qpos[1]] = (
            -min(d.site_xpos[i, 2] for i in self.sole_sites) + clearance
        )
        if qd_human is not None:
            d.qvel[:] = self.coordinate_map() @ qd_human
        d.qvel[self.root_dof[0]] = root_vx
        if trunk_velocity is not None:
            # Match the moving reference while keeping the lowest sole's
            # vertical velocity zero. Zero root velocities at a mid-squat
            # reset otherwise launch an artificial contact transient.
            share = float(self.torso_control.get("pelvis_share", 0.)) if self.articulated_torso else 1.
            d.qvel[self.root_dof[2]] = share * trunk_velocity
            if self.articulated_torso:
                slope = float(np.interp(d.qpos[self.lumbar_qpos], self.lumbar_angles,
                                       np.gradient(self.torso_angles, self.lumbar_angles)))
                d.qvel[self.lumbar_dof] = (1 - share) * trunk_velocity / slope
            mujoco.mj_forward(m, d)
            site = self.sole_sites[int(np.argmin(d.site_xpos[self.sole_sites, 2]))]
            jacobian = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacobian, None, site)
            d.qvel[self.root_dof[1]] -= float(jacobian[2] @ d.qvel)
        mujoco.mj_forward(m, d)

    def implied_speed(self, motion, n: int = 200) -> float:
        from .reference import TRAVELING_MOTIONS

        if motion.name not in TRAVELING_MOTIONS:
            return 0.0
        m, d = self.model, self.data
        saved = d.qpos.copy()
        xs, ids = [], []
        for t in np.arange(n) * motion.period / n:
            q, trunk = motion.angles(t)
            self._set_configuration(q, trunk)
            mujoco.mj_kinematics(m, d)
            k = int(np.argmin(d.site_xpos[self.sole_sites, 2]))
            ids.append(k)
            xs.append(d.site_xpos[self.sole_sites[k], 0])
        velocities = [
            -(xs[(i + 1) % n] - xs[i]) / (motion.period / n)
            for i in range(n)
            if ids[i] == ids[(i + 1) % n]
        ]
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return float(np.mean(velocities)) if velocities else 0.0

    def sensors(self) -> dict:
        sensors = super().sensors()
        sensors["q_exo"] *= self.signs[EXO_INDICES]
        sensors["qd_exo"] *= self.signs[EXO_INDICES]
        return sensors

    def joint_positions(self) -> np.ndarray:
        return self.signs * self.data.qpos[self.human_qpos]


class MuscleController(HumanController):
    def reset(self) -> None:
        super().reset()
        self.pitch_integral = 0.0
        self.foot_pitch_torque = np.zeros(2)
        self.toe_torque = np.zeros(2)
        self.last_recruitment_diagnostics: dict[str, float] = {}
        self.lumbar_torque = 0.

    def __init__(self, cfg: dict, mass: float):
        super().__init__(cfg, mass)
        self.regularization = float(cfg["myo"]["activation_regularization"])
        self.foot_control = cfg["myo"].get("foot_control", {})
        self.standing_control = cfg["myo"].get("standing_control", {})
        self.diagnostics_enabled = False
        if self.regularization <= 0:
            raise ValueError("human.myo.activation_regularization must be positive")

    def compute(
        self,
        hm: HumanExoModel,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        qdd_ref: np.ndarray,
        trunk_ref: float,
        trunk_d_ref: float,
        v_ref: float,
        tau_exo_human_order: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(hm, MyoLegModel):
            raise TypeError("MuscleController requires MyoLegModel")
        m, d = hm.model, hm.data
        mapping = hm.coordinate_map()
        current_q = hm.joint_positions()
        error = q_ref - current_q
        velocity_error = qd_ref - hm.signs * d.qvel[hm.human_dof]
        foot = self.foot_control
        standing = self.standing_control
        stiffness = 1.0
        if standing.get("enabled", False):
            # Soften only once both the reference and actual knees approach
            # standing. Retain damping and gravity compensation throughout.
            knee_angle = max(float(np.max(np.abs(q_ref[[1, 4]]))),
                             float(np.max(np.abs(current_q[[1, 4]]))))
            transition = np.radians(float(standing.get("transition_deg", 15.0)))
            depth = float(np.clip(knee_angle / max(transition, 1e-6), 0.0, 1.0))
            depth = depth * depth * (3.0 - 2.0 * depth)
            minimum = float(standing.get("stiffness_scale", 0.4))
            stiffness = minimum + (1.0 - minimum) * depth
            leak = float(standing.get("integral_release_seconds", 0.5))
            self.integral *= np.exp(-(1.0 - depth) * dt / max(leak, 1e-6))
        self.integral = np.clip(
            self.integral + stiffness * error * dt,
            -self.i_limit / np.maximum(self.ki, 1e-9),
            self.i_limit / np.maximum(self.ki, 1e-9),
        )
        inertial = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, inertial, mapping @ qdd_ref)
        self.tau_needed = (
            mapping.T @ (inertial + d.qfrc_bias - d.qfrc_passive)
            + stiffness * self.kp * error
            + self.kd * velocity_error
            + self.ki * self.integral
        )
        if foot.get("enabled", False):
            # For the squat, resist heel/forefoot rocking only on a grounded foot.
            # Positive heel-to-forefoot pitch means the forefoot is lifting;
            # plantarflexion torque opposes it. No contact constraints are added.
            grounded = [False, False]
            for contact in d.contact[:d.ncon]:
                b1, b2 = (int(m.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                for side, bodies in enumerate(hm.foot_bodies):
                    grounded[side] |= (b1 in bodies and b2 == 0) or (b2 in bodies and b1 == 0)
            for side, ankle in enumerate((2, 5)):
                heel, forefoot = hm.sole_sites[2 * side:2 * side + 2]
                direction = d.site_xpos[forefoot] - d.site_xpos[heel]
                pitch = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
                tolerance = np.radians(float(foot.get("pitch_deadband_deg", 1.0)))
                pitch = np.sign(pitch) * max(abs(pitch) - tolerance, 0.0)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, forefoot, velocity, 0)
                correction = (-stiffness * float(foot["pitch_kp"]) * pitch
                              + float(foot["pitch_kd"]) * velocity[1])
                limit = float(foot["pitch_torque_limit"])
                target = np.clip(correction, -limit, limit) if grounded[side] else 0.0
                # Contact can alternate on successive physics steps. Fade the
                # correction in/out instead of switching the requested torque.
                alpha = dt / (float(foot.get("smoothing_seconds", 0.08)) + dt)
                max_delta = float(foot.get("torque_rate_limit", 100.0)) * dt
                delta = alpha * (target - self.foot_pitch_torque[side])
                self.foot_pitch_torque[side] += np.clip(delta, -max_delta, max_delta)
                self.tau_needed[ankle] += self.foot_pitch_torque[side]
        desired = np.clip(
            self.tau_needed - self.beta * tau_exo_human_order,
            -self.tau_max,
            self.tau_max,
        )
        mujoco.mju_sparse2dense(
            hm.moment,
            d.actuator_moment,
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind,
        )
        moments = mapping.T @ hm.moment.T
        torque_scale = self.tau_max
        if foot.get("enabled", False):
            # Recruit the same muscles for toe posture as well as leg torque.
            # Native zero MTP angle is the neutral toe posture, not a floor pin.
            toe_limit = float(foot["toe_torque_limit"])
            if toe_limit <= 0:
                raise ValueError("human.myo.foot_control.toe_torque_limit must be positive")
            toe_torque = (d.qfrc_bias[hm.toe_dof] - d.qfrc_passive[hm.toe_dof]
                          - stiffness * float(foot["toe_kp"]) * d.qpos[hm.toe_qpos]
                          - float(foot["toe_kd"]) * d.qvel[hm.toe_dof])
            toe_target = np.clip(toe_torque, -toe_limit, toe_limit)
            alpha = dt / (float(foot.get("smoothing_seconds", 0.08)) + dt)
            max_delta = float(foot.get("torque_rate_limit", 100.0)) * dt
            self.toe_torque += np.clip(alpha * (toe_target - self.toe_torque),
                                       -max_delta, max_delta)
            desired = np.r_[desired, self.toe_torque]
            moments = np.vstack([moments, hm.moment[:, hm.toe_dof].T])
            torque_scale = np.r_[self.tau_max, [toe_limit, toe_limit]]
        gain = np.array(
            [
                mujoco.mju_muscleGain(
                    d.actuator_length[i],
                    d.actuator_velocity[i],
                    m.actuator_lengthrange[i],
                    m.actuator_acc0[i],
                    m.actuator_gainprm[i, :9],
                )
                for i in range(m.nu)
            ]
        )
        bias = np.array(
            [
                mujoco.mju_muscleBias(
                    d.actuator_length[i],
                    m.actuator_lengthrange[i],
                    m.actuator_acc0[i],
                    m.actuator_biasprm[i, :9],
                )
                for i in range(m.nu)
            ]
        )
        matrix = moments * gain / torque_scale[:, None]
        target = (desired - moments @ bias) / torque_scale
        solution = lsq_linear(
            np.vstack([matrix, np.sqrt(self.regularization) * np.eye(m.nu)]),
            np.r_[target, np.zeros(m.nu)],
            bounds=(0.0, 1.0),
            method="bvls",
        )
        if not solution.success:
            raise RuntimeError(f"Muscle recruitment failed: {solution.message}")
        if self.diagnostics_enabled:
            recruited = moments @ (gain * solution.x + bias)
            current = moments @ (gain * d.act + bias)
            self.last_recruitment_diagnostics = {
                "recruitment_fit_rms_fraction": float(np.sqrt(np.mean(((recruited - desired) / torque_scale) ** 2))),
                "activation_lag_rms_fraction": float(np.sqrt(np.mean(((current - recruited) / torque_scale) ** 2))),
                "excitation_saturation_fraction": float(np.mean(solution.x > .99)),
            }
        d.ctrl[:] = solution.x  # MuJoCo integrates muscle activation dynamics.
        r = self.residual
        pitch = d.qpos[hm.root_qpos[2]]
        pelvis_ref = hm.pelvis_reference(trunk_ref)
        share = float(hm.torso_control.get("pelvis_share", 0.)) if hm.articulated_torso else 1.
        pitch_error = pelvis_ref - pitch
        self.lumbar_torque = 0.
        if hm.articulated_torso:
            control = hm.torso_control
            target = hm.lumbar_reference(trunk_ref, pitch)
            self.lumbar_torque = float(np.clip(
                d.qfrc_bias[hm.lumbar_dof] - d.qfrc_passive[hm.lumbar_dof]
                + float(control.get("kp", 600.)) * (target - d.qpos[hm.lumbar_qpos])
                - float(control.get("kd", 60.)) * d.qvel[hm.lumbar_dof],
                -float(control.get("max_torque_Nm", 160.)), float(control.get("max_torque_Nm", 160.))))
        # The loaded torso needs a sustained balancing moment. PD alone leaves
        # a posture offset; bound the integral so the helper cannot wind up.
        pitch_limit = float(r.get("pitch_integral_limit", 0.0))
        self.pitch_integral = float(np.clip(
            self.pitch_integral + float(r.get("pitch_ki", 0.0)) * pitch_error * dt,
            -pitch_limit, pitch_limit,
        ))
        residual = self.mass_scale * np.array(
            [
                r["x_kv"] * (v_ref - d.qvel[hm.root_dof[0]]),
                0.0,
                r["pitch_kp"] * pitch_error
                + r["pitch_kd"] * (share * trunk_d_ref - d.qvel[hm.root_dof[2]])
                + self.pitch_integral,
            ]
        )
        return mapping.T @ d.qfrc_actuator, residual
