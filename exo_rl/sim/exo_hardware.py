"""Mount a CAD exoskeleton (URDF) on the human model.

Each exo link becomes a rigid child body of the human segment it moves with:
    pelvis group -> pelvis     thigh links -> thigh_l/r     shank links -> shank_l/r
with its CAD mass, center of mass, full inertia tensor, and mesh. The exo hip flexion axis
is placed on the human hip joint and (with knee_mode: align_knee) the exo knee axis on the
human knee. Exo torque is still applied at the human hip/knee joints, i.e. the coupling is
rigid (see README, "Known limitations").

Frames: the URDF is loaded in MuJoCo at its zero pose, or at the spec's cad_pose_deg (joint
angles that put the CAD in the worn pose: legs straight, hip flexion axes sideways). Every point is
re-expressed in the human frame (x forward, y pilot's left, z up) relative to an anchor:
    pelvis parts: midpoint of the two exo hip-flexion axes  -> pelvis origin (hip height)
    thigh parts:  that side's exo hip-flexion axis          -> human hip
    shank parts:  that side's exo knee axis                 -> human knee
Meshes are re-baked into those frames once and passed to MuJoCo in memory.

Motor signs (+ human flexion vs + motor rotation) are derived from the URDF joint axes, so
wiring mistakes show up as a failing test rather than an exo pushing the wrong way.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import mujoco
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUMAN_HIP_Y = 0.09          # must match model_builder leg("l", 0.09)
HUMAN_HIP_AXIS = np.array([0.0, -1.0, 0.0])    # + = flexion (model_builder hip axis)
HUMAN_KNEE_AXIS = np.array([0.0, 1.0, 0.0])    # + = flexion (model_builder knee axis)


def _axis(s: str) -> np.ndarray:
    v = np.zeros(3)
    v["xyz".index(s[-1])] = -1.0 if s.startswith("-") else 1.0
    return v


def read_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary STL reader -> (vertices[n,3], faces[m,3])."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * n != len(data):
        raise ValueError(f"{path}: not a binary STL (convert ASCII STL to binary first)")
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)
    verts = rec["v"].reshape(-1, 3).astype(np.float64)
    return verts, np.arange(len(verts)).reshape(-1, 3)


def write_stl_bytes(verts: np.ndarray, faces: np.ndarray) -> bytes:
    tri = verts[faces].astype(np.float32)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tri
    return b"\0" * 80 + struct.pack("<I", len(faces)) + rec.tobytes()


@dataclass
class ExoPart:
    name: str
    parent: str                 # human body: pelvis | thigh_l | thigh_r | shank_l | shank_r
    group: str                  # pelvis | thigh | shank
    side: str                   # l | r | "" (pelvis)
    mass: float
    com: np.ndarray             # in anchor frame (human axes)
    inertia: np.ndarray         # 3x3 about COM, human axes
    mesh_key: str
    anchor_lateral: float       # anchor y relative to the human body origin
    extras: dict = field(default_factory=dict)


class ExoHardware:
    def __init__(self, spec_path: str):
        spec_path = spec_path if os.path.isabs(spec_path) else os.path.join(REPO_ROOT, spec_path)
        with open(spec_path) as f:
            self.spec = yaml.safe_load(f)
        self.spec_path = spec_path
        urdf = os.path.join(REPO_ROOT, self.spec["urdf"])
        self.model = mujoco.MjModel.from_xml_path(urdf)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        for joint, deg in (self.spec.get("cad_pose_deg") or {}).items():   # pose the CAD as worn
            d.qpos[m.jnt_qposadr[m.joint(joint).id]] = np.radians(deg)
        mujoco.mj_forward(m, d)

        fwd, up = _axis(self.spec["cad_forward"]), _axis(self.spec["cad_up"])
        self.R = np.stack([fwd, np.cross(up, fwd), up])           # human = R @ cad

        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        legs = self.spec["legs"]
        self.hip_anchor = {s: d.xanchor[jid(legs[s]["hip_joint"])].copy() for s in "lr"}
        self.knee_anchor = {s: d.xanchor[jid(legs[s]["knee_joint"])].copy() for s in "lr"}
        self.mid = 0.5 * (self.hip_anchor["l"] + self.hip_anchor["r"])

        # Verify the left/right assignment against the geometry.
        for s, sign in (("l", 1), ("r", -1)):
            y = (self.R @ (self.hip_anchor[s] - self.mid))[1]
            if np.sign(y) != sign:
                raise ValueError(f"exo spec: legs.{s} chain is on the pilot's {'right' if y < 0 else 'left'}; "
                                 "swap the l/r chains (or fix cad_forward) in " + spec_path)

        self.thigh_length = float(np.mean([np.linalg.norm(self.knee_anchor[s] - self.hip_anchor[s]) for s in "lr"]))
        self.motor_signs, self.axis_tilt_deg = {}, {}
        for s in "lr":
            for j, ref in (("hip", HUMAN_HIP_AXIS), ("knee", HUMAN_KNEE_AXIS)):
                a = self.R @ d.xaxis[jid(legs[s][f"{j}_joint"])]
                dot = float(a @ ref)
                self.motor_signs[f"{j}_{s}"] = 1 if dot > 0 else -1
                self.axis_tilt_deg[f"{j}_{s}"] = float(np.degrees(np.arccos(min(abs(dot), 1.0))))

        self._parts = self._make_parts()
        self._mesh_cache: dict[str, dict] = {}

    def _link_world(self, link):
        b = self.bid(link)
        if b < 0:
            raise ValueError(f"link '{link}' not in URDF")
        d, m = self.data, self.model
        Ri = d.ximat[b].reshape(3, 3)
        return dict(pos=d.xpos[b].copy(), mat=d.xmat[b].reshape(3, 3).copy(), mass=float(m.body_mass[b]),
                    com=d.xipos[b].copy(), inertia=Ri @ np.diag(m.body_inertia[b]) @ Ri.T)

    def _make_parts(self):
        parts = []
        groups = [("pelvis", "", self.spec["pelvis_links"], self.mid, "pelvis")]
        for s in "lr":
            leg = self.spec["legs"][s]
            groups.append(("thigh", s, leg["thigh_links"], self.hip_anchor[s], f"thigh_{s}"))
            groups.append(("shank", s, leg["shank_links"], self.knee_anchor[s], f"shank_{s}"))
        for group, side, links, anchor, parent in groups:
            lateral = 0.0 if group == "pelvis" else (self.R @ (anchor - self.mid))[1] - (HUMAN_HIP_Y if side == "l" else -HUMAN_HIP_Y)
            for link in links:
                w = self._link_world(link)
                parts.append(ExoPart(
                    name=f"exo_{link}", parent=parent, group=group, side=side, mass=w["mass"],
                    com=self.R @ (w["com"] - anchor), inertia=self.R @ w["inertia"] @ self.R.T,
                    mesh_key=link, anchor_lateral=float(lateral),
                    extras={"link_pos": w["pos"], "link_mat": w["mat"], "anchor": anchor}))
        return parts

    @property
    def total_mass(self) -> float:
        return float(sum(p.mass for p in self._parts))

    def group_mass(self, group: str, side: str = "") -> float:
        return float(sum(p.mass for p in self._parts if p.group == group and (not side or p.side == side)))

    def meshes(self, quality: str = "lo") -> dict[str, bytes]:
        """Meshes re-baked into each part's anchor frame (human axes), keyed by asset filename."""
        if quality not in self._mesh_cache:
            mdir = os.path.join(REPO_ROOT, self.spec["meshes"][quality])
            out = {}
            for p in self._parts:
                v, f = read_stl(os.path.join(mdir, f"{p.mesh_key}.STL"))
                e = p.extras
                vw = v @ e["link_mat"].T + e["link_pos"]           # URDF visual origin is identity
                vh = (vw - e["anchor"]) @ self.R.T
                out[f"{p.name}.stl"] = write_stl_bytes(vh, f)
            self._mesh_cache[quality] = out
        return self._mesh_cache[quality]

    def body_offset(self, part: ExoPart, thigh_len_human: float, hip_half_width: float = HUMAN_HIP_Y) -> np.ndarray:
        """Anchor position in the parent human body's frame (hip_half_width: the human's hip-center offset
        from the midline; the exo pelvis is rigid, so the exo legs keep the CAD hip width)."""
        if part.group == "pelvis":
            return np.zeros(3)
        lateral = part.anchor_lateral + (HUMAN_HIP_Y - hip_half_width) * (1 if part.side == "l" else -1)
        if part.group == "shank" and self.spec.get("knee_mode", "align_knee") == "rigid":
            return np.array([0.0, lateral, thigh_len_human - self.thigh_length])
        return np.array([0.0, lateral, 0.0])

    def mjcf_snippets(self, thigh_len_human: float, mass_scale: float = 1.0, quality: str = "lo",
                      rgba: str = ".45 .55 .75 1", hip_half_width: float = HUMAN_HIP_Y) -> tuple[str, dict[str, str]]:
        """Returns (<asset> mesh lines, {human body name: xml of exo child bodies})."""
        assets = "\n".join(f'    <mesh name="{p.name}" file="{p.name}.stl" inertia="shell"/>' for p in self._parts)
        bodies: dict[str, str] = {}
        for p in self._parts:
            o = self.body_offset(p, thigh_len_human, hip_half_width)
            I = p.inertia * mass_scale
            full = f"{I[0, 0]:.6g} {I[1, 1]:.6g} {I[2, 2]:.6g} {I[0, 1]:.6g} {I[0, 2]:.6g} {I[1, 2]:.6g}"
            bodies.setdefault(p.parent, "")
            bodies[p.parent] += (
                f'\n        <body name="{p.name}" pos="{o[0]:.5f} {o[1]:.5f} {o[2]:.5f}">'
                f'<inertial pos="{p.com[0]:.5f} {p.com[1]:.5f} {p.com[2]:.5f}" mass="{p.mass * mass_scale:.5f}" fullinertia="{full}"/>'
                f'<geom type="mesh" mesh="{p.name}" contype="0" conaffinity="0" group="1" rgba="{rgba}"/></body>')
        return assets, bodies

    def limb_params(self, seg, mass_scale: float = 1.0) -> dict:
        """Leg inertial params (human + exo) for the gravity feedforward, left/right averaged."""
        def combine(m_h, c_h, grp):
            m_e = self.group_mass(grp, "l") * mass_scale
            parts = [p for p in self._parts if p.group == grp and p.side == "l"]
            c_e = -sum(p.mass * p.com[2] for p in parts) / max(sum(p.mass for p in parts), 1e-9)
            return m_h + m_e, (m_h * c_h + m_e * c_e) / (m_h + m_e)
        tm, tc = combine(seg.thigh_mass, seg.thigh_com, "thigh")
        sm, sc = combine(seg.shank_foot_mass, seg.shank_foot_com, "shank")
        return dict(thigh_mass=tm, thigh_len=seg.thigh_len, thigh_com=tc, shank_foot_mass=sm, shank_foot_com=sc)

    def motor_map(self) -> dict:
        zeros = self.spec.get("motor_zero_offsets", {})
        order = ["hip_l", "knee_l", "hip_r", "knee_r"]
        legs = self.spec["legs"]
        return {
            "order": order,
            "urdf_joints": [legs[j[-1]][f"{j[:-2]}_joint"] for j in order],
            "signs": [self.motor_signs[j] for j in order],
            "zero_offsets": [float(zeros.get(j, 0.0)) for j in order],
        }

    def report(self, thigh_len_human: float | None = None) -> str:
        lines = [f"Exo: {os.path.basename(self.spec['urdf'])}  total {self.total_mass:.2f} kg "
                 f"(pelvis group {self.group_mass('pelvis'):.2f}, per leg thigh {self.group_mass('thigh', 'l'):.2f} "
                 f"+ shank {self.group_mass('shank', 'l'):.2f})",
                 f"  exo hip->knee axis distance {self.thigh_length:.3f} m"]
        if thigh_len_human is not None:
            lines.append(f"  human thigh {thigh_len_human:.3f} m -> knee axis mismatch {100 * (self.thigh_length - thigh_len_human):+.1f} cm"
                         f" (knee_mode: {self.spec.get('knee_mode', 'align_knee')})")
        mm = self.motor_map()
        for j, u, s in zip(mm["order"], mm["urdf_joints"], mm["signs"]):
            lines.append(f"  {j:7s} <- URDF {u:24s} sign {s:+d}   axis tilt from sagittal normal {self.axis_tilt_deg[j]:.1f} deg")
        return "\n".join(lines)
