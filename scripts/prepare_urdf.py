"""Turn a SolidWorks URDF export into a MuJoCo-ready exo asset folder.

    python scripts/prepare_urdf.py MkI_FA900_Full_Body_Assembly assets/exo_mk1_fa900

Reads <export>/urdf/<name>.urdf and <export>/meshes/*.STL and writes to the output folder:
    <name>.urdf     mesh paths rewritten to meshes_hi/ (relative), MuJoCo compiler block added
    meshes_hi/      meshes decimated to <= --hi-faces triangles (viewing)
    meshes_lo/      meshes decimated to <= --lo-faces triangles (training)
Decimation is vertex clustering on a grid: coarse, but the exo meshes are visual only (mass and
inertia come from the URDF <inertial> tags, and the exo has no collision geometry).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo_rl.sim.exo_hardware import read_stl, write_stl_bytes  # noqa: E402


def cluster(verts: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that fall in the same grid cell; drop collapsed and duplicate triangles."""
    keys = np.floor(verts / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    merged = np.zeros((inverse.max() + 1, 3))
    np.add.at(merged, inverse, verts)
    merged /= np.bincount(inverse)[:, None]
    faces = inverse.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    faces = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]
    return merged, inverse.reshape(-1, 3)[np.sort(faces)]


def decimate(verts: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Smallest grid cell (binary search) that brings the mesh to <= max_faces triangles."""
    n = len(verts) // 3
    if n <= max_faces:
        return verts, np.arange(len(verts)).reshape(-1, 3)
    lo, hi = 1e-6, float(np.ptp(verts, axis=0).max())
    best = cluster(verts, hi)
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        v, f = cluster(verts, mid)
        if len(f) <= max_faces:
            hi, best = mid, (v, f)
        else:
            lo = mid
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="SolidWorks export folder (contains urdf/ and meshes/)")
    ap.add_argument("out", help="output folder, e.g. assets/exo_mk1_fa900")
    ap.add_argument("--hi-faces", type=int, default=20000)
    ap.add_argument("--lo-faces", type=int, default=2000)
    args = ap.parse_args()

    urdfs = [f for f in os.listdir(os.path.join(args.export, "urdf")) if f.endswith(".urdf")]
    if len(urdfs) != 1:
        raise SystemExit(f"expected one .urdf in {args.export}/urdf, found {urdfs}")
    text = open(os.path.join(args.export, "urdf", urdfs[0]), encoding="utf-8").read()
    text = re.sub(r'filename="package://[^"]*/meshes/([^"]+)"', r'filename="meshes_hi/\1"', text)
    text = re.sub(r"(<robot\s[^>]*>)", r'\1\n  <mujoco>\n    <compiler discardvisual="false" fusestatic="false"/>\n'
                  r"  </mujoco>", text, count=1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, urdfs[0]), "w", encoding="utf-8") as f:
        f.write(text)

    meshes = sorted(m for m in os.listdir(os.path.join(args.export, "meshes")) if m.lower().endswith(".stl"))
    for quality, limit in (("hi", args.hi_faces), ("lo", args.lo_faces)):
        os.makedirs(os.path.join(args.out, f"meshes_{quality}"), exist_ok=True)
        for name in meshes:
            verts, _ = read_stl(os.path.join(args.export, "meshes", name))
            v, f = decimate(verts, limit)
            with open(os.path.join(args.out, f"meshes_{quality}", name), "wb") as fh:
                fh.write(write_stl_bytes(v, f))
            print(f"{quality:2s} {name:28s} {len(verts) // 3:7d} -> {len(f):6d} faces")
    print(f"wrote {args.out}/{urdfs[0]}")


if __name__ == "__main__":
    main()
