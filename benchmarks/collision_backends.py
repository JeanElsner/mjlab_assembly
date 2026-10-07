"""Compare collision representations for a peg-in-hole pair on MuJoCo Warp.

A peg on five compliant joints (x, y, z slides; tilt about x and y) is pressed
into a blind bore from a randomized lateral offset and tilt, in N parallel
worlds. Variants:

  sdf      peg and socket as MuJoCo octree SDFs built from the meshes
  convex   peg as one convex mesh, socket as an exact convex sector decomposition
  prim     peg as a primitive cylinder, socket as convex sectors

Reported per variant and world count: step time (CUDA-graph captured), the
fraction of worlds seated, NaN worlds, and the largest radial excursion of the
peg tip beyond the clearance (a penetration proxy).

  uv run python benchmarks/collision_backends.py --worlds 1024 4096
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mjlab_assembly.asset_zoo.workpieces import geometry as g

PEG_R = 0.006
PEG_L = 0.05
DEPTH = 0.025
FLOOR = 0.005
OUTER = 0.02
START_Z = 0.005  # tip above the mouth at the start
CAPTURE = False
NCON, NJ = 256, 1536
BROAD = None
FILT = None


def _add_mesh(spec: mujoco.MjSpec, name: str, m: g.Mesh, sdf: bool, depth: int) -> None:
  me = spec.add_mesh()
  me.name = name
  me.uservert = m.vertices.flatten()
  me.userface = m.faces.flatten()
  if sdf:
    me.needsdf = True
    me.octree_maxdepth = depth
  else:
    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL


def build(variant: str, clearance: float, chamfer: float, n_seg: int, oct_depth: int, peg_seg: int = 0) -> mujoco.MjModel:
  peg_seg = peg_seg or n_seg
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.sdf_iterations = 10
  spec.option.sdf_initpoints = 40
  rb = PEG_R + clearance
  # Socket, fixed to the world.
  if variant == "sdf":
    _add_mesh(spec, "socket", g.socket_mesh(rb, OUTER, DEPTH, FLOOR, chamfer, n_seg), True, oct_depth)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname="socket")
  else:
    for i, p in enumerate(g.socket_sectors(rb, OUTER, DEPTH, FLOOR, chamfer, n_seg)):
      _add_mesh(spec, f"sec{i}", p, False, 0)
      spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"sec{i}")
  # Peg on five compliant joints.
  b = spec.worldbody.add_body(name="peg", pos=[0, 0, START_Z])
  for name, jt, axis in (("x", "slide", [1, 0, 0]), ("y", "slide", [0, 1, 0]),
                         ("z", "slide", [0, 0, 1]), ("rx", "hinge", [1, 0, 0]),
                         ("ry", "hinge", [0, 1, 0])):
    j = b.add_joint(name=name, axis=axis,
                    type=mujoco.mjtJoint.mjJNT_SLIDE if jt == "slide" else mujoco.mjtJoint.mjJNT_HINGE)
    j.armature = 0.01 if jt == "slide" else 1e-4
    kp = 400.0 if jt == "slide" else 2.0
    a = spec.add_actuator(name=f"a_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gainprm[0] = kp
    a.biasprm[:3] = [0.0, -kp, -2.0 * np.sqrt(kp * (0.05 if jt == "slide" else 1e-4))]
  if variant == "sdf":
    _add_mesh(spec, "peg", g.peg_mesh(PEG_R, PEG_L, chamfer, n_seg), True, oct_depth)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname="peg", density=7800)
  elif variant == "convex":
    _add_mesh(spec, "peg", g.peg_mesh(PEG_R, PEG_L, chamfer, peg_seg), False, 0)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="peg", density=7800)
  else:
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PEG_R, PEG_L / 2, 0],
               pos=[0, 0, PEG_L / 2], density=7800)
  return spec.compile()


def run(variant: str, nworld: int, clearance: float, chamfer: float, n_seg: int,
        oct_depth: int, seconds: float, seed: int = 0, peg_seg: int = 0) -> dict:
  m = build(variant, clearance, chamfer, n_seg, oct_depth, peg_seg)
  d = mujoco.MjData(m)
  t0 = time.perf_counter()
  mw = mjw.put_model(m)
  if BROAD is not None:
    mw.opt.broadphase = BROAD
  if FILT is not None:
    mw.opt.broadphase_filter = FILT
  dw = mjw.put_data(m, d, nworld=nworld, nconmax=NCON, njmax=NJ)
  rng = np.random.default_rng(seed)
  q = np.zeros((nworld, m.nq), np.float32)
  # Lateral offset up to 1.5 clearances plus the chamfer, tilt up to 1 degree.
  rmax = 1.5 * clearance + chamfer
  r = rmax * np.sqrt(rng.uniform(size=nworld))
  phi = rng.uniform(0, 2 * np.pi, nworld)
  q[:, 0], q[:, 1] = r * np.cos(phi), r * np.sin(phi)
  q[:, 3:5] = np.deg2rad(rng.uniform(-1, 1, (nworld, 2)))
  ctrl = np.zeros((nworld, m.nu), np.float32)
  ctrl[:, 2] = -(DEPTH + START_Z + 0.01)  # press 10 mm past the floor: ~4 N at seat
  wp.copy(dw.qpos, wp.array(q, dtype=wp.float32))
  wp.copy(dw.ctrl, wp.array(ctrl, dtype=wp.float32))
  mjw.forward(mw, dw)
  mjw.step(mw, dw)  # compile every kernel before the graph capture
  wp.synchronize()
  graph = None
  if CAPTURE:
    with wp.ScopedCapture() as cap:
      mjw.step(mw, dw)
    graph = cap.graph
  setup = time.perf_counter() - t0
  steps = int(seconds / m.opt.timestep)
  wp.synchronize()
  t1 = time.perf_counter()
  for _ in range(steps):
    if graph is not None:
      wp.capture_launch(graph)
    else:
      mjw.step(mw, dw)
  wp.synchronize()
  dt = (time.perf_counter() - t1) / steps
  qf = dw.qpos.numpy()
  ncon = dw.nacon.numpy()[0] if hasattr(dw, "nacon") else -1
  nan = ~np.isfinite(qf).all(axis=1)
  ok = qf[~nan]
  # Tip depth below the mouth and the tip's radial offset (small-angle).
  depth = -(START_Z + ok[:, 2])
  tip_r = np.hypot(ok[:, 0], ok[:, 1])
  seated = depth > DEPTH - 0.001
  return dict(variant=variant, nworld=nworld, ms_per_step=dt * 1e3,
              us_per_world_step=dt * 1e6 / nworld, setup_s=setup,
              seated=float(seated.mean()) if len(ok) else 0.0,
              nan=int(nan.sum()), contacts=int(ncon),
              tip_excess_mm=float(np.max(tip_r - clearance, initial=0) * 1e3),
              noct=int(m.noct))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--worlds", type=int, nargs="+", default=[1024, 4096])
  ap.add_argument("--variants", nargs="+", default=["convex", "sdf"])
  ap.add_argument("--clearance", type=float, default=0.0005)
  ap.add_argument("--chamfer", type=float, default=0.0005)
  ap.add_argument("--segments", type=int, default=64)
  ap.add_argument("--oct-depth", type=int, default=6)
  ap.add_argument("--seconds", type=float, default=2.0)
  ap.add_argument("--capture", action="store_true")
  ap.add_argument("--peg-segments", type=int, default=0)
  ap.add_argument("--broadphase", choices=["nxn", "sap_tile", "sap_segmented"], default=None)
  ap.add_argument("--filter", default=None, help="comma list of plane,sphere,aabb,obb")
  a = ap.parse_args()
  global CAPTURE
  CAPTURE = a.capture
  global BROAD, FILT
  from mujoco_warp._src.types import BroadphaseFilter, BroadphaseType
  if a.broadphase:
    BROAD = {"nxn": BroadphaseType.NXN, "sap_tile": BroadphaseType.SAP_TILE, "sap_segmented": BroadphaseType.SAP_SEGMENTED}[a.broadphase]
  if a.filter:
    FILT = BroadphaseFilter(0)
    for f in a.filter.split(","):
      FILT |= getattr(BroadphaseFilter, f.upper())
  print(f"clearance {a.clearance*1e3:.2f} mm, chamfer {a.chamfer*1e3:.2f} mm, "
        f"{a.segments} segments, octree depth {a.oct_depth}", flush=True)
  for v in a.variants:
    for n in a.worlds:
      try:
        r = run(v, n, a.clearance, a.chamfer, a.segments, a.oct_depth, a.seconds, peg_seg=a.peg_segments)
        print(f"{r['variant']:7s} {r['nworld']:5d} worlds: {r['ms_per_step']:7.2f} ms/step "
              f"({r['us_per_world_step']:.3f} us/world-step), seated {r['seated']:.3f}, "
              f"NaN {r['nan']}, contacts {r['contacts']}, tip excess {r['tip_excess_mm']:.3f} mm, "
              f"setup {r['setup_s']:.1f} s, noct {r['noct']}", flush=True)
      except Exception as e:  # report and continue with the next variant
        print(f"{v:7s} {n:5d} worlds: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
  main()
