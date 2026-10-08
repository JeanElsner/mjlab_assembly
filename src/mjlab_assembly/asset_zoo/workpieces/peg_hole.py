"""Round peg and socket as mjlab entities.

Collision geometry is exact convex pieces, so MuJoCo Warp's convex narrowphase
handles every contact: the peg is one convex mesh, the socket an exact convex
decomposition into annular sectors (see ``geometry``). The socket's visual is a
single smooth mesh of the same shape, so what is rendered is what collides, up
to the polygon resolution.

Collision bitmasks (MuJoCo pairs geoms when ``contype_a & conaffinity_b`` or
``contype_b & conaffinity_a`` is non-zero):

=================  =======  ===========  ========================================
geom               contype  conaffinity  collides with
=================  =======  ===========  ========================================
hand, fingers      1        1            socket (arm links: 0, no contacts)
fingertip pads     5        5            socket, held peg
peg                2        6            socket, fingertip pads (6 = 2 | 4)
socket             2        3            peg, robot
=================  =======  ===========  ========================================

A welded peg sits on a bare flange (no hand) and uses conaffinity 2, so it touches
only the socket; its adapter collides like the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab_assembly.asset_zoo.workpieces import geometry as g

ROBOT_BITS = (1, 1)
PAD_BITS = (5, 5)
PEG_BITS_HELD = (2, 6)
PEG_BITS_WELDED = (2, 2)
SOCKET_BITS = (2, 3)

# Workpiece contacts are stiffer than MuJoCo's default (solref 0.02 s), which lets a
# 10 N press sink a millimetre into the bore floor. A 5 ms time constant needs a
# physics step of at most 2.5 ms.
SOLREF = (0.005, 1.0)
SOLIMP = (0.95, 0.99, 0.0005, 0.5, 2.0)

PEG_RGBA = (0.80, 0.55, 0.20, 1.0)
SOCKET_RGBA = (0.62, 0.64, 0.68, 1.0)

TIP_SITE = "tip"
MOUTH_SITE = "mouth"


@dataclass(frozen=True)
class PegHoleCfg:
  """Geometry of a round peg-in-hole pair. Lengths in metres."""

  peg_radius: float = 0.006
  peg_length: float = 0.05
  clearance: float = 0.0005
  """Radial clearance between peg and bore."""
  peg_chamfer: float = 0.0005
  hole_chamfer: float = 0.0005
  hole_depth: float = 0.025
  hole_floor: float = 0.005
  """Slab under the bore; 0 makes a through-bore."""
  socket_radius: float = 0.02
  socket_height: float = 0.06
  """Height of the socket's top face (the mouth) above its base."""
  segments: int = 32
  """Polygon resolution of the round collision geometry."""
  grip_depth: float = 0.015
  """Distance from the peg's top face to the grasp point (the robot TCP)."""
  peg_mass: float = 0.05
  friction: float = 0.5

  @property
  def bore_radius(self) -> float:
    return self.peg_radius + self.clearance

  def __post_init__(self) -> None:
    if not 0.0 < self.clearance < self.peg_radius:
      raise ValueError("clearance must be positive and smaller than the peg radius")
    if self.socket_height < self.hole_depth + self.hole_floor:
      raise ValueError("socket_height must cover the bore and its floor")
    if not 0.0 < self.grip_depth < self.peg_length:
      raise ValueError("grip_depth must lie on the peg")


def _contact(geom: mujoco.MjsGeom, friction: float) -> None:
  geom.friction[0] = friction
  geom.solref[:] = SOLREF
  geom.solimp[:] = SOLIMP


def _mesh(spec: mujoco.MjSpec, name: str, m: g.Mesh) -> str:
  me = spec.add_mesh()
  me.name = name
  me.uservert = m.vertices.flatten()
  me.userface = m.faces.flatten()
  return name


def socket_spec(cfg: PegHoleCfg) -> mujoco.MjSpec:
  """Fixed socket: root body at the base, the mouth ``socket_height`` above it."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="socket")
  top = cfg.socket_height
  # Below the bore floor the socket is a solid cylinder (a pedestal).
  pedestal = cfg.socket_height - cfg.hole_depth - cfg.hole_floor
  visual = g.socket_mesh(
    cfg.bore_radius, cfg.socket_radius, cfg.hole_depth, cfg.hole_floor,
    cfg.hole_chamfer, max(cfg.segments, 96),
  )
  body.add_geom(
    name="socket_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=_mesh(spec, "socket_visual", visual), pos=(0, 0, top),
    contype=0, conaffinity=0, group=2, rgba=SOCKET_RGBA,
  )
  sectors = g.socket_sectors(
    cfg.bore_radius, cfg.socket_radius, cfg.hole_depth, cfg.hole_floor,
    cfg.hole_chamfer, cfg.segments,
  )
  for i, part in enumerate(sectors):
    geom = body.add_geom(
      name=f"socket{i}_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=_mesh(spec, f"socket{i}", part), pos=(0, 0, top),
      contype=SOCKET_BITS[0], conaffinity=SOCKET_BITS[1], group=3,
    )
    _contact(geom, cfg.friction)
  if pedestal > 0.0:
    # Round to look at, square to collide with (box contacts are cheap and
    # multi-point; the pedestal only ever meets the robot's hand).
    body.add_geom(
      name="pedestal_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(cfg.socket_radius, pedestal / 2, 0), pos=(0, 0, pedestal / 2),
      contype=0, conaffinity=0, group=2, rgba=SOCKET_RGBA,
    )
    body.add_geom(
      name="pedestal_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(cfg.socket_radius, cfg.socket_radius, pedestal / 2), pos=(0, 0, pedestal / 2),
      contype=SOCKET_BITS[0], conaffinity=SOCKET_BITS[1], group=3,
    )
  body.add_site(name=MOUTH_SITE, pos=(0, 0, top), size=(0.002, 0, 0))
  return spec


def _add_peg_geoms(body: mujoco.MjsBody, spec: mujoco.MjSpec, cfg: PegHoleCfg,
                   bits: tuple[int, int], pos: tuple[float, float, float],
                   quat: tuple[float, float, float, float], name: str) -> None:
  """Peg geometry in ``body``; the peg's tip at ``pos``, its axis +z rotated by quat."""
  mesh = g.peg_mesh(cfg.peg_radius, cfg.peg_length, cfg.peg_chamfer, cfg.segments)
  geom = body.add_geom(
    name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=_mesh(spec, "peg", mesh), pos=pos, quat=quat,
    contype=bits[0], conaffinity=bits[1], rgba=PEG_RGBA, mass=cfg.peg_mass,
  )
  _contact(geom, cfg.friction)
  geom.friction[1:] = (0.005, 0.0001)
  # The tip site's +z is the peg axis, pointing from the tip to the top.
  body.add_site(name=TIP_SITE, pos=pos, quat=quat, size=(0.002, 0, 0))


def peg_spec(cfg: PegHoleCfg) -> mujoco.MjSpec:
  """Free peg for grasping. Root body origin at the grasp point, axis +z to the top."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="peg")
  body.add_freejoint(name="peg_joint")
  tip_z = -(cfg.peg_length - cfg.grip_depth)
  _add_peg_geoms(body, spec, cfg, PEG_BITS_HELD, (0.0, 0.0, tip_z), (1, 0, 0, 0),
                 "peg_collision")
  return spec


def weld_peg_to_flange(spec: mujoco.MjSpec, cfg: PegHoleCfg, tcp_offset: float,
                       flange_body: str = "attachment") -> None:
  """Fix the peg to a bare flange through a cylindrical adapter.

  The adapter is sized so that the peg's grasp point lands on the TCP, where a
  hand would hold it: the peg then sits exactly as in the grasped task. The peg
  takes the flange's gravity compensation, as a welded tool is part of the
  end-effector load the real arm compensates.
  """
  flange = spec.body(flange_body)
  adapter_len = tcp_offset - cfg.grip_depth
  if adapter_len <= 0.0:
    raise ValueError("grip_depth must be shorter than the TCP offset")
  adapter = flange.add_geom(
    name="adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(cfg.peg_radius + 0.008, adapter_len / 2, 0), pos=(0, 0, adapter_len / 2),
    contype=ROBOT_BITS[0], conaffinity=ROBOT_BITS[1], rgba=(0.55, 0.57, 0.6, 1.0),
    mass=0.1,
  )
  del adapter
  peg = flange.add_body(name="peg", pos=(0.0, 0.0, tcp_offset))
  peg.gravcomp = flange.gravcomp
  # The peg tip lies further out along the flange's +z; the mesh grows along +z
  # from its tip, so flip it to grow back toward the flange. Named outside the
  # robot's ``.*_collision`` pattern, so the robot's collision config leaves it alone.
  tip = (0.0, 0.0, cfg.peg_length - cfg.grip_depth)
  _add_peg_geoms(peg, spec, cfg, PEG_BITS_WELDED, tip, (0, 1, 0, 0), "welded_peg")


def peg_in_grasp_pose(cfg: PegHoleCfg) -> tuple[np.ndarray, np.ndarray]:
  """Pose of the free peg's root in the TCP frame: (position, quaternion wxyz).

  The peg root is its grasp point and its +z axis runs from tip to top, so in the
  TCP frame (+z out of the hand) it sits at the origin, rotated 180 degrees about x.
  """
  return np.zeros(3), np.array([0.0, 1.0, 0.0, 0.0])
