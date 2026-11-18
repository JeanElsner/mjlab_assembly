"""Assembly parts built with mjfit, as mjlab entities.

Every assembly provides the same pieces to a task:

- ``socket_spec()``: the fixed part, with a ``mouth`` site where the held part
  enters (its +z out of the opening),
- ``part_spec()``: the held part, a free body whose origin is the grasp point,
  with a ``tip`` site on its axis (+z from the tip toward the grasp),
- ``weld_to_flange(spec, tcp_offset)``: the held part fixed to a bare flange,
- ``seat_depth``: how far the tip travels below the mouth when seated,
- ``clearance``, ``grip_width``, and ``symmetry``: the order of the part's
  rotational symmetry about its axis (0 for round parts, whose yaw does not
  matter; 2 for a rectangle; the tooth count for a gear).

Threaded parts add ``lead``, the advance per turn (positive for a right-hand
thread), and ``success_depth``, how far the tip must be below the mouth to
count as done when that is not 95 % of ``seat_depth``.

Collision geometry is mjfit's exact convex pieces, so the represented fit is
the specified one. Collision bitmasks and contact parameters follow
``peg_hole``.
"""

from __future__ import annotations

import functools
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mjfit
import mujoco
import numpy as np
from mjfit import shapes

from mjlab_assembly.asset_zoo.workpieces import geometry as g
from mjlab_assembly.asset_zoo.workpieces.peg_hole import (
  MOUTH_SITE,
  PEG_BITS_HELD,
  PEG_BITS_WELDED,
  PEG_RGBA,
  ROBOT_BITS,
  SOCKET_BITS,
  SOCKET_RGBA,
  SOLIMP,
  SOLREF,
  TIP_SITE,
)

GEAR_RGBA = ((0.55, 0.58, 0.62, 1.0), (0.80, 0.55, 0.20, 1.0), (0.45, 0.50, 0.70, 1.0))
FLIP = (0.0, 1.0, 0.0, 0.0)  # 180 degrees about x


# ----- helpers -----------------------------------------------------------------


@dataclass
class _Piece:
  name: str
  vertices: np.ndarray  # (n, 3), in the frame of the mjfit body
  mass: float | None


def _bodies(xml: str) -> dict[str, list[_Piece]]:
  """Collision pieces of every body in an mjfit model, with inline vertices."""
  root = ET.fromstring(xml)
  verts = {
    m.get("name"): np.array(m.get("vertex").split(), float).reshape(-1, 3)
    for m in root.iter("mesh")
    if m.get("vertex")
  }
  out: dict[str, list[_Piece]] = {}
  for body in root.iter("body"):
    pieces = []
    for geom in body.findall("geom"):
      mesh = geom.get("mesh")
      if mesh in verts and geom.get("class") != "mjfit_visual":
        mass = geom.get("mass")
        pieces.append(_Piece(mesh, verts[mesh], float(mass) if mass else None))
    out[body.get("name")] = pieces
  return out


def _add_pieces(
  spec: mujoco.MjSpec,
  body: mujoco.MjsBody,
  pieces: list[_Piece],
  prefix: str,
  bits: tuple[int, int],
  friction: float,
  rgba,
  pos=(0.0, 0.0, 0.0),
  quat=(1.0, 0.0, 0.0, 0.0),
  density: float | None = None,
) -> None:
  for p in pieces:
    mesh = spec.add_mesh()
    mesh.name = f"{prefix}_{p.name}"
    mesh.uservert = p.vertices.flatten()
    geom = body.add_geom(
      name=f"{prefix}_{p.name}_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh.name, pos=pos, quat=quat, contype=bits[0], conaffinity=bits[1],
      rgba=rgba,
    )
    if p.mass is not None:
      geom.mass = p.mass
    elif density is not None:
      geom.density = density
    geom.friction[:] = (friction, 0.005, 0.0001)
    geom.solref[:] = SOLREF
    geom.solimp[:] = SOLIMP


def _pedestal(body: mujoco.MjsBody, half: float, height: float) -> None:
  if height <= 0.0:
    return
  body.add_geom(
    name="pedestal_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half, half, height / 2), pos=(0, 0, height / 2),
    contype=SOCKET_BITS[0], conaffinity=SOCKET_BITS[1], rgba=SOCKET_RGBA,
  )


def _adapter(flange: mujoco.MjsBody, length: float, radius: float) -> None:
  if length <= 0.0:
    raise ValueError("The grasp point must lie closer to the flange than the TCP.")
  flange.add_geom(
    name="adapter", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(radius, length / 2, 0), pos=(0, 0, length / 2),
    contype=ROBOT_BITS[0], conaffinity=ROBOT_BITS[1], rgba=(0.55, 0.57, 0.6, 1.0),
    mass=0.1,
  )


# ----- peg (rod) insertion ---------------------------------------------------


@dataclass(frozen=True)
class InsertCfg:
  """A peg of any convex outline in a blind hole, from mjfit.

  The hole's outline is an mjfit shape spec; the peg is the hole moved in by
  ``clearance`` on every edge. Lengths in metres.
  """

  hole: dict = field(default_factory=lambda: {"type": "circle", "radius": 0.0045})
  clearance: float = 0.000507
  peg_length: float = 0.05
  hole_depth: float = 0.025
  hole_floor: float = 0.005
  peg_chamfer: float = 0.0005
  hole_chamfer: float = 0.001
  socket_size: float = 0.04
  socket_height: float = 0.06
  grip_depth: float = 0.015
  """Distance from the peg's top face to the grasp point (the TCP)."""
  peg_mass: float = 0.019
  friction: float = 0.75
  max_error: float = 1e-5
  """How far round outlines may stray from their circles."""
  symmetry: int = 0
  socket_yaw_range: float = 0.0
  """Random socket yaw per episode, +- this [rad] (for parts whose yaw matters)."""
  grasp_yaw_noise: float = 0.0
  episode_length_s: float = 10.0
  nconmax: int = 128
  njmax: int = 1500

  # Protocol.

  @property
  def seat_depth(self) -> float:
    return self.hole_depth

  @property
  def grip_width(self) -> float:
    """Width of the peg across the jaws (the outline's extent along y)."""
    pts = self._peg_outline()
    return float(pts[:, 1].max() - pts[:, 1].min())

  def socket_spec(self) -> mujoco.MjSpec:
    pieces = _bodies(self._xml())["plate"]
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="socket")
    pedestal = self.socket_height - self.hole_depth - self.hole_floor
    _pedestal(body, self.socket_size / 2, pedestal)
    # mjfit's plate frame is the plate's mid-height; put its top on the mouth.
    centre = (0.0, 0.0, self.socket_height - self.hole_depth / 2)
    _add_pieces(spec, body, pieces, "socket", SOCKET_BITS, self.friction, SOCKET_RGBA, pos=centre)
    body.add_site(name=MOUTH_SITE, pos=(0, 0, self.socket_height), size=(0.002, 0, 0))
    return spec

  def part_spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="peg")
    body.add_freejoint(name="peg_joint")
    self._add_peg(spec, body, PEG_BITS_HELD, sign=1.0)
    return spec

  def weld_to_flange(self, spec: mujoco.MjSpec, tcp_offset: float,
                     flange_body: str = "attachment") -> None:
    flange = spec.body(flange_body)
    _adapter(flange, tcp_offset - self.grip_depth, self.grip_width / 2 + 0.008)
    peg = flange.add_body(name="peg", pos=(0.0, 0.0, tcp_offset))
    peg.gravcomp = flange.gravcomp
    self._add_peg(spec, peg, PEG_BITS_WELDED, sign=-1.0)

  # Internals.

  def _xml(self) -> str:
    hole = {**self.hole}
    if hole.get("type") in ("circle", "stadium"):
      hole.setdefault("max_error", self.max_error)
    return mjfit.peg(
      hole=hole, clearance=self.clearance, peg_length=self.peg_length,
      plate_thickness=self.hole_depth, bottom_thickness=self.hole_floor,
      peg_chamfer=self.peg_chamfer, hole_chamfer=self.hole_chamfer,
      plate_size=self.socket_size, peg_mass=self.peg_mass, center=False,
      standalone=False,
    )

  def _peg_outline(self) -> np.ndarray:
    pieces = _bodies(self._xml())["peg"]
    return np.vstack([p.vertices[:, :2] for p in pieces])

  def _add_peg(self, spec, body, bits, sign: float) -> None:
    """Peg geoms in ``body`` (origin at the grasp point). With sign=1 the peg's
    axis is the body's +z (tip below); with sign=-1 it is flipped, the tip
    further out along the flange's +z."""
    pieces = _bodies(self._xml())["peg"]
    # mjfit's peg frame is the middle of the insertion section.
    shift = -(self.peg_length / 2 - self.grip_depth)
    if sign > 0:
      pos, quat, tip = (0, 0, shift), (1, 0, 0, 0), (0, 0, -(self.peg_length - self.grip_depth))
    else:
      pos, quat, tip = (0, 0, -shift), FLIP, (0, 0, self.peg_length - self.grip_depth)
    _add_pieces(spec, body, pieces, "peg", bits, self.friction, PEG_RGBA, pos=pos, quat=quat)
    body.add_site(name=TIP_SITE, pos=tip, quat=(1, 0, 0, 0) if sign > 0 else FLIP,
                  size=(0.002, 0, 0))


def rod_insert(peg_diameter: float = 0.007986, **overrides) -> InsertCfg:
  """Round peg and hole. The defaults match Isaac Lab Factory's 8 mm assets as
  shipped: a 7.986 mm peg in a 9 mm hole (0.507 mm radial clearance), 25 mm deep,
  chamfered 0.5 mm (peg) and 1 mm (hole) at 45 degrees."""
  c = overrides.pop("clearance", 0.000507)
  return InsertCfg(hole={"type": "circle", "radius": peg_diameter / 2 + c}, clearance=c,
                   **overrides)


def peg_insert(width: float = 0.012, height: float = 0.008, **overrides) -> InsertCfg:
  """Rectangular peg and hole, gripped across ``height`` (the outline's y
  extent); clearance and chamfers as for the round 8 mm part."""
  c = overrides.pop("clearance", 0.000507)
  overrides.setdefault("socket_yaw_range", math.radians(30))
  return InsertCfg(
    hole={"type": "rect", "width": width + 2 * c, "height": height + 2 * c},
    clearance=c, symmetry=2, **overrides,
  )


# ----- gear meshing ----------------------------------------------------------


@dataclass(frozen=True)
class GearMeshCfg:
  """Factory's gear meshing task: a medium gear slid onto the middle of three
  shafts, meshing with a small and a large gear already in place.

  Module-1 spur gears with 20, 40 and 60 teeth, 10 mm thick, each with a grip
  hub on top; 9.5 mm shafts with a 1 mm chamfer, 20 mm tall above a base.
  Dimensions follow Isaac Lab's Factory assets. The small and large gears are
  fixed; the held gear's teeth must find their phase.
  """

  module: float = 0.001
  teeth: tuple[int, int, int] = (20, 40, 60)  # small, medium (held), large
  face_width: float = 0.01
  hub_height: float = 0.015
  hub_radii: tuple[float, float, float] = (0.00875, 0.015, 0.01875)
  backlash_gap: float = 0.0005
  """Extra centre distance beyond the pitch radii (Factory: 0.5 mm)."""
  bore: float = 0.01054
  bore_chamfer: float = 0.0005
  shaft: float = 0.0095
  shaft_height: float = 0.02
  shaft_chamfer: float = 0.001
  base_height: float = 0.04
  """Height of the base's top face (the gears' seat) above the ground."""
  seat_radius: float = 0.007
  """Radius of the ring around the shaft that the held gear seats on."""
  grip_depth: float = 0.003
  """Distance from the hub's top face to the grasp point; the fingertip pads
  reach about 8 mm below it, so this leaves them about 4 mm above the disc."""
  gear_mass: float = 0.012
  friction: float = 0.75
  max_error: float = 1e-4
  socket_yaw_range: float = 0.0
  episode_length_s: float = 10.0
  nconmax: int = 384
  njmax: int = 3000

  @property
  def grasp_yaw_noise(self) -> float:
    """The held gear's tooth phase in the hand: anywhere within a pitch."""
    return math.pi / self.teeth[1]

  # Protocol.

  @property
  def symmetry(self) -> int:
    return self.teeth[1]

  @property
  def clearance(self) -> float:
    return (self.bore - self.shaft) / 2

  @property
  def seat_depth(self) -> float:
    return self.shaft_height

  @property
  def grip_width(self) -> float:
    return 2 * self.hub_radii[1]

  def socket_spec(self) -> mujoco.MjSpec:
    """The base with three shafts and the fixed small and large gears. The
    mouth is the top of the medium gear's shaft, at the body origin's xy."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="socket")
    xs = self._centres()
    half_x = max(abs(x) for x in xs) + max(self._outer_radii())
    # The base collides with the robot only. The held gear seats on a ring
    # around its shaft, flush with the base's top: resting on the whole base,
    # each of its ~60 lower pieces would add contacts (about 230 in all).
    body.add_geom(
      name="base_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(half_x, 0.03, self.base_height / 2), pos=(0, 0, self.base_height / 2),
      contype=ROBOT_BITS[0], conaffinity=ROBOT_BITS[1], rgba=SOCKET_RGBA,
    )
    seat = shapes.circle(self.seat_radius, n=16)
    _add_pieces(spec, body, [_Piece("seat", _prism(seat, self.base_height - 0.002,
                                                   self.base_height), None)],
                "seat", SOCKET_BITS, self.friction, SOCKET_RGBA)
    top = self.base_height + self.shaft_height
    shaft = shapes.circle(self.shaft / 2, max_error=self.max_error)
    tip = np.asarray(_inset(shaft, self.shaft_chamfer))
    shaft_pieces = [
      _Piece("shaft", _prism(shaft, self.base_height, top - self.shaft_chamfer), None),
      _Piece("shaft_tip", np.vstack([_ring(shaft, top - self.shaft_chamfer), _ring(tip, top)]),
             None),
    ]
    for k, x in enumerate(xs):
      _add_pieces(spec, body, shaft_pieces, f"shaft{k}", SOCKET_BITS, self.friction,
                  (0.2, 0.2, 0.23, 1.0), pos=(x, 0, 0))
    for k in (0, 2):  # the fixed gears, seated on the base
      # Bored like the held gear: a solid hub would be one prism through every
      # root point (~180 corners for 60 teeth), and MuJoCo Warp sizes its
      # convex-collision buffers by the largest face of any mesh.
      pieces = self._gear_pieces(k, bored=True)
      z = self.base_height + self.face_width / 2
      _add_pieces(spec, body, pieces, f"gear{k}", SOCKET_BITS, self.friction, GEAR_RGBA[k],
                  pos=(xs[k], 0, z))
    body.add_site(name=MOUTH_SITE, pos=(0, 0, top), size=(0.002, 0, 0))
    return spec

  def part_spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="peg")
    body.add_freejoint(name="peg_joint")
    self._add_held_gear(spec, body, PEG_BITS_HELD, sign=1.0)
    return spec

  def weld_to_flange(self, spec: mujoco.MjSpec, tcp_offset: float,
                     flange_body: str = "attachment") -> None:
    flange = spec.body(flange_body)
    _adapter(flange, tcp_offset - self.grip_depth, self.hub_radii[1])
    gear = flange.add_body(name="peg", pos=(0.0, 0.0, tcp_offset))
    gear.gravcomp = flange.gravcomp
    self._add_held_gear(spec, gear, PEG_BITS_WELDED, sign=-1.0)

  # Internals.

  def _pitch_radii(self) -> list[float]:
    return [self.module * t / 2 for t in self.teeth]

  def _outer_radii(self) -> list[float]:
    return [r + self.module for r in self._pitch_radii()]

  def _centres(self) -> list[float]:
    """x of the small, medium and large gears; the medium gear at the origin,
    the small one at +x and the large one at -x, as on Factory's base."""
    r = self._pitch_radii()
    return [r[0] + r[1] + self.backlash_gap, 0.0, -(r[1] + r[2] + self.backlash_gap)]

  def _phase(self, k: int) -> float:
    """Tooth phase that meshes gear k with the medium gear at phase 0 (a tooth
    on +x): a gap of the neighbour faces the medium gear."""
    t = self.teeth[k]
    if k == 1:
      return 0.0
    gap = math.pi if k == 0 else 0.0  # direction of the medium gear from gear k
    return gap - math.pi / t

  def _gear_pieces(self, k: int, bored: bool) -> list[_Piece]:
    kw = dict(teeth=self.teeth[k], module=self.module, face_width=self.face_width,
              standalone=False, axles=False, motor=False)
    if bored:
      kw.update(bore=self.bore, bore_chamfer=self.bore_chamfer, max_error=self.max_error)
    pieces = [p for p in _bodies(mjfit.gear(**kw))["gear_a"] if "_col_" in p.name]
    rot = self._phase(k)
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    hub_inner = self.bore / 2 if bored else 0.6 * self.hub_radii[k]
    hub = g.socket_sectors(hub_inner, self.hub_radii[k], self.hub_height, 0.0, 0.0, 16)
    top = self.face_width / 2 + self.hub_height
    pieces = [_Piece(p.name, p.vertices @ R.T, None) for p in pieces]
    pieces += [_Piece(f"hub_{i}", h.vertices + np.array([0, 0, top]), None)
               for i, h in enumerate(hub)]
    return pieces

  def _add_held_gear(self, spec, body, bits, sign: float) -> None:
    """The medium gear in ``body`` (origin at the grasp point on the hub)."""
    pieces = self._gear_pieces(1, bored=True)
    # Gear frame: mid-plane of the toothed disc. Grasp point: grip_depth below
    # the hub's top.
    grasp = self.face_width / 2 + self.hub_height - self.grip_depth
    bottom = -self.face_width / 2
    volume = sum(_hull_volume(p.vertices) for p in pieces)
    density = self.gear_mass / volume
    if sign > 0:
      pos, quat, tip = (0, 0, -grasp), (1, 0, 0, 0), (0, 0, bottom - grasp)
    else:
      pos, quat, tip = (0, 0, grasp), FLIP, (0, 0, grasp - bottom)
    _add_pieces(spec, body, pieces, "gear", bits, self.friction, GEAR_RGBA[1], pos=pos,
                quat=quat, density=density)
    body.add_site(name=TIP_SITE, pos=tip, quat=(1, 0, 0, 0) if sign > 0 else FLIP,
                  size=(0.002, 0, 0))



# ----- nut threading ---------------------------------------------------------


@dataclass(frozen=True)
class NutThreadCfg:
  """Factory's nut threading task: an M16x2 hex nut held in the gripper and
  turned onto a fixed bolt.

  Dimensions follow Isaac Lab's Factory meshes: a 24 mm hex nut, 13 mm tall, on
  a bolt with 25 mm of thread above a 24 mm hex head 10 mm tall. The threads are
  mjfit's ISO profile swept as convex wedges, chamfered 45 degrees at the bolt's
  tip and the nut's faces. ``clearance`` is the gap between facing flanks.

  As in Factory, the nut's target is its bottom face 1.5 pitches below the
  bolt's tip, and it counts as threaded at 1.125 pitches (2.25 mm). The thread's
  phase at the start, the nut's yaw relative to the bolt, is ``phase`` +-
  ``socket_yaw_range``, as Factory's bolt (+-30 degrees) and hand (+-15 degrees)
  yaw ranges give. Within it the nut comes to rest 1.6 to 2.0 mm down and must
  turn about 0.1 to 0.4 of a turn. A fully random phase (``socket_yaw_range=pi``)
  makes some starts rest 2.4 mm down and others need more than a turn, more
  than the wrist allows: the policy would have to find the thread's start by
  turning back first.
  """

  major_diameter: float = 0.016
  pitch: float = 0.002
  clearance: float = 0.0003
  """Gap between facing flanks. Tighter fits jam at the thread's start once
  the hand tilts a degree or two while turning."""
  segments_per_turn: int = 24
  chamfer: float = 0.001
  """Axial length of the chamfers; about the thread depth, so 45 degrees."""
  thread_length: float = 0.025
  collision_length: float = 0.008
  """Bolt thread below the tip that collides; deeper turns are only drawn,
  since the nut cannot reach them within the wrist's range."""
  head_size: float = 0.024
  head_height: float = 0.01
  nut_size: float = 0.024
  nut_height: float = 0.013
  socket_height: float = 0.09
  """Height of the bolt's tip above the ground."""
  grip_depth: float = 0.003
  """Distance from the nut's top face to the grasp point."""
  nut_mass: float = 0.03
  friction: float = 0.25
  seat_pitches: float = 1.5
  success_depth: float = 0.00225
  phase: float = math.radians(30)
  socket_yaw_range: float = math.radians(45)
  grasp_yaw_noise: float = 0.0
  episode_length_s: float = 10.0
  nconmax: int = 512
  njmax: int = 3000

  symmetry = 0  # the thread's phase cannot be observed

  # Protocol.

  @property
  def lead(self) -> float:
    return self.pitch

  @property
  def seat_depth(self) -> float:
    return self.seat_pitches * self.pitch

  @property
  def grip_width(self) -> float:
    return self.nut_size

  def socket_spec(self) -> mujoco.MjSpec:
    """The bolt, tip up, on its head and a pedestal. The mouth is the tip."""
    xml = self._xml()
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="socket")
    pedestal = self.socket_height - self.thread_length - self.head_height
    _pedestal(body, self.head_size / 2, pedestal)
    head = _prism(_hexagon(self.head_size), pedestal, pedestal + self.head_height)
    _add_pieces(spec, body, [_Piece("head", head, None)], "bolt", SOCKET_BITS, self.friction,
                GEAR_RGBA[0])
    # mjfit's bolt frame is its tip, the thread along +z; turn it tip up as in
    # mjfit's nut-driven scene, so the nut's frame there interlocks.
    tip = (0.0, 0.0, self.socket_height)
    core = body.add_geom(
      name="bolt_core_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(self._core_radius(xml), self.thread_length / 2, 0),
      pos=(0, 0, self.socket_height - self.thread_length / 2),
      contype=SOCKET_BITS[0], conaffinity=SOCKET_BITS[1], rgba=GEAR_RGBA[0],
    )
    core.friction[:] = (self.friction, 0.005, 0.0001)
    core.solref[:] = SOLREF
    core.solimp[:] = SOLIMP
    near, far = [], []
    for p in _bodies(xml)["bolt"]:
      (near if p.vertices[:, 2].min() < self.collision_length else far).append(p)
    _add_pieces(spec, body, near, "bolt", SOCKET_BITS, self.friction, GEAR_RGBA[0],
                pos=tip, quat=_TIP_UP)
    _add_pieces(spec, body, far, "bolt_visual", (0, 0), self.friction, GEAR_RGBA[0],
                pos=tip, quat=_TIP_UP)
    body.add_site(name=MOUTH_SITE, pos=tip, size=(0.002, 0, 0))
    return spec

  def part_spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="peg")
    body.add_freejoint(name="peg_joint")
    self._add_nut(spec, body, PEG_BITS_HELD, sign=1.0)
    return spec

  def weld_to_flange(self, spec: mujoco.MjSpec, tcp_offset: float,
                     flange_body: str = "attachment") -> None:
    flange = spec.body(flange_body)
    _adapter(flange, tcp_offset - self.grip_depth, self.nut_size / 2)
    nut = flange.add_body(name="peg", pos=(0.0, 0.0, tcp_offset))
    nut.gravcomp = flange.gravcomp
    self._add_nut(spec, nut, PEG_BITS_WELDED, sign=-1.0)

  # Internals.

  def _xml(self) -> str:
    return _thread_xml(
      major_diameter=self.major_diameter, pitch=self.pitch, thread_length=self.thread_length,
      clearance=self.clearance, segments_per_turn=self.segments_per_turn, chamfer=self.chamfer,
      bolt_head_height=0.0, nut_size=self.nut_size, nut_height=self.nut_height,
      driven="nut", friction=self.friction, standalone=False,
    )

  @staticmethod
  def _core_radius(xml: str) -> float:
    core = next(g for g in ET.fromstring(xml).iter("geom") if g.get("name") == "bolt_core")
    return float(core.get("size").split()[0])

  def _add_nut(self, spec, body, bits, sign: float) -> None:
    """The nut in ``body`` (origin at the grasp point). mjfit's nut frame is
    the centre of its bottom face, the axis +z."""
    pieces = _bodies(self._xml())["nut"]
    bottom = self.nut_height - self.grip_depth
    volume = sum(_hull_volume(p.vertices) for p in pieces)
    if sign > 0:
      pos, quat, tip = (0, 0, -bottom), (1, 0, 0, 0), (0, 0, -bottom)
    else:
      pos, quat, tip = (0, 0, bottom), FLIP, (0, 0, bottom)
    _add_pieces(spec, body, pieces, "nut", bits, self.friction, GEAR_RGBA[1], pos=pos,
                quat=quat, density=self.nut_mass / volume)
    body.add_site(name=TIP_SITE, pos=tip, quat=(1, 0, 0, 0) if sign > 0 else FLIP,
                  size=(0.002, 0, 0))


_TIP_UP = (0.0, 0.0, 1.0, 0.0)  # 180 degrees about y, as in mjfit's nut-driven scene


@functools.lru_cache(maxsize=8)
def _thread_xml(**kw) -> str:
  return mjfit.thread(**kw)


def _hexagon(across_flats: float) -> np.ndarray:
  """A regular hexagon with flats facing +-y (across the jaws)."""
  a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
  return across_flats / math.sqrt(3) * np.column_stack([np.cos(a), np.sin(a)])


def _ring(poly: np.ndarray, z: float) -> np.ndarray:
  return np.column_stack([poly, np.full(len(poly), z)])


def _prism(poly: np.ndarray, z0: float, z1: float) -> np.ndarray:
  return np.vstack([_ring(poly, z0), _ring(poly, z1)])


def _inset(poly: np.ndarray, d: float) -> np.ndarray:
  """A regular polygon around the origin moved in by d on every edge."""
  r = np.linalg.norm(poly, axis=1, keepdims=True)
  apothem = r * math.cos(math.pi / len(poly))
  return poly * (apothem - d) / apothem


def _hull_volume(v: np.ndarray) -> float:
  from scipy.spatial import ConvexHull

  return float(ConvexHull(v).volume)
