"""Parametric triangle meshes for peg-in-hole workpieces.

Every mesh is closed, outward-oriented and built with numpy only, so the same
geometry can serve as a MuJoCo convex mesh, as the source of a MuJoCo octree
SDF, and as the visual. Lengths are in metres; the bore axis is +z with the
socket's top face at z = 0 and material below it.

Two socket representations are provided:

- ``socket_mesh``: one closed, non-convex mesh with a cylindrical bore, for SDF
  collision and for rendering.
- ``socket_sectors``: an exact convex decomposition into ``n`` annular sectors
  (plus an optional floor), for convex-convex collision. Each sector's inner face
  is a flat chord whose apothem equals the bore radius, so the polygonal bore is
  never narrower than the round one; at ``n = 64`` its corners stand
  ``r * (1 / cos(pi / n) - 1)``, about 1.2e-3 of the radius, outside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
  vertices: np.ndarray  # (V, 3) float
  faces: np.ndarray  # (F, 3) int, counter-clockwise seen from outside


def _revolve(profile: list[tuple[float, float]], n: int, close_axis: bool) -> Mesh:
  """Revolve an (r, z) polyline about the z axis into a closed mesh.

  ``profile`` runs from the bottom on the axis (r = 0) to the top on the axis
  when ``close_axis`` is True. Points with r = 0 become single pole vertices.
  """
  th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
  verts: list[np.ndarray] = []
  rings: list[list[int] | int] = []
  for r, z in profile:
    if r == 0.0:
      rings.append(len(verts))
      verts.append(np.array([0.0, 0.0, z]))
    else:
      start = len(verts)
      for t in th:
        verts.append(np.array([r * np.cos(t), r * np.sin(t), z]))
      rings.append(list(range(start, start + n)))
  faces: list[tuple[int, int, int]] = []
  for a, b in zip(rings[:-1], rings[1:], strict=True):
    for i in range(n):
      j = (i + 1) % n
      if isinstance(a, int) and isinstance(b, list):
        faces.append((a, b[j], b[i]))
      elif isinstance(a, list) and isinstance(b, int):
        faces.append((a[i], a[j], b))
      elif isinstance(a, list) and isinstance(b, list):
        faces.append((a[i], a[j], b[j]))
        faces.append((a[i], b[j], b[i]))
  del close_axis
  return Mesh(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32))


def peg_mesh(radius: float, length: float, chamfer: float = 0.0, n: int = 64) -> Mesh:
  """Round peg along +z, tip at z = 0, top at z = length.

  ``chamfer`` is a 45-degree bevel of the tip edge, in metres along both the
  radial and axial directions. The result is convex.
  """
  if not 0.0 <= chamfer < radius:
    raise ValueError("chamfer must be in [0, radius)")
  profile = [(0.0, 0.0)]
  if chamfer > 0.0:
    profile += [(radius - chamfer, 0.0), (radius, chamfer)]
  else:
    profile += [(radius, 0.0)]
  profile += [(radius, length), (0.0, length)]
  return _revolve(profile, n, close_axis=True)


def socket_mesh(
  bore_radius: float,
  outer_radius: float,
  depth: float,
  floor: float = 0.0,
  chamfer: float = 0.0,
  n: int = 64,
) -> Mesh:
  """Round socket with a bore along -z from its top face at z = 0.

  ``floor`` > 0 closes the bore with a slab of that thickness (a blind bore of
  the given ``depth``); ``floor`` = 0 makes a through-bore. ``chamfer`` bevels
  the bore's entry edge at 45 degrees.
  """
  if not 0.0 <= chamfer < min(depth, outer_radius - bore_radius):
    raise ValueError("chamfer too large for the socket")
  z_bot = -depth - floor
  # Outer surface from the bottom, then the top face inward, then down the bore.
  prof: list[tuple[float, float]] = []
  if floor > 0.0:
    prof += [(0.0, z_bot), (outer_radius, z_bot), (outer_radius, 0.0)]
    if chamfer > 0.0:
      prof += [(bore_radius + chamfer, 0.0), (bore_radius, -chamfer)]
    else:
      prof += [(bore_radius, 0.0)]
    prof += [(bore_radius, -depth), (0.0, -depth)]
    return _revolve(prof, n, close_axis=True)
  # Through-bore: a closed annulus (torus-like profile loop).
  prof = [(bore_radius, z_bot), (outer_radius, z_bot), (outer_radius, 0.0)]
  if chamfer > 0.0:
    prof += [(bore_radius + chamfer, 0.0), (bore_radius, -chamfer)]
  else:
    prof += [(bore_radius, 0.0)]
  prof += [(bore_radius, z_bot)]
  m = _revolve(prof, n, close_axis=False)
  return _dedupe_seam(m, n, len(prof))


def _dedupe_seam(m: Mesh, n: int, nprof: int) -> Mesh:
  """Merge the duplicated first and last rings of a closed revolved loop."""
  last = (nprof - 1) * n
  verts = m.vertices[:last]
  faces = m.faces.copy()
  faces[faces >= last] -= last
  return Mesh(verts, faces)


def _prism(polygon_xy: np.ndarray, z0: float, z1: float) -> Mesh:
  """Extrude a convex counter-clockwise polygon between z0 < z1."""
  k = len(polygon_xy)
  bot = np.column_stack([polygon_xy, np.full(k, z0)])
  top = np.column_stack([polygon_xy, np.full(k, z1)])
  verts = np.vstack([bot, top])
  faces = []
  for i in range(1, k - 1):
    faces.append((0, i + 1, i))  # bottom, facing -z
    faces.append((k, k + i, k + i + 1))  # top, facing +z
  for i in range(k):
    j = (i + 1) % k
    faces.append((i, j, k + j))
    faces.append((i, k + j, k + i))
  return Mesh(verts, np.asarray(faces, dtype=np.int32))


def socket_sectors(
  bore_radius: float,
  outer_radius: float,
  depth: float,
  floor: float = 0.0,
  chamfer: float = 0.0,
  n: int = 64,
) -> list[Mesh]:
  """Exact convex decomposition of ``socket_mesh`` into ``n`` sectors (+ floor).

  The chamfer is cut into each sector as a second, smaller prism on top, so
  every part stays convex.
  """
  parts: list[Mesh] = []
  half = np.pi / n
  # Outer corners at radius outer_radius / cos(half) keep the outer face round
  # to the same tolerance as the bore.
  ro = outer_radius / np.cos(half)
  for i in range(n):
    a0 = 2.0 * np.pi * i / n - half
    a1 = a0 + 2.0 * half

    def ring(r_apothem: float, a0: float = a0, a1: float = a1) -> tuple[np.ndarray, np.ndarray]:
      rc = r_apothem / np.cos(half)
      return (np.array([rc * np.cos(a0), rc * np.sin(a0)]),
              np.array([rc * np.cos(a1), rc * np.sin(a1)]))

    i0, i1 = ring(bore_radius)
    o0 = np.array([ro * np.cos(a0), ro * np.sin(a0)])
    o1 = np.array([ro * np.cos(a1), ro * np.sin(a1)])
    if chamfer > 0.0:
      parts.append(_prism(np.array([i0, o0, o1, i1]), -depth - floor, -chamfer))
      c0, c1 = ring(bore_radius + chamfer)
      # Chamfer band: a convex hull of the inner chord at z=-chamfer and the
      # widened chord at z=0, closed by the outer edge.
      parts.append(_hull_band(i0, i1, c0, c1, o0, o1, -chamfer, 0.0))
    else:
      parts.append(_prism(np.array([i0, o0, o1, i1]), -depth - floor, 0.0))
  if floor > 0.0:
    # A square slab under the bore; its corners run into the sectors, which is
    # harmless for static geometry and cheaper than a round floor.
    h = 1.05 * bore_radius
    square = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    parts.append(_prism(square, -depth - floor, -depth))
  return parts


def _hull_band(i0, i1, c0, c1, o0, o1, z0, z1) -> Mesh:
  """Convex piece between the bore chord at z0 and the chamfer chord at z1."""
  verts = np.array([
    [*i0, z0], [*o0, z0], [*o1, z0], [*i1, z0],
    [*c0, z1], [*o0, z1], [*o1, z1], [*c1, z1],
  ])
  faces = [
    (0, 2, 1), (0, 3, 2),  # bottom
    (4, 5, 6), (4, 6, 7),  # top
    (0, 1, 5), (0, 5, 4),  # side a0
    (3, 7, 6), (3, 6, 2),  # side a1
    (1, 2, 6), (1, 6, 5),  # outer
    (0, 4, 7), (0, 7, 3),  # chamfer face
  ]
  return Mesh(verts, np.asarray(faces, dtype=np.int32))


def signed_volume(m: Mesh) -> float:
  """Volume by the divergence theorem; positive for outward-oriented meshes."""
  v = m.vertices[m.faces]
  return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
