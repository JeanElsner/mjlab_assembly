import numpy as np
import pytest
from mjlab_assembly.asset_zoo.workpieces import geometry as g


def _watertight_oriented(m: g.Mesh) -> bool:
  """Every directed edge appears once and its reverse once."""
  edges = {}
  for a, b, c in m.faces:
    for e in ((a, b), (b, c), (c, a)):
      edges[e] = edges.get(e, 0) + 1
  return all(v == 1 and edges.get((e[1], e[0]), 0) == 1 for e, v in edges.items())


R, L, N = 0.006, 0.05, 64


@pytest.mark.parametrize("chamfer", [0.0, 0.0005])
def test_peg_closed_and_volume(chamfer):
  m = g.peg_mesh(R, L, chamfer, N)
  assert _watertight_oriented(m)
  ideal = np.pi * R**2 * L
  assert g.signed_volume(m) == pytest.approx(ideal, rel=0.01)


@pytest.mark.parametrize("floor", [0.0, 0.005])
@pytest.mark.parametrize("chamfer", [0.0, 0.0005])
def test_socket_closed_and_volume(floor, chamfer):
  rb, ro, depth = R + 0.0005, 0.02, 0.025
  m = g.socket_mesh(rb, ro, depth, floor, chamfer, N)
  assert _watertight_oriented(m)
  ideal = np.pi * (ro**2 - rb**2) * depth + np.pi * ro**2 * floor
  assert g.signed_volume(m) == pytest.approx(ideal, rel=0.02)


@pytest.mark.parametrize("chamfer", [0.0, 0.0005])
def test_sectors_closed_positive_and_cover_the_socket(chamfer):
  rb, ro, depth = R + 0.0005, 0.02, 0.025
  parts = g.socket_sectors(rb, ro, depth, 0.0, chamfer, N)
  for p in parts:
    assert _watertight_oriented(p)
    assert g.signed_volume(p) > 0.0
  total = sum(g.signed_volume(p) for p in parts)
  ideal = g.signed_volume(g.socket_mesh(rb, ro, depth, 0.0, chamfer, N))
  assert total == pytest.approx(ideal, rel=0.01)


def test_sector_bore_is_never_narrower_than_the_round_bore():
  rb = R + 0.0005
  for p in g.socket_sectors(rb, 0.02, 0.025, 0.0, 0.0, N):
    rho = np.hypot(p.vertices[:, 0], p.vertices[:, 1])
    assert rho.min() >= rb * (1 - 1e-12)
