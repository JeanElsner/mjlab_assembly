"""MDP terms for peg insertion.

Geometry is read from two sites: the peg's ``tip`` (whose +z is the peg axis,
pointing from the tip to the top) and the socket's ``mouth`` (whose +z is the
bore axis, pointing out of the bore). The seated pose puts the tip
``hole_depth`` below the mouth, along the bore axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_matrix, quat_inv, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_Z = (0.0, 0.0, 1.0)
_SITE_GID: dict[tuple[int, str, tuple[str, ...]], int] = {}


def _site_index(cfg: SceneEntityCfg):
  ids = cfg.site_ids
  return ids[0] if isinstance(ids, (list, tuple)) else ids


def _site_pose(env: ManagerBasedRlEnv, cfg: SceneEntityCfg):
  """World pose of one site, read straight from the simulator.

  The entity's ``site_pos_w`` / ``site_quat_w`` convert every site of the entity
  on each access, which dominated the per-step cost of these terms.
  """
  key = (id(env), cfg.name, tuple(cfg.site_names or ()))
  gid = _SITE_GID.get(key)
  if gid is None:  # resolved by name once; int() of a device tensor synchronizes
    entity = env.scene[cfg.name]
    local = entity.find_sites(list(cfg.site_names))[0][0]
    gid = _SITE_GID[key] = int(entity.indexing.site_ids[local])
  data = env.sim.data
  quat = quat_from_matrix(data.site_xmat[:, gid].reshape(-1, 3, 3))
  return data.site_xpos[:, gid], quat


def _axis(quat: torch.Tensor) -> torch.Tensor:
  z = torch.tensor(_Z, device=quat.device).expand(quat.shape[0], 3)
  return quat_apply(quat, z)


def tip_in_mouth_frame(env, tip_cfg: SceneEntityCfg, mouth_cfg: SceneEntityCfg):
  """Peg tip position in the socket mouth frame [m]; z < 0 is inside the bore."""
  tip, _ = _site_pose(env, tip_cfg)
  mouth, mq = _site_pose(env, mouth_cfg)
  return quat_apply(quat_inv(mq), tip - mouth)


def peg_axis_in_mouth_frame(env, tip_cfg: SceneEntityCfg, mouth_cfg: SceneEntityCfg):
  """Peg axis (tip to top) in the mouth frame; (0, 0, 1) when aligned with the bore."""
  _, tq = _site_pose(env, tip_cfg)
  _, mq = _site_pose(env, mouth_cfg)
  return _axis(quat_mul(quat_inv(mq), tq))


def _keypoint_distance(env, tip_cfg, mouth_cfg, hole_depth: float, span: float, n: int):
  """Mean distance between peg keypoints and their seated positions.

  Keypoints lie on the peg axis, from the tip up over ``span``; their targets lie
  on the bore axis, from the seated tip position up over the same span.
  """
  tip, tq = _site_pose(env, tip_cfg)
  mouth, mq = _site_pose(env, mouth_cfg)
  a_peg, a_hole = _axis(tq), _axis(mq)
  goal = mouth - hole_depth * a_hole
  s = torch.linspace(0.0, span, n, device=tip.device).view(1, n, 1)
  kp = tip.unsqueeze(1) + s * a_peg.unsqueeze(1)
  kg = goal.unsqueeze(1) + s * a_hole.unsqueeze(1)
  return (kp - kg).norm(dim=-1).mean(dim=-1)


def keypoint_squash(
  env: ManagerBasedRlEnv,
  tip_cfg: SceneEntityCfg,
  mouth_cfg: SceneEntityCfg,
  hole_depth: float,
  a: float,
  b: float,
  span: float = 0.025,
  num_keypoints: int = 4,
) -> torch.Tensor:
  """Factory's squashed keypoint reward, 1 / (exp(a d) + b + exp(-a d)).

  Narrowing ``a`` at fixed ``b`` sharpens the reward around the seated pose; the
  Factory and FORGE recipes sum a baseline (a=5, b=4), a coarse (a=50, b=2) and a
  fine (a=100, b=0) term.
  """
  d = _keypoint_distance(env, tip_cfg, mouth_cfg, hole_depth, span, num_keypoints)
  return 1.0 / (torch.exp(a * d) + b + torch.exp(-a * d))


def lateral_alignment(
  env: ManagerBasedRlEnv,
  tip_cfg: SceneEntityCfg,
  mouth_cfg: SceneEntityCfg,
  std: float = 0.002,
  height: float = 0.01,
) -> torch.Tensor:
  """1 - tanh(r / std) for the tip's distance r from the bore axis, while the tip is
  within ``height`` above the mouth or inside the bore.

  The keypoint reward is dominated by depth once the peg rests on the socket's top
  face, so a policy can settle a few millimetres beside the hole; this term keeps
  a gradient toward the axis there.
  """
  p = tip_in_mouth_frame(env, tip_cfg, mouth_cfg)
  near = p[:, 2] <= height
  return near.float() * (1.0 - torch.tanh(p[:, :2].norm(dim=-1) / std))


def is_inserted(
  env: ManagerBasedRlEnv,
  tip_cfg: SceneEntityCfg,
  mouth_cfg: SceneEntityCfg,
  hole_depth: float,
  depth_fraction: float = 0.9,
  lateral_tol: float = 0.002,
) -> torch.Tensor:
  """Tip at least ``depth_fraction`` of the way down and within ``lateral_tol`` of the axis."""
  p = tip_in_mouth_frame(env, tip_cfg, mouth_cfg)
  return (-p[:, 2] >= depth_fraction * hole_depth) & (p[:, :2].norm(dim=-1) <= lateral_tol)


def success_bonus(env, tip_cfg, mouth_cfg, hole_depth: float, **kw) -> torch.Tensor:
  return is_inserted(env, tip_cfg, mouth_cfg, hole_depth, **kw).float()


def peg_lost(
  env: ManagerBasedRlEnv,
  peg_cfg: SceneEntityCfg,
  tcp_cfg: SceneEntityCfg,
  max_offset: float = 0.02,
) -> torch.Tensor:
  """The held peg has moved more than ``max_offset`` from the TCP."""
  peg = env.scene[peg_cfg.name].data.root_link_pos_w
  tcp, _ = _site_pose(env, tcp_cfg)
  return (peg - tcp).norm(dim=-1) > max_offset


def reset_peg_in_grasp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  peg_cfg: SceneEntityCfg,
  tcp_cfg: SceneEntityCfg,
  finger_cfg: SceneEntityCfg,
  half_width: float,
  axial_noise: float = 0.003,
) -> None:
  """Put the free peg between the jaws, its grasp point at the TCP.

  Runs after the robot's joint reset, so the kinematics are refreshed first. The
  jaws are placed at the peg's half width, so the passive grip spring is already
  loaded and the grasp holds from the first physics step. ``axial_noise`` slides
  the peg along its own axis, i.e. how deep it sits in the hand.
  """
  env.sim.forward()
  robot = env.scene[tcp_cfg.name]
  peg = env.scene[peg_cfg.name]
  i = _site_index(tcp_cfg)
  tcp = robot.data.site_pos_w[env_ids, i]
  tq = robot.data.site_quat_w[env_ids, i]
  n = len(env_ids)
  # Peg frame = TCP frame rotated 180 degrees about x (peg +z points into the hand).
  flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=tcp.device).expand(n, 4)
  quat = quat_mul(tq, flip)
  shift = (2.0 * torch.rand(n, 1, device=tcp.device) - 1.0) * axial_noise
  pos = tcp + shift * _axis(quat)
  state = torch.zeros(n, 13, device=tcp.device)
  state[:, :3], state[:, 3:7] = pos, quat
  peg.write_root_state_to_sim(state, env_ids)
  jaw = torch.full((n, len(finger_cfg.joint_ids)), half_width, device=tcp.device)
  robot.write_joint_state_to_sim(
    jaw, torch.zeros_like(jaw), joint_ids=finger_cfg.joint_ids, env_ids=env_ids
  )
