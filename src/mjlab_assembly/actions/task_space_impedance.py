"""Cartesian impedance action term, with fixed or policy-commanded stiffness.

The control law has been verified on a real Franka Panda; its kinematic
null-space projector is the form of panda-py's ``CartesianImpedance``
controller.

The policy moves a reference pose for a site (the TCP) and sets, optionally, the
diagonal stiffness. At the physics rate the term computes joint torques

    tau = J^T [ K_t (x_ref - x) - D_t v ;  K_r e_rot - D_r w ] + N^T u_posture

with D = 2 zeta sqrt(K) per axis (unit-mass critical damping, as on the arm) and
a joint-space posture PD u_posture = kp (q0 - q) - 2 sqrt(kp) qd projected into the
null space of the site Jacobian:

- ``nullspace="kinematic"`` (default): N^T = I - J^T (J J^T)^-1 J, applied to u
  directly, as in panda-py. It needs no mass matrix, so it is what a real arm runs
  without an identified M, and it is the cheaper of the two.
- ``nullspace="dynamic"``: the dynamically consistent projector,
  N^T = I - J^T Lambda J M^-1 with Lambda = (J M^-1 J^T)^-1, applied to M u (seven
  mass-matrix solves per physics step).

No gravity term: the torque-controlled robot sets ``gravcomp`` on its bodies, as
the real arm's torque interface sits on its built-in compensation.

Action layout, clipped to [-1, 1]:

- ``[dx, dy, dz, drx, dry, drz]`` reference increments of at most ``pos_step`` [m]
  and ``rot_step`` [rad] per policy step, about world axes. ``lock_yaw`` drops
  ``drz``, for round parts whose yaw does no work; ``lock_tilt`` drops ``drx`` and
  ``dry`` and holds the tool upright, as Isaac Lab's Factory tasks do. A locked
  axis keeps the orientation the episode started with, under the same stiffness.
- with ``stiffness="variable"``, six more entries mapped log-linearly onto
  ``k_pos_range`` and ``k_rot_range``.

The reference integrates the increments and is leashed to the current pose
(``leash_pos`` on the translation error's norm, ``leash_rot`` on the rotation
error), so the commanded wrench is bounded by K times the leash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_from_matrix,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
  """(n, 3) rotation vectors to (n, 4) unit quaternions (w, x, y, z)."""
  angle = v.norm(dim=-1, keepdim=True)
  half = 0.5 * angle
  scale = torch.where(angle > 1e-8, torch.sin(half) / angle.clamp_min(1e-8),
                      torch.full_like(angle, 0.5))
  return torch.cat([torch.cos(half), v * scale], dim=-1)


def _clip_norm(v: torch.Tensor, limit: float) -> torch.Tensor:
  n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  return v * (n.clamp(max=limit) / n)


@dataclass(kw_only=True)
class TaskSpaceImpedanceActionCfg(ActionTermCfg):
  """Configuration of :class:`TaskSpaceImpedanceAction`. Gains, steps, leash and
  stiffness bounds are those run on the real arm."""

  entity_name: str = "robot"
  site_name: str = "tcp"
  joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
  stiffness: Literal["fixed", "variable"] = "fixed"
  k_pos: float = 400.0
  """Translational stiffness [N/m] of the fixed law."""
  k_rot: float = 30.0
  """Rotational stiffness [N m/rad] of the fixed law."""
  k_pos_range: tuple[float, float] = (50.0, 800.0)
  k_rot_range: tuple[float, float] = (5.0, 30.0)
  """Rotational bound 30 N m/rad: the real arm's yaw rings above 45 under this damping."""
  zeta: float = 1.0
  """Damping ratio of D = 2 zeta sqrt(K)."""
  pos_step: float = 0.01
  rot_step: float = 0.05
  leash_pos: float = 0.025
  leash_rot: float = 0.5
  lock_yaw: bool = False
  lock_tilt: bool = False
  nullspace: Literal["kinematic", "dynamic"] = "kinematic"
  posture_kp: float = 10.0
  """Null-space posture gain; 0 disables the posture term."""

  def build(self, env: ManagerBasedRlEnv) -> TaskSpaceImpedanceAction:
    return TaskSpaceImpedanceAction(self, env)


class TaskSpaceImpedanceAction(ActionTerm):
  cfg: TaskSpaceImpedanceActionCfg

  def __init__(self, cfg: TaskSpaceImpedanceActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    jids, _ = self._entity.find_joints(list(cfg.joint_names), preserve_order=True)
    self._joint_ids = torch.tensor(jids, device=self.device, dtype=torch.long)
    sids, _ = self._entity.find_sites([cfg.site_name])
    self._site_gid = int(self._entity.indexing.site_ids[sids[0]])
    n = env.num_envs
    rot_axes = ([] if cfg.lock_tilt else [0, 1]) + ([] if cfg.lock_yaw else [2])
    self._rot_axes = torch.tensor(rot_axes, device=self.device, dtype=torch.long)
    self._pose_dim = 3 + len(rot_axes)
    self._dim = self._pose_dim + (6 if cfg.stiffness == "variable" else 0)
    self._raw = torch.zeros(n, self._dim, device=self.device)
    self._x_ref = torch.zeros(n, 3, device=self.device)
    self._q_ref = torch.zeros(n, 4, device=self.device)
    self._q_ref[:, 0] = 1.0
    self._needs_ref = torch.ones(n, 1, dtype=torch.bool, device=self.device)
    self._K = torch.empty(n, 6, device=self.device)
    self._K[:, :3] = cfg.k_pos
    self._K[:, 3:] = cfg.k_rot
    lo = [math.log(cfg.k_pos_range[0])] * 3 + [math.log(cfg.k_rot_range[0])] * 3
    hi = [math.log(cfg.k_pos_range[1])] * 3 + [math.log(cfg.k_rot_range[1])] * 3
    self._log_lo = torch.tensor(lo, device=self.device)
    self._log_hi = torch.tensor(hi, device=self.device)
    self._q_posture: torch.Tensor | None = None
    self._jacp = self._jacr = self._wp_body = self._dof = None
    self._wp_x = self._wp_y = None

  @property
  def action_dim(self) -> int:
    return self._dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  @property
  def stiffness(self) -> torch.Tensor:
    """Current diagonal stiffness, (num_envs, 6): three translational, three rotational."""
    return self._K

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._needs_ref[env_ids] = True
    self._raw[env_ids] = 0.0

  def process_actions(self, actions: torch.Tensor) -> None:
    """Per policy step: integrate and leash the reference, map the stiffness."""
    cfg = self.cfg
    self._raw[:] = actions.clamp(-1.0, 1.0)
    pos, quat = self._site_pose()
    self._init_reference(pos, quat)
    self._x_ref = pos + _clip_norm(
      self._x_ref + self._raw[:, :3] * cfg.pos_step - pos, cfg.leash_pos)
    drot = torch.zeros(self.num_envs, 3, device=self.device)
    drot[:, self._rot_axes] = self._raw[:, 3 : self._pose_dim]
    q = quat_mul(_quat_from_rotvec(drot * cfg.rot_step), self._q_ref)
    err = _clip_norm(axis_angle_from_quat(quat_mul(q, quat_inv(quat))), cfg.leash_rot)
    self._q_ref = quat_mul(_quat_from_rotvec(err), quat)
    if cfg.stiffness == "variable":
      u = 0.5 * (self._raw[:, self._pose_dim :] + 1.0)
      self._K = torch.exp(self._log_lo + u * (self._log_hi - self._log_lo))

  def apply_actions(self) -> None:
    """Per physics substep: impedance law to joint torques."""
    cfg = self.cfg
    pos, quat = self._site_pose()
    self._init_reference(pos, quat)
    J = self._jacobian(pos)  # (n, 6, nj)
    qd = self._entity.data.joint_vel[:, self._joint_ids]
    twist = torch.einsum("nij,nj->ni", J, qd)  # site velocity, [linear, angular]
    e = torch.cat([self._x_ref - pos,
                   axis_angle_from_quat(quat_mul(self._q_ref, quat_inv(quat)))], -1)
    D = 2.0 * cfg.zeta * torch.sqrt(self._K)
    tau = torch.einsum("nij,ni->nj", J, self._K * e - D * twist)
    if cfg.posture_kp > 0.0:
      tau = tau + self._posture_torque(J, qd)
    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  # Internals.

  def _init_reference(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
    # No branch on the mask: a .any() here would synchronize with the GPU.
    self._x_ref = torch.where(self._needs_ref, pos, self._x_ref)
    self._q_ref = torch.where(self._needs_ref, quat, self._q_ref)
    self._needs_ref.zero_()

  def _site_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    data = self._env.sim.data
    quat = quat_from_matrix(data.site_xmat[:, self._site_gid].reshape(-1, 3, 3))
    return data.site_xpos[:, self._site_gid], quat

  def _jacobian(self, point_w: torch.Tensor) -> torch.Tensor:
    sim = self._env.sim
    if self._jacp is None:
      n, nv, dev = self.num_envs, sim.mj_model.nv, str(self.device)
      self._jacp = wp.zeros((n, 3, nv), dtype=wp.float32, device=dev)
      self._jacr = wp.zeros((n, 3, nv), dtype=wp.float32, device=dev)
      body = int(sim.mj_model.site_bodyid[self._site_gid])
      self._wp_body = wp.array([body] * n, dtype=wp.int32, device=dev)
      self._dof = self._entity.indexing.joint_v_adr[self._joint_ids]
    point = wp.from_torch(point_w.contiguous(), dtype=wp.vec3)
    mjwarp.jac(sim.model, sim.data, self._jacp, self._jacr, point, self._wp_body)
    jp = wp.to_torch(self._jacp)[:, :, self._dof]
    jr = wp.to_torch(self._jacr)[:, :, self._dof]
    return torch.cat([jp, jr], dim=1)

  def _mass_op(self, vec: torch.Tensor, solve: bool) -> torch.Tensor:
    """M @ vec or M^-1 @ vec for vectors supported on the arm's dofs."""
    sim = self._env.sim
    if self._wp_x is None:
      shape = (self.num_envs, sim.mj_model.nv)
      self._wp_x = wp.zeros(shape, dtype=wp.float32, device=str(self.device))
      self._wp_y = wp.zeros(shape, dtype=wp.float32, device=str(self.device))
    x, y = wp.to_torch(self._wp_x), wp.to_torch(self._wp_y)
    y.zero_()
    y[:, self._dof] = vec
    if solve:
      mjwarp.solve_m(sim.model, sim.data, self._wp_x, self._wp_y)
    else:
      mjwarp.mul_m(sim.model, sim.data, self._wp_x, self._wp_y)
    return x[:, self._dof].clone()

  def _posture_torque(self, J: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """Posture PD toward the start configuration, projected into null(J)."""
    data = self._entity.data
    q = data.joint_pos[:, self._joint_ids]
    if self._q_posture is None:
      self._q_posture = data.default_joint_pos[:, self._joint_ids].clone()
    kp = self.cfg.posture_kp
    u = kp * (self._q_posture - q) - 2.0 * math.sqrt(kp) * qd
    JT = J.transpose(1, 2)
    eye = torch.eye(JT.shape[1], device=J.device).unsqueeze(0)
    if self.cfg.nullspace == "kinematic":
      N = eye - JT @ torch.linalg.solve(J @ JT, J)
      return torch.einsum("nij,nj->ni", N, u)
    MiJT = torch.stack([self._mass_op(JT[:, :, k], solve=True) for k in range(6)], 2)
    lam = torch.linalg.inv(J @ MiJT)
    N = eye - JT @ (lam @ MiJT.transpose(1, 2))
    return torch.einsum("nij,nj->ni", N, self._mass_op(u, solve=False))
