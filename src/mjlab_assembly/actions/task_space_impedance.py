"""Task-space impedance action term, with fixed or policy-commanded stiffness.

The policy moves a reference pose for a site on the robot (the TCP by default),
and a Cartesian impedance law turns the pose error into joint torques at the
physics rate:

    w   = K (x_ref - x) - D v                  (wrench at the site, 6-D)
    tau = J^T w + N^T M u_posture               (u_posture: joint-space PD)

N is the dynamically consistent null-space projector, which keeps the arm's
posture without disturbing the site. The damping is critical per axis with
respect to the task-space inertia Lambda = (J M^-1 J^T)^-1 at the current pose,
D = 2 zeta sqrt(K diag(Lambda)), so the response is as damped as configured
whatever the arm's configuration; ``damping="unit_mass"`` uses D = 2 zeta sqrt(K)
instead (a common simplification that underdamps an arm of several kilograms). Gravity
is not in the law: the torque-controlled Panda entity sets ``gravcomp`` on every
body, as the real arm's torque interface sits on its built-in compensation.

Action layout, each entry clipped to [-1, 1]:

- ``[dx, dy, dz, drx, dry, drz]`` reference increments, scaled by ``pos_step`` [m]
  and ``rot_step`` [rad] per policy step. With ``lock_yaw`` the ``drz`` entry is
  omitted (5-D), useful for round parts whose yaw does no work.
- With ``stiffness="variable"``, six more entries set the diagonal stiffness,
  mapped log-linearly onto ``k_pos_range`` and ``k_rot_range``.

The reference is leashed: it never runs further than ``leash_pos`` / ``leash_rot``
from the current pose, so the commanded wrench is bounded by K times the leash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_inv, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
  """(n, 3) rotation vectors to (n, 4) unit quaternions (w, x, y, z)."""
  angle = v.norm(dim=-1, keepdim=True)
  half = 0.5 * angle
  # sin(half) / angle, with its limit 1/2 at zero.
  scale = torch.where(angle > 1e-8, torch.sin(half) / angle.clamp_min(1e-8),
                      torch.full_like(angle, 0.5))
  return torch.cat([torch.cos(half), v * scale], dim=-1)


def _clip_norm(v: torch.Tensor, limit: float) -> torch.Tensor:
  n = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  return v * (n.clamp(max=limit) / n)


@dataclass(kw_only=True)
class TaskSpaceImpedanceActionCfg(ActionTermCfg):
  """Configuration of :class:`TaskSpaceImpedanceAction`."""

  entity_name: str = "robot"
  site_name: str = "tcp"
  joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
  stiffness: Literal["fixed", "variable"] = "fixed"
  k_pos: float = 400.0
  """Translational stiffness [N/m] when ``stiffness="fixed"``."""
  k_rot: float = 30.0
  """Rotational stiffness [N m/rad] when ``stiffness="fixed"``."""
  k_pos_range: tuple[float, float] = (100.0, 1000.0)
  k_rot_range: tuple[float, float] = (5.0, 50.0)
  zeta: float = 1.0
  """Damping ratio, per axis."""
  damping: Literal["inertia", "unit_mass"] = "inertia"
  """Critical damping with respect to the task-space inertia, or to a unit mass."""
  pos_step: float = 0.01
  rot_step: float = 0.05
  leash_pos: float = 0.025
  leash_rot: float = 0.25
  lock_yaw: bool = False
  posture_kp: float = 10.0
  """Null-space posture stiffness [1/s^2]; 0 disables the posture term."""

  def build(self, env: ManagerBasedRlEnv) -> TaskSpaceImpedanceAction:
    return TaskSpaceImpedanceAction(self, env)


class TaskSpaceImpedanceAction(ActionTerm):
  cfg: TaskSpaceImpedanceActionCfg

  def __init__(self, cfg: TaskSpaceImpedanceActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    jids, _ = self._entity.find_joints(list(cfg.joint_names), preserve_order=True)
    self._joint_ids = torch.tensor(jids, device=self.device, dtype=torch.long)
    sids, _ = self._entity.find_sites([cfg.site_name])
    self._site = sids[0]
    n = env.num_envs
    self._pose_dim = 5 if cfg.lock_yaw else 6
    self._dim = self._pose_dim + (6 if cfg.stiffness == "variable" else 0)
    self._raw = torch.zeros(n, self._dim, device=self.device)
    self._x_ref = torch.zeros(n, 3, device=self.device)
    self._q_ref = torch.zeros(n, 4, device=self.device)
    self._q_ref[:, 0] = 1.0
    self._needs_ref = torch.ones(n, dtype=torch.bool, device=self.device)
    self._K = torch.empty(n, 6, device=self.device)
    self._K[:, :3] = cfg.k_pos
    self._K[:, 3:] = cfg.k_rot
    lo = torch.tensor([math.log(cfg.k_pos_range[0])] * 3 + [math.log(cfg.k_rot_range[0])] * 3)
    hi = torch.tensor([math.log(cfg.k_pos_range[1])] * 3 + [math.log(cfg.k_rot_range[1])] * 3)
    self._log_lo, self._log_hi = lo.to(self.device), hi.to(self.device)
    self._q_posture: torch.Tensor | None = None
    # Lazily allocated warp scratch for the Jacobian and mass-matrix solves.
    self._jacp = self._jacr = self._wp_body = None
    self._dof = None
    self._wp_x = self._wp_y = None

  # ActionTerm interface.

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
    cfg = self.cfg
    self._raw[:] = actions.clamp(-1.0, 1.0)
    pos, quat, _, _ = self._site_state()
    self._init_reference(pos, quat)
    # Translation: integrate, then leash to the current position.
    self._x_ref = pos + _clip_norm(
      self._x_ref + self._raw[:, :3] * cfg.pos_step - pos, cfg.leash_pos)
    # Rotation: world-frame increment, then leash the orientation error.
    drot = torch.zeros(self.num_envs, 3, device=self.device)
    drot[:, : self._pose_dim - 3] = self._raw[:, 3 : self._pose_dim]
    q = quat_mul(_quat_from_rotvec(drot * cfg.rot_step), self._q_ref)
    err = _clip_norm(axis_angle_from_quat(quat_mul(q, quat_inv(quat))), cfg.leash_rot)
    self._q_ref = quat_mul(_quat_from_rotvec(err), quat)
    if cfg.stiffness == "variable":
      u = 0.5 * (self._raw[:, self._pose_dim :] + 1.0)
      self._K = torch.exp(self._log_lo + u * (self._log_hi - self._log_lo))

  def apply_actions(self) -> None:
    cfg = self.cfg
    pos, quat, v_lin, v_ang = self._site_state()
    self._init_reference(pos, quat)
    e_pos = self._x_ref - pos
    e_rot = axis_angle_from_quat(quat_mul(self._q_ref, quat_inv(quat)))
    J = self._jacobian(pos)  # (n, 6, nj)
    need_lambda = cfg.damping == "inertia" or cfg.posture_kp > 0.0
    if need_lambda:
      JT = J.transpose(1, 2)
      MiJT = torch.stack([self._mass_op(JT[:, :, k], solve=True) for k in range(6)], 2)
      lam = torch.linalg.inv(J @ MiJT + 1e-6 * torch.eye(6, device=J.device))
    if cfg.damping == "inertia":
      m_eff = torch.diagonal(lam, dim1=-2, dim2=-1).clamp_min(1e-6)
      D = 2.0 * cfg.zeta * torch.sqrt(self._K * m_eff)
    else:
      D = 2.0 * cfg.zeta * torch.sqrt(self._K)
    wrench = self._K * torch.cat([e_pos, e_rot], -1) - D * torch.cat([v_lin, v_ang], -1)
    tau = torch.einsum("nij,ni->nj", J, wrench)
    if cfg.posture_kp > 0.0:
      tau = tau + self._posture_torque(JT, MiJT, lam)
    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  # Internals.

  def _init_reference(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
    m = self._needs_ref
    if m.any():
      self._x_ref[m] = pos[m]
      self._q_ref[m] = quat[m]
      self._needs_ref[m] = False

  def _site_state(self):
    d = self._entity.data
    vel = d.site_vel_w[:, self._site]
    return d.site_pos_w[:, self._site], d.site_quat_w[:, self._site], vel[:, :3], vel[:, 3:]

  def _jacobian(self, point_w: torch.Tensor) -> torch.Tensor:
    sim = self._env.sim
    if self._jacp is None:
      n, nv = self.num_envs, sim.mj_model.nv
      self._jacp = wp.zeros((n, 3, nv), dtype=wp.float32, device=str(self.device))
      self._jacr = wp.zeros((n, 3, nv), dtype=wp.float32, device=str(self.device))
      site_id = int(self._entity.indexing.site_ids[self._site])
      body = int(sim.mj_model.site_bodyid[site_id])
      self._wp_body = wp.array([body] * n, dtype=wp.int32, device=str(self.device))
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

  def _posture_torque(self, JT: torch.Tensor, MiJT: torch.Tensor,
                      lam: torch.Tensor) -> torch.Tensor:
    """N^T M (kp (q0 - q) - 2 sqrt(kp) qd), N the dynamically consistent projector."""
    data = self._entity.data
    q = data.joint_pos[:, self._joint_ids]
    qd = data.joint_vel[:, self._joint_ids]
    if self._q_posture is None:
      self._q_posture = data.default_joint_pos[:, self._joint_ids].clone()
    kp = self.cfg.posture_kp
    u = kp * (self._q_posture - q) - 2.0 * math.sqrt(kp) * qd
    # N^T = I - J^T Lambda J M^-1, applied to M u.
    Mu = self._mass_op(u, solve=False)
    return Mu - torch.einsum("nij,nj->ni", JT @ (lam @ MiJT.transpose(1, 2)), Mu)
