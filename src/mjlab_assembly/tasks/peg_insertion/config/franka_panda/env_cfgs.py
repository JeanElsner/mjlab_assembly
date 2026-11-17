"""Peg insertion with the Franka Panda."""

from __future__ import annotations

from typing import Literal

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_assembly.actions.task_space_impedance import TaskSpaceImpedanceActionCfg
from mjlab_assembly.asset_zoo.robots.franka_panda import panda_constants as panda
from mjlab_assembly.asset_zoo.workpieces import peg_hole
from mjlab_assembly.tasks.peg_insertion import mdp
from mjlab_assembly.tasks.peg_insertion.peg_insertion_env_cfg import (
  make_peg_insertion_env_cfg,
)

Control = Literal["impedance", "variable_impedance"]

# The socket stands in front of the robot, its mouth a few centimetres below the
# peg tip at the home pose (TCP at about x = 0.485 m, z = 0.136 m).
SOCKET_POS = (0.485, 0.0, 0.0005)


def franka_peg_insertion_env_cfg(
  geometry=None,
  grasped: bool = True,
  control: Control = "impedance",
  upright: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka Panda peg insertion.

  Args:
    geometry: The assembly: ``peg_hole.PegHoleCfg`` or one of ``assemblies``
      (rod, rectangular peg, gear mesh); the default is a 12 mm peg in a bore
      with 0.5 mm radial clearance.
    grasped: Hold the peg in the gripper (a free body squeezed by the jaws);
      False welds it to the hand.
    control: Task-space impedance with fixed or policy-set stiffness.
    upright: Hold the tool upright (no roll or pitch actions), as Factory does.
      Without it, a policy may tilt a rigidly held peg into the chamfer and jam
      it there.
    play: Evaluation settings (no observation noise, long episodes).
  """
  geo = geometry or peg_hole.PegHoleCfg()
  peg_entity = "peg" if grasped else "robot"
  cfg = make_peg_insertion_env_cfg(geo, peg_entity=peg_entity)

  width = geo.grip_width
  # A welded peg sits on a bare flange: no hand, no fingers.
  robot = panda.get_panda_cfg(grip_width=width, hand=grasped)
  if grasped:
    # Jaws start at the peg's half width, so the grip spring is loaded at t = 0.
    robot.init_state.joint_pos["finger_joint.*"] = width / 2.0
  else:
    base_spec_fn = robot.spec_fn

    def welded_spec() -> mujoco.MjSpec:
      spec = base_spec_fn()
      geo.weld_to_flange(spec, panda.TCP_OFFSET)
      return spec

    robot.spec_fn = welded_spec

  cfg.scene.entities = {
    "robot": robot,
    "socket": EntityCfg(
      init_state=EntityCfg.InitialStateCfg(pos=SOCKET_POS),
      spec_fn=geo.socket_spec,
    ),
  }
  if grasped:
    cfg.scene.entities["peg"] = EntityCfg(spec_fn=geo.part_spec)

  arm = SceneEntityCfg("robot", joint_names=panda.ARM_JOINT_NAMES)
  for name in ("joint_pos", "joint_vel"):
    cfg.observations["actor"].terms[name].params["asset_cfg"] = arm
    cfg.observations["critic"].terms[name].params["asset_cfg"] = arm

  cfg.actions = {
    "impedance": TaskSpaceImpedanceActionCfg(
      entity_name="robot", site_name=panda.TCP_SITE,
      joint_names=panda.ARM_JOINT_NAMES,
      stiffness="variable" if control == "variable_impedance" else "fixed",
      lock_yaw=geo.symmetry == 0,  # round parts: yaw does no work
      lock_tilt=upright,
    )
  }

  tcp = SceneEntityCfg("robot", site_names=(panda.TCP_SITE,))
  if grasped:
    cfg.events["reset_peg_in_grasp"] = EventTermCfg(
      func=mdp.reset_peg_in_grasp, mode="reset",
      params={
        "peg_cfg": SceneEntityCfg("peg"), "tcp_cfg": tcp,
        "finger_cfg": SceneEntityCfg("robot", joint_names=panda.FINGER_JOINT_NAMES),
        "half_width": width / 2.0,
        "yaw_noise": geo.grasp_yaw_noise,
      },
    )
    cfg.terminations["peg_lost"] = TerminationTermCfg(
      func=mdp.peg_lost, params={"peg_cfg": SceneEntityCfg("peg"), "tcp_cfg": tcp},
    )

  cfg.scene.num_envs = 4096
  if play:
    cfg.scene.num_envs = 16
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg
