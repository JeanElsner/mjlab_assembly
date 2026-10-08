"""Peg-in-hole insertion: the robot-agnostic task configuration.

The robot starts holding a round peg a few centimetres above a socket whose
position is randomized per episode; the policy must insert the peg to the
bottom of the bore. Robot-specific pieces (the entity, the action term, the
site and joint names) are filled in per robot, see ``config/``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as mjlab_events
from mjlab.envs.mdp import observations as mjlab_obs
from mjlab.envs.mdp import rewards as mjlab_rewards
from mjlab.envs.mdp import terminations as mjlab_terms
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_assembly.asset_zoo.workpieces.peg_hole import (
  MOUTH_SITE,
  TIP_SITE,
  PegHoleCfg,
)
from mjlab_assembly.tasks.peg_insertion import mdp


def make_peg_insertion_env_cfg(
  geometry: PegHoleCfg,
  peg_entity: str,
  socket_xy_range: float = 0.02,
  obs_noise: float = 0.001,
) -> ManagerBasedRlEnvCfg:
  """Base configuration. ``peg_entity`` names the entity that carries the tip site:
  "peg" for a held peg, "robot" for a peg welded to the hand."""
  tip = SceneEntityCfg(peg_entity, site_names=(TIP_SITE,))
  mouth = SceneEntityCfg("socket", site_names=(MOUTH_SITE,))
  geo = {"tip_cfg": tip, "mouth_cfg": mouth}
  depth = geometry.hole_depth
  # Seated: the tip within 5 % of the bore depth of the floor, on the axis to within
  # the clearance plus half a millimetre.
  success = {**geo, "hole_depth": depth, "depth_fraction": 0.95,
             "lateral_tol": geometry.clearance + 0.0005}

  actor = {
    "joint_pos": ObservationTermCfg(
      func=mjlab_obs.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per robot.
    ),
    "joint_vel": ObservationTermCfg(
      func=mjlab_obs.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05),
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per robot.
    ),
    "tip_in_mouth": ObservationTermCfg(
      func=mdp.tip_in_mouth_frame, params=geo,
      noise=Unoise(n_min=-obs_noise, n_max=obs_noise),
    ),
    "peg_axis": ObservationTermCfg(func=mdp.peg_axis_in_mouth_frame, params=geo),
    "actions": ObservationTermCfg(func=mjlab_obs.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor, enable_corruption=True),
    "critic": ObservationGroupCfg(dict(actor), enable_corruption=False),
  }

  rewards = {
    "keypoint_baseline": RewardTermCfg(
      func=mdp.keypoint_squash, weight=1.0, params={**geo, "hole_depth": depth, "a": 5.0, "b": 4.0}),
    "keypoint_coarse": RewardTermCfg(
      func=mdp.keypoint_squash, weight=1.0, params={**geo, "hole_depth": depth, "a": 50.0, "b": 2.0}),
    "keypoint_fine": RewardTermCfg(
      func=mdp.keypoint_squash, weight=1.0, params={**geo, "hole_depth": depth, "a": 100.0, "b": 0.0}),
    "lateral_alignment": RewardTermCfg(
      func=mdp.lateral_alignment, weight=1.0, params={**geo, "std": 0.002}),
    # The episode ends on success, so this pays once. mjlab scales rewards by the
    # 20 ms step, making the bonus 250 * 0.02 = 5; staying inserted instead would be
    # worth about 0.92 * 0.02 / (1 - 0.99) = 1.8, so finishing is always preferred.
    "success": RewardTermCfg(func=mdp.success_bonus, weight=250.0, params=success),
    "action_rate_l2": RewardTermCfg(func=mjlab_rewards.action_rate_l2, weight=-0.01),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mjlab_terms.time_out, time_out=True),
    "success": TerminationTermCfg(func=mdp.is_inserted, params=success),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mjlab_events.reset_root_state_uniform, mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mjlab_events.reset_joints_by_offset, mode="reset",
      params={
        "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_socket": EventTermCfg(
      func=mjlab_events.reset_root_state_uniform, mode="reset",
      params={
        "pose_range": {"x": (-socket_xy_range, socket_xy_range),
                       "y": (-socket_xy_range, socket_xy_range)},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("socket"),
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                   env_spacing=1.0),
    observations=observations,
    actions={},  # Set per robot.
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="socket",
      body_name="socket", distance=0.6, elevation=-20.0, azimuth=140.0,
    ),
    sim=SimulationCfg(
      nconmax=128,
      njmax=1500,
      # CG, not Newton: MuJoCo Warp 3.11's Newton solver clamps t^3 in the
      # elliptic-cone Hessian, which turns it indefinite for a pressed contact
      # that barely slides (t < 1e-5) and returns NaN. Fixed upstream in MuJoCo
      # Warp 3.15 (google-deepmind/mujoco_warp#1657); mjlab 1.6 pins 3.11.
      # CG converges more slowly: at 10 iterations the grasp lets the peg slip
      # out, at 30 and more it holds.
      mujoco=MujocoCfg(timestep=0.002, iterations=50, ls_iterations=20,
                       impratio=10, cone="elliptic", solver="cg"),
    ),
    decimation=10,  # 50 Hz policy over 500 Hz physics
    episode_length_s=5.0,
  )
