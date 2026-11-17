"""Franka Panda assembly tasks, with Factory's part dimensions.

===========================  ===========================================================
task id                      part
===========================  ===========================================================
Mjlab-RodInsert-8mm-Franka   7.986 mm round peg into a 9 mm hole, 0.5/1 mm chamfers
Mjlab-PegInsert-Rect-Franka  12 x 8 mm rectangular peg, socket yawed up to 30 degrees
Mjlab-GearMesh-Franka        40-tooth gear onto its shaft, between two fixed gears
===========================  ===========================================================

All hold the part in the gripper and act through task-space impedance with fixed
stiffness. Other parts, a welded part or policy-set stiffness:
``env_cfgs.franka_peg_insertion_env_cfg(geometry=..., grasped=..., control=...)``.
"""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_assembly.asset_zoo.workpieces import assemblies

from .env_cfgs import franka_peg_insertion_env_cfg
from .rl_cfg import franka_peg_insertion_ppo_runner_cfg

# Factory/NIST reference parts, built with mjfit.
_ROD_8MM = assemblies.rod_insert()
_PEG_RECT = assemblies.peg_insert()
_GEAR = assemblies.GearMeshCfg()

_VARIANTS = {
  "Mjlab-RodInsert-8mm-Franka": dict(geometry=_ROD_8MM),
  "Mjlab-PegInsert-Rect-Franka": dict(geometry=_PEG_RECT),
  "Mjlab-GearMesh-Franka": dict(geometry=_GEAR),
}

for _task_id, _kw in _VARIANTS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=franka_peg_insertion_env_cfg(**_kw),
    play_env_cfg=franka_peg_insertion_env_cfg(**_kw, play=True),
    rl_cfg=franka_peg_insertion_ppo_runner_cfg(
      experiment_name=_task_id.lower().replace("mjlab-", "").replace("-", "_")),
  )
