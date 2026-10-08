"""Franka Panda peg-insertion tasks.

=========================================  ========  =====================================
task id                                    peg       action space
=========================================  ========  =====================================
Mjlab-PegInsert-Franka                     grasped   task-space impedance, fixed stiffness
Mjlab-PegInsert-Franka-Vic                 grasped   task-space impedance, policy stiffness
Mjlab-PegInsert-Franka-Welded              welded    task-space impedance, fixed stiffness
Mjlab-PegInsert-Franka-Welded-Vic          welded    task-space impedance, policy stiffness
=========================================  ========  =====================================

All use a 12 mm peg in a bore with 0.5 mm radial clearance. Other geometries:
``env_cfgs.franka_peg_insertion_env_cfg(geometry=PegHoleCfg(clearance=...))``.
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import franka_peg_insertion_env_cfg
from .rl_cfg import franka_peg_insertion_ppo_runner_cfg

_VARIANTS = {
  "Mjlab-PegInsert-Franka": dict(grasped=True, control="impedance"),
  "Mjlab-PegInsert-Franka-Vic": dict(grasped=True, control="variable_impedance"),
  "Mjlab-PegInsert-Franka-Welded": dict(grasped=False, control="impedance"),
  "Mjlab-PegInsert-Franka-Welded-Vic": dict(grasped=False, control="variable_impedance"),
}

for _task_id, _kw in _VARIANTS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=franka_peg_insertion_env_cfg(**_kw),
    play_env_cfg=franka_peg_insertion_env_cfg(**_kw, play=True),
    rl_cfg=franka_peg_insertion_ppo_runner_cfg(
      experiment_name=_task_id.lower().replace("mjlab-", "").replace("-", "_")),
  )
