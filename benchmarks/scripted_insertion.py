"""Scripted insertion: steer the impedance reference to the bore axis, then down.

  uv run python benchmarks/scripted_insertion.py {grasped,welded} {impedance,variable_impedance} [num_envs]

A physics sanity check, not a policy: every environment should end inserted.
"""
import sys

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_assembly.tasks.peg_insertion import mdp
from mjlab_assembly.tasks.peg_insertion.config.franka_panda.env_cfgs import (
    franka_peg_insertion_env_cfg,
)

grasped = sys.argv[1] == "grasped"
control = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
cfg = franka_peg_insertion_env_cfg(grasped=grasped, control=control)
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg, device="cuda:0")
env.reset()
A = env.action_manager.total_action_dim
term = env.action_manager.get_term("impedance")
tip = SceneEntityCfg("peg" if grasped else "robot", site_names=("tip",))
tip.resolve(env.scene)
mouth = SceneEntityCfg("socket", site_names=("mouth",))
mouth.resolve(env.scene)
robot = env.scene["robot"]
env.step(torch.zeros(n, A, device="cuda:0"))  # initializes the reference
for i in range(200):
    p = mdp.tip_in_mouth_frame(env, tip, mouth)
    tcp = robot.data.site_pos_w[:, term._site]
    goal = tcp - p  # TCP target that puts the tip on the mouth centre
    goal[:, 2] += 0.005  # hover 5 mm above the mouth first
    aligned = p[:, :2].norm(dim=-1) < 0.0003
    goal[:, 2] = torch.where(aligned | (p[:, 2] < 0.002), tcp[:, 2] - p[:, 2] - 0.035, goal[:, 2])
    a = torch.zeros(n, A, device="cuda:0")
    a[:, :3] = ((goal - term._x_ref) / term.cfg.pos_step).clamp(-1, 1)
    obs, rew, terminated, trunc, info = env.step(a)
    if i % 50 == 49:
        p = mdp.tip_in_mouth_frame(env, tip, mouth)
        ins = mdp.is_inserted(env, tip, mouth, 0.025)
        print(f"step {i+1}: tip z mean {float(p[:,2].mean())*1e3:.1f} mm, min {float(p[:,2].min())*1e3:.1f} mm, "
              f"lateral max {float(p[:,:2].norm(dim=-1).max())*1e3:.2f} mm, inserted {float(ins.float().mean()):.3f}, "
              f"terminated {int(terminated.sum())}, nan {bool(torch.isnan(obs['actor']).any())}", flush=True)
