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
seated = torch.zeros(n, dtype=torch.bool, device="cuda:0")  # ever ended by success
first = torch.full((n,), float("nan"), device="cuda:0")
p_prev = mdp.tip_in_mouth_frame(env, tip, mouth)
for i in range(200):
    p = mdp.tip_in_mouth_frame(env, tip, mouth)
    tcp = env.sim.data.site_xpos[:, term._site_gid]
    goal = tcp - p  # TCP target that puts the tip on the mouth centre
    goal[:, 2] += 0.005  # hover 5 mm above the mouth first
    # Descend only once the tip sits on the axis and has stopped swinging: under
    # unit-mass damping (as on the real arm) the TCP overshoots before it settles.
    v = (p - p_prev) / 0.02
    p_prev = p.clone()
    aligned = (p[:, :2].norm(dim=-1) < 0.0004) & (v[:, :2].norm(dim=-1) < 0.01)
    goal[:, 2] = torch.where(aligned | (p[:, 2] < 0.002), tcp[:, 2] - p[:, 2] - 0.035, goal[:, 2])
    a = torch.zeros(n, A, device="cuda:0")
    a[:, :3] = ((goal - term._x_ref) / term.cfg.pos_step).clamp(-1, 1)
    obs, rew, terminated, trunc, info = env.step(a)
    done = env.termination_manager.get_term("success").bool() & ~seated
    first[done] = (i + 1) * 0.02
    seated |= done
    if i % 50 == 49:
        p = mdp.tip_in_mouth_frame(env, tip, mouth)
        ins = mdp.is_inserted(env, tip, mouth, 0.025)
        print(f"step {i+1}: seated (episode ended by success) {float(seated.float().mean()):.3f}, "
              f"median time {float(first[seated].median()) if seated.any() else float('nan'):.2f} s, "
              f"nan {bool(torch.isnan(obs['actor']).any())}", flush=True)
