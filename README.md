# mjlab_assembly

Parameterized robotic assembly tasks for [mjlab](https://github.com/mujocolab/mjlab)
on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), starting with
peg-in-hole on the Franka Panda.

The goal is a stable, fast and simple baseline: exact collision geometry at a
clearance you choose, the robot holding the part in its gripper, and task-space
impedance control as the default action space.

> Status: pre-release (v0.0.x). Interfaces may change.

## Tasks

| Task ID | Peg | Action space |
|---|---|---|
| `Mjlab-PegInsert-Franka` | grasped | task-space impedance, fixed stiffness |
| `Mjlab-PegInsert-Franka-Vic` | grasped | task-space impedance, policy-set stiffness |
| `Mjlab-PegInsert-Franka-JointPos` | grasped | joint position targets |
| `Mjlab-PegInsert-Franka-Welded` | welded to the flange (no hand) | task-space impedance, fixed stiffness |
| `Mjlab-PegInsert-Franka-Welded-Vic` | welded to the flange (no hand) | task-space impedance, policy-set stiffness |

Defaults: a 12 mm peg, 50 mm long, in a 25 mm deep blind bore with 0.5 mm radial
clearance and 0.5 mm entry chamfers on both parts; the socket's position is
randomized by ±20 mm per episode; 50 Hz policy over 500 Hz physics.

## Getting started

```bash
git clone https://github.com/JeanElsner/mjlab_assembly.git && cd mjlab_assembly
uv sync
uv run list-envs --keyword PegInsert
uv run train Mjlab-PegInsert-Franka
```

Training uses 4096 environments by default and logs to Weights & Biases (add
`--agent.logger tensorboard` to log locally). Checkpoints land in
`logs/rsl_rl/<experiment>/<run>/`. To watch a policy in the browser:

```bash
# a local checkpoint
uv run play Mjlab-PegInsert-Franka --viewer viser \
  --checkpoint-file logs/rsl_rl/peginsert_franka/<run>/model_599.pt
# or a W&B run (entity/project/run-id)
uv run play Mjlab-PegInsert-Franka --viewer viser --wandb-run-path <entity>/mjlab/<run-id>
# or no policy at all, to look at the scene
uv run play Mjlab-PegInsert-Franka --viewer viser --agent zero
```

Other geometries are a configuration away:

```python
from mjlab_assembly.asset_zoo.workpieces.peg_hole import PegHoleCfg
from mjlab_assembly.tasks.peg_insertion.config.franka_panda.env_cfgs import (
  franka_peg_insertion_env_cfg,
)

cfg = franka_peg_insertion_env_cfg(geometry=PegHoleCfg(clearance=0.001, peg_radius=0.008))
```

## Design

**Collision geometry.** The peg is one convex mesh and the socket an exact convex
decomposition of the bore into annular sectors, so every contact goes through
MuJoCo Warp's convex narrowphase. The polygon resolution (`segments`, default 32)
trades fidelity for speed; at 32 segments the polygonal bore deviates from the
round one by 0.5 % of its radius. The socket is rendered as one smooth mesh of
the same shape. Analytic signed distance fields would be exact and are planned,
but MuJoCo Warp has no public API for user-defined SDFs yet: a custom distance
function can only be injected by overriding a private kernel function
(`mujoco_warp._src.collision_sdf.user_sdf`), which is too fragile for a library.
MuJoCo's native mesh SDFs need no such override but are not reliable yet: in
MuJoCo Warp 3.11 a mesh-SDF peg passed through the bore floor
(`benchmarks/collision_backends.py`; an octree-gradient fix is pending upstream).
On the same GPU, the convex decomposition is also about 30 % cheaper per physics
step than an analytic SDF peg-in-hole.

**Grasp.** The Franka Hand's jaws are held shut by a passive joint spring that
presses with 40 N on the peg; the gripper is not part of the action. Each
episode starts with the peg already in the hand.

**Action spaces.** `TaskSpaceImpedanceAction` moves a reference pose for the TCP
and applies a Cartesian impedance law at the physics rate, with damping critical
with respect to the task-space inertia and a dynamically consistent null-space
posture term. Stiffness is fixed (400 N/m, 30 N m/rad) or set by the policy
within (100, 1000) N/m and (5, 50) N m/rad. The yaw of a round peg is not
commanded. Gravity compensation is on for every robot body, as on the real arm.

**Reward.** Factory's squashed keypoint distance to the seated pose, at three
scales, plus a bonus while inserted and a small action-rate penalty.

## Robot model

The Panda and the Franka Hand come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`franka_emika_panda`, Apache-2.0), vendored unmodified.

## License

Apache-2.0.
