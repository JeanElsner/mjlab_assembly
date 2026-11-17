# Roadmap

The aim is a go-to suite for FORGE-style contact-rich assembly learning on MuJoCo:
the task set of the NIST Assembly Task Board and Isaac Lab's Factory/FORGE
environments, with reference configurations that match theirs for comparability,
and every difficulty knob exposed so the same tasks can be made much harder
(tighter fits, no chamfers, more uncertainty). Every result should be reportable
as success *and* the contact quality behind it.

## Reference: what other suites offer

| Suite | Tasks |
|---|---|
| Factory (Isaac Lab) | `PegInsert` (8 mm round), `GearMesh` (medium gear onto its shaft, meshing with two flanking gears), `NutThread` (M16×2) |
| FORGE (Isaac Lab) | Factory's three tasks with a smoothed, noisy wrist force observation and a randomized force threshold (5–10 N) the policy is conditioned on |
| Isaac Lab NIST presets | round rods and rectangular pegs at 4, 8, 12, 16 mm; small, medium, large gear; M16 nut |
| IndustReal | 8–16 mm pegs, gear insertion; sim-to-real onto the NIST board |
| AutoMate (Isaac Lab) | 100 plug/socket assemblies; scripted disassembly for demonstrations |
| NIST Task Board 1 | round and square pegs (4–16 mm), gears, BNC/USB/RJ45/D-sub connectors, nut threading; Board 2 adds a belt drive |
| mjfit / Blind Precise Assembly (MuJoCo) | arbitrary-outline pegs, gear meshing, nut threading; proprioception only |
| MuJoCo Warp benchmarks | `franka_threading`: M16×2 nut, convex helical threads (mjfit) |

Reference parameters to match, measured from Factory's shipped meshes where they
disagree with its configuration:

| Part | Factory/FORGE | NIST Task Board 1 |
|---|---|---|
| 8 mm round peg | Ø7.986 mm × 50 mm, 0.5 mm × 45° chamfer at both ends | 50 mm long; 45° chamfer of 0.1 × the largest cross-section on one end only (so chamfered and plain ends can both be tested) |
| 8 mm hole | Ø9.000 mm (0.507 mm radial; the configuration says 8.1 mm), 25 mm deep, 1 mm × 45° mouth chamfer | slip fit, no chamfer |
| Gear shaft / bore | shaft Ø9.5 mm with a two-step 1 mm tip chamfer; medium-gear bore radius about 5.27 mm with a 0.5–0.7 mm lead-in | gears on guide rods |
| M16×2 bolt | tip chamfered from the 7.84 mm thread radius to 6.43 mm | M4/M6/M8 threads on the practice board |
| Friction, episode | 0.75; pegs 10 s, gears 20 s, nut 30 s | |

## Foundations (needed by every task)

1. **Geometry from mjfit.** Pegs and holes from any outline, ISO threads and
   involute gears as exact convex pieces, plus its connector configs (USB-A/C,
   RJ45, DB9, HDMI, XLR). Its threads already take a chamfer (bolt tip and nut
   countersink), but its pegs and holes have none and its gears have a solid hub
   on an axle, while both standards chamfer pegs and Factory slides a gear's bore
   onto a shaft. In a fork, add peg-tip and hole-mouth chamfers for any convex
   outline and gears with a (chamfered) bore, as exact convex pieces like the
   rest of mjfit, and propose them upstream. Its dependency on `mujoco` collides with the `mujoco-core` that
   mjlab now uses, so install it with a dependency override until upstream
   accepts either package.
2. **Task presets with difficulty knobs.** Each task ships a standard preset
   (NIST/Factory dimensions, Factory/FORGE randomization ranges) and exposes:
   clearance (a ladder per task), chamfers, friction, part-pose and in-grasp pose
   noise, observation noise, episode length, upright/yaw locks, fixed or
   policy-set stiffness, welded or grasped part.
3. **Wrench sensing that matches the hardware.** The Franka has joint torque
   sensors only and estimates the end-effector wrench from them. Default channel:
   the external joint torques (constraint forces on the arm's joints, contacts
   only), mapped to a wrench through the pseudo-inverse of the Jacobian transpose
   at the flange or TCP, with low-pass filtering, noise and bias like the robot's
   estimate. An ideal wrist F/T reading stays available for FORGE parity
   comparisons, never as the default.
4. **Simulation-aware policy update (SAPU).** Down-weight credit for progress made
   while parts interpenetrate beyond the clearance (Tang et al., IndustReal 2023),
   with per-step weighting from a sustained-overlap measure as an option.
5. **Success and reward options.** Factory's keypoint reward and termination, and
   alternatively a seat condition paid per step without termination, so a
   policy is scored on holding the seat.
6. **Evaluation harness.** Deterministic evaluation (mean action, observation
   noise kept, no early termination) reporting, per task and clearance: success,
   time to seat, interpenetration (95th percentile and the deepest overlap held
   for 100 ms, from the part poses against the exact geometry), a static
   effective-clearance probe, and force statistics from the wrench channel.
7. **Throughput reporting** against Factory/FORGE at equal batch size.

## Phase 1: NIST/FORGE parity

| Task | Variants | Notes |
|---|---|---|
| Task (Isaac Lab / NIST name) | Variants | Notes |
|---|---|---|
| Rod insert (round peg) | 4, 8, 12, 16 mm | replaces today's `PegInsert` |
| Peg insert (rectangular peg) | 4, 8, 12, 16 mm | needs the yaw action unlocked |
| Gear mesh | small, medium, large gear, with flanking gears | gears held in the gripper work with Newton; hinged gears need PGS |
| Nut thread | M16×2 | yaw action; joint 7 allows less than one turn, so a partial-thread goal as in FORGE |

Parts start in the gripper by default, as in Factory, with welded variants.
Each task comes in a Factory variant (pose and joint observations) and a FORGE
variant (wrench observation and force threshold).

## Phase 2: breadth

- Keyed connectors from the NIST board: USB, RJ45, D-sub, BNC, using mjfit's configs.
- A procedural part family (mjfit's random sampling) for generalist policies, as AutoMate.

## Phase 3: beyond existing suites

- Snap fits (a compliant latch, e.g. the RJ45 clip, as a sprung hinge).
- Bayonet (BNC push-and-twist).
- Disassembly, as a task and as a demonstration source.
- Cables and belts once MuJoCo Warp's flex support is complete.

## Later

- Regrasping (full nut runs, multi-turn tasks).
- Picking the part from a fixture before inserting it.
