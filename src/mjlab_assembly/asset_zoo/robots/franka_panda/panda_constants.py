"""Franka Emika Panda with the Franka Hand, for mjlab.

The model is MuJoCo Menagerie's ``franka_emika_panda/panda.xml``, vendored
unmodified (Apache-2.0, see ``xmls/LICENSE``). ``get_spec`` prepares it for mjlab:

- Menagerie's actuators, gripper tendon and keyframe are removed; actuation comes
  from the entity's ``EntityArticulationInfoCfg``.
- Collision geoms get names, so collision configs and domain randomization can
  address them: ``<body>_collision[N]`` for the link meshes and
  ``{lf,rf}_pad{1..5}_collision`` for the fingertip pads.
- A ``tcp`` site sits between the fingertips (0.1034 m from the hand frame, the
  Franka flange-to-TCP offset) and a ``flange`` site at the hand mount.
- The fingers are held shut by a passive joint spring that presses with
  ``grip_force`` at a jaw opening of ``grip_width / 2``; the gripper is not
  actuated, so a held part stays held without spending action dimensions.

Gravity compensation is on for every robot body, the way the real arm's control
interfaces sit on its built-in compensation. The arm is driven by torque
motors, into which task-space impedance actions write joint torques.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

PANDA_XML: Path = Path(__file__).parent / "xmls" / "panda.xml"
PANDA_NOHAND_XML: Path = Path(__file__).parent / "xmls" / "panda_nohand.xml"
assert PANDA_XML.exists() and PANDA_NOHAND_XML.exists()

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
TCP_SITE = "tcp"
FLANGE_SITE = "flange"
TCP_OFFSET = 0.1034  # [m] hand frame to fingertip centre

# Franka datasheet: 87 N m on joints 1-4, 12 N m on joints 5-7.
EFFORT_LIMIT = {
  "joint1": 87.0, "joint2": 87.0, "joint3": 87.0, "joint4": 87.0,
  "joint5": 12.0, "joint6": 12.0, "joint7": 12.0,
}
ARMATURE = 0.1  # Menagerie's joint default


def get_spec(
  grip_force: float = 40.0,
  grip_width: float = 0.012,
  gravcomp: bool = True,
  hand: bool = True,
) -> mujoco.MjSpec:
  """Panda MjSpec prepared for mjlab (see module docstring).

  With ``hand=False`` the arm ends at a bare flange (Menagerie's
  ``panda_nohand.xml``), for tools fixed to the flange; the ``tcp`` site then
  sits ``TCP_OFFSET`` out from the flange, where the hand would put it.
  """
  spec = mujoco.MjSpec.from_file(str(PANDA_XML if hand else PANDA_NOHAND_XML))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for tendon in list(spec.tendons):
    spec.delete(tendon)
  for key in list(spec.keys):
    spec.delete(key)

  for body in spec.bodies:
    pad = 0
    mesh = 0
    for geom in body.geoms:
      if geom.classname.name.startswith("fingertip_pad_collision"):
        pad += 1
        prefix = "lf" if body.name == "left_finger" else "rf"
        geom.name = f"{prefix}_pad{pad}_collision"
      elif geom.classname.name == "collision" and not geom.name:
        geom.name = f"{body.name}_collision" + (f"{mesh}" if mesh else "")
        mesh += 1
    if gravcomp and body.name:
      body.gravcomp = 1.0

  tool = spec.body("hand" if hand else "attachment")
  tool.add_site(name=TCP_SITE, pos=(0.0, 0.0, TCP_OFFSET))
  tool.add_site(name=FLANGE_SITE)
  if not hand:
    return spec

  # The jaws never touch each other: closed on nothing, the finger meshes would be
  # pressed into one another, and that contact can go non-finite.
  spec.add_exclude(bodyname1="left_finger", bodyname2="right_finger")

  # Passive grip: the spring force at the held width equals grip_force.
  half = grip_width / 2.0
  for name in FINGER_JOINT_NAMES:
    j = spec.joint(name)
    j.stiffness = [grip_force / max(half, 1e-6), 0.0, 0.0]
    j.springref = 0.0
    j.damping = [20.0, 0.0, 0.0]
  return spec


##
# Collisions.
##

# Only the hand and the fingers collide (contype/conaffinity 1); the arm links are
# kept out of the contact pipeline to save compute. The fingertip pads add bit 4,
# which a held part's conaffinity selects (see ``workpieces.peg_hole``), and carry
# torsional and rolling friction so the part does not spin or slip under load.
COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={r"[lr]f_pad\d_collision": 5, r"(hand|left_finger|right_finger)_collision": 1,
           ".*_collision": 0},
  conaffinity={r"[lr]f_pad\d_collision": 5,
               r"(hand|left_finger|right_finger)_collision": 1, ".*_collision": 0},
  condim={r"[lr]f_pad\d_collision": 6, ".*_collision": 3},
  friction={r"[lr]f_pad\d_collision": (1.0, 0.03, 0.003), ".*_collision": (0.6,)},
  # MuJoCo's default contact (solref 0.02 s) is soft for light bodies: a 40 N grip
  # pressed the pads 3.5-5 mm into a 50 g peg. The pads have priority, so their
  # stiffer contact parameters govern every pad contact.
  solref={r"[lr]f_pad\d_collision": (0.005, 1.0)},
  solimp={r"[lr]f_pad\d_collision": (0.95, 0.99, 0.0005, 0.5, 2.0)},
  priority={r"[lr]f_pad\d_collision": 1, ".*": 0},
  # Leave unmatched geoms alone: a tool welded to the hand (for example
  # ``workpieces.peg_hole.weld_peg_to_hand``) carries its own bitmask.
  disable_other_geoms=False,
)

# Bare flange: no robot geom collides; tools fixed to the flange bring their own.
COLLISION_NOHAND = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=0,
  condim=3,
  priority=0,
  disable_other_geoms=False,
)

##
# Articulations.
##

TORQUE_ACTUATORS = tuple(
  BuiltinMotorActuatorCfg(
    target_names_expr=(name,), effort_limit=EFFORT_LIMIT[name], armature=ARMATURE
  )
  for name in ARM_JOINT_NAMES
)

TORQUE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=TORQUE_ACTUATORS, soft_joint_pos_limit_factor=0.95
)
##
# Initial state: hand pointing down, TCP about 0.45 m in front of the base.
##

HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.15,
    "joint3": 0.0,
    "joint4": -2.45,
    "joint5": 0.0,
    "joint6": 2.6,
    "joint7": 0.7854,
    "finger_joint.*": 0.006,
  },
  joint_vel={".*": 0.0},
)


HOME_NOHAND = EntityCfg.InitialStateCfg(
  pos=HOME.pos,
  joint_pos={k: v for k, v in HOME.joint_pos.items() if not k.startswith("finger")},
  joint_vel={".*": 0.0},
)


def get_panda_cfg(
  grip_force: float = 40.0,
  grip_width: float = 0.012,
  hand: bool = True,
  init_state: EntityCfg.InitialStateCfg | None = None,
) -> EntityCfg:
  """Panda entity with torque-driven joints, with the hand or a bare flange."""

  def spec_fn() -> mujoco.MjSpec:
    return get_spec(grip_force=grip_force, grip_width=grip_width, hand=hand)

  if init_state is None:
    init_state = HOME if hand else HOME_NOHAND
  return EntityCfg(
    init_state=dataclasses.replace(init_state, joint_pos=dict(init_state.joint_pos)),
    collisions=(COLLISION if hand else COLLISION_NOHAND,),
    spec_fn=spec_fn,
    articulation=TORQUE_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_panda_cfg()).spec.compile())
