"""Parameterized robotic assembly tasks for mjlab.

Importing this package registers its tasks with mjlab's task registry; mjlab
discovers it through the ``mjlab.tasks`` entry point.
"""

from mjlab_assembly.tasks.peg_insertion.config import franka_panda  # noqa: F401
