"""MuJoCo practice simulation for the SO-101 follower.

Lets you rehearse Xbox-controller teleoperation before any hardware exists. The
sim presents the SAME observation/action interface as the real LeRobot arm
(``get_observation`` / ``send_action`` with ``"<joint>.pos"`` keys in normalized
units), so ``so101.controller.XboxTeleopController`` drives it unchanged.
"""
