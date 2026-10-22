"""Disco103, the frozen discovered update rule, in place of PPO's loss.

The rule itself is not in this package: `bridge` imports it from `disco_rl`,
with its published weights and settings. `heads` and `learner` are the torch
side, and import `bridge` only when a run turns `disco103` on.
"""
