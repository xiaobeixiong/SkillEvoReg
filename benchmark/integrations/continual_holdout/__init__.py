"""Frozen-library held-out evaluation for ContinualSkillBench.

Separate package on purpose: it reuses ``continual_regularization`` read-only and leaves every
existing experiment's code path untouched.
"""
from .agent import MathHeldout

__all__ = ["MathHeldout"]
