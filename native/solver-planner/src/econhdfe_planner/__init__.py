"""Opt-in FE planner over pinned, independently certified numerical backends."""
from .planner import PlannedAbsorber, PlanningError, ProjectionDisagreement
__all__ = ["PlannedAbsorber", "PlanningError", "ProjectionDisagreement"]
__version__ = "0.2.0a1"
