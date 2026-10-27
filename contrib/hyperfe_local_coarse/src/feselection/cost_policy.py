"""Explicit cold-start heuristics + bounded useful-column measurement.

These are not trained probabilities or guaranteed time bounds. No future RHS
are invented. Post-stage deadlines are soft; a BLAS/native call is not preempted.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class CostPolicy:
    minimum_iterations: int = 40
    # If adaptive escalated, obtain one REAL warm RHS before extrapolating its
    # startup cost across a wide batch. That RHS is retained, never a throwaway.
    warm_confirmation_min_rhs: int = 6
    # Heuristic setup-work floor, not a rigorous lower bound. A budget must
    # cover at least 48 measured native iterations before building a global
    # profile/star/partition. Calibrated on v0.1; frozen before new seeds.
    setup_iteration_equivalents: int = 48
    exploration_fraction: float = .35
    planning_seconds_cap: float = 5.
    gain_margin: float = .20
    def __post_init__(self):
        if (isinstance(self.warm_confirmation_min_rhs,bool) or
                not isinstance(self.warm_confirmation_min_rhs,int) or
                self.warm_confirmation_min_rhs < 3):
            raise ValueError("warm_confirmation_min_rhs must be an integer >=3")
        if isinstance(self.setup_iteration_equivalents,bool) or not isinstance(self.setup_iteration_equivalents,int) or self.setup_iteration_equivalents < 0:
            raise ValueError('setup_iteration_equivalents must be a nonnegative integer')
        if isinstance(self.minimum_iterations,bool) or not isinstance(self.minimum_iterations,int) or self.minimum_iterations<0:
            raise ValueError('minimum_iterations must be a nonnegative integer')
        if not math.isfinite(self.exploration_fraction) or not 0<self.exploration_fraction<=1:
            raise ValueError('invalid exploration_fraction')
        if not math.isfinite(self.gain_margin) or not 0<=self.gain_margin<1:
            raise ValueError('invalid gain_margin')
        if not math.isfinite(self.planning_seconds_cap) or self.planning_seconds_cap<=0:
            raise ValueError('planning_seconds_cap must be finite and positive')
    def construction_budget(self, baseline_seconds_per_rhs, remaining_rhs):
        return min(self.planning_seconds_cap,
                   self.exploration_fraction*baseline_seconds_per_rhs*remaining_rhs)
    def continue_local(self, measured_local_seconds, baseline_seconds):
        return measured_local_seconds < (1-self.gain_margin)*baseline_seconds

    def exploration_admitted(self, iterations, remaining_rhs):
        """Do not start a costly build for a small remaining amount of work."""
        return (self.exploration_fraction * iterations * remaining_rhs
                >= self.setup_iteration_equivalents)
