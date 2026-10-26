"""One-time fail-closed wiring for the dedicated opt-in backend branch.

The branch-only bootstrap workflow commits these generated changes and removes
this helper. It is never a runtime import or an installation side effect.
"""
from pathlib import Path
import ast


def replace(path, old, new):
    p = Path(path); text = p.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"refusing ambiguous/stale integration edit: {path}")
    changed = text.replace(old, new, 1)
    ast.parse(changed)
    p.write_text(changed)

replace("econhdfe/config.py", '{"auto", "map", "lsmr", "twoway"}',
        '{"auto", "map", "lsmr", "twoway", "within"}')
replace("econhdfe/config.py", 'HDFE solver must be auto/map/lsmr/twoway',
        'HDFE solver must be auto/map/lsmr/twoway/within')
replace("econhdfe/pipeline.py", '''    if method == "twoway":
        if backend != "numpy":''', '''    if method == "within":
        if backend != "numpy" or any(s is not None for s in slopes) or not all(intercepts):
            raise ValueError("method='within' requires CPU pure-intercept categorical FEs")
    if method == "twoway":
        if backend != "numpy":''')
replace("econhdfe/pipeline.py", '''    with scheduler.reserve_memory(build_bytes, label="linear_fe_build"):
        if method == "twoway":''', '''    with scheduler.reserve_memory(build_bytes, label="linear_fe_build"):
        if method == "within":
            from .hdfe.within_backend import WithinFEAbsorber
            return WithinFEAbsorber(
                groups, weights=weights, tol=tol, max_iter=max_iter,
                absorb_threads=absorb_threads, projection_memory_budget_mb=memory_budget_mb,
            )
        if method == "twoway":''')
replace("econhdfe/pipeline.py", '''    if solver_selection is not None:
        out["solver_selection"] = dict(solver_selection)''', '''    if absorber.method == "within":
        out["within"] = absorber.resource_info
    if solver_selection is not None:
        out["solver_selection"] = dict(solver_selection)''')
replace("econhdfe/hdfe/workspace.py", '''    arrays = {}

    def add(a):''', '''    if getattr(absorber, "method", None) == "within":
        return absorber.workspace_cost()
    arrays = {}

    def add(a):''')
replace("econhdfe/hdfe/plan.py", '''    def absorber(self, weights, *, tol, engine="replica", method="map"):
        if not self.groups:
            return None''', '''    def absorber(self, weights, *, tol, engine="replica", method="map"):
        if not self.groups:
            return None
        if method == "within":
            from .within_backend import WithinFEAbsorber
            return WithinFEAbsorber(self.groups, weights=weights, tol=tol)''')
replace("econhdfe/hdfe/weighted_projection.py", '''    def _kind(self, weights) -> str:
        if (''', '''    def _kind(self, weights) -> str:
        if self.method == "within":
            return "within" if np.all(np.asarray(weights) > 0) else "generic"
        if (''')
replace("econhdfe/hdfe/weighted_projection.py", '''    def _build(self, weights, tol, kind: str):
        if kind == "twoway":''', '''    def _build(self, weights, tol, kind: str):
        if kind == "within":
            from .within_backend import WithinFEAbsorber
            return WithinFEAbsorber(
                self.groups, weights=weights, tol=tol, max_iter=self.max_iter,
                absorb_threads=self.absorb_threads_requested,
                projection_memory_budget_mb=self.projection_memory_budget_mb,
            )
        if kind == "twoway":''')
replace("econhdfe/hdfe/weighted_projection.py", '''            method=self.method,
            transform="symmetric",''', '''            method="lsmr" if self.method == "within" else self.method,
            transform="symmetric",''')
print("WIRED: config, shared linear factory, FEPlan, weighted projector, workspace accounting")
