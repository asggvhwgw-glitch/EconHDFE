"""Three-way FE benchmark families with a constructed projection oracle.

The hard families couple the two-dimensional kernels of full 2x2x2 local
categorical cubes.  They have six categories per module and normal rank,
so increasing ``size`` increases both observations and effective rank.

The oracle never calls an FE solver.  We form y = D alpha + r, where each
cube's weighted residual is a combination of its four Walsh interaction
contrasts.  Thus D.T @ W @ r = 0 by cancellation.  The actual coefficient
arrays are returned for an independent feasibility check.

Generator time and temporary memory are outside solver timing.  No general
near-linear FE complexity or finite-precision certificate is claimed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "kernel_grid": {
        "size_kind": "periodic_grid_side",
        "min_size": 3,
        "rows_per_module": 16,
        "categories_per_module": 6,
        "unique_rows_per_module": 16,
    },
    "rigidity_grid": {
        "size_kind": "periodic_grid_side",
        "min_size": 4,
        "rows_per_module": 20,
        "categories_per_module": 6,
        "unique_rows_per_module": 16,
    },
    "rigidity_grid_unique": {
        "size_kind": "periodic_grid_side",
        "min_size": 4,
        "rows_per_module": 16,
        "categories_per_module": 6,
        "unique_rows_per_module": 16,
    },
    "kernel_cycle": {
        "size_kind": "cycle_vertices",
        "min_size": 3,
        "rows_per_module": 12,
        "categories_per_module": 6,
        "unique_rows_per_module": 12,
    },
    "easy_crossed": {
        "size_kind": "graph_vertices",
        "min_size": 3,
        "rows_per_module": None,
        "categories_per_module": 6,
        "unique_rows_per_module": None,
    },
}
FAMILY_NAMES = tuple(FAMILY_SPECS)
RHS_KINDS = ("mixed", "slow", "broadband", "random")

# This order agrees with itertools.product(range(2), repeat=3) in the papers.
_CUBE_BITS = np.array(
    [[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
    dtype=np.int64,
)
_WALSH = np.column_stack(
    [
        1.0 - 2.0 * ((_CUBE_BITS[:, 0] + _CUBE_BITS[:, 1]) % 2),
        1.0 - 2.0 * ((_CUBE_BITS[:, 0] + _CUBE_BITS[:, 2]) % 2),
        1.0 - 2.0 * ((_CUBE_BITS[:, 1] + _CUBE_BITS[:, 2]) % 2),
        1.0 - 2.0 * (_CUBE_BITS.sum(axis=1) % 2),
    ]
)


@dataclass
class FECase:
    """A plain categorical design, target matrix and known residual.

    ``groups`` contains zero-based labels local to each of three dimensions.
    ``y`` and ``truth`` are always two-dimensional, including for one RHS.
    ``metadata`` contains JSON-compatible small values only.  The optional
    coefficient arrays live separately so runners do not serialize them.
    """

    name: str
    groups: np.ndarray
    y: np.ndarray
    truth: np.ndarray
    weights: np.ndarray | None
    metadata: dict[str, Any]
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def fitted_values(
    groups: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Evaluate a supplied feasible FE fit without solving a system."""
    result = coefficients[0][groups[:, 0]].copy()
    result += coefficients[1][groups[:, 1]]
    result += coefficients[2][groups[:, 2]]
    return result


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _both_directions(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Adjacent rows are opposite orientations, as in the reference constructor.
    return np.stack(
        (np.column_stack((source, target)), np.column_stack((target, source))),
        axis=1,
    ).reshape(-1, 2)


def _grid_edges(side: int) -> np.ndarray:
    nodes = np.arange(side * side, dtype=np.int64).reshape(side, side)
    neighbours = np.stack(
        (
            np.roll(nodes, -1, axis=0),
            np.roll(nodes, 1, axis=0),
            np.roll(nodes, -1, axis=1),
            np.roll(nodes, 1, axis=1),
        ),
        axis=2,
    )
    return np.column_stack((np.repeat(nodes.ravel(), 4), neighbours.ravel()))


def _rigidity_edges(side: int, unique: bool) -> tuple[np.ndarray, ...]:
    nodes = np.arange(side * side, dtype=np.int64).reshape(side, side)
    a = nodes.ravel()
    b = np.roll(nodes, -1, axis=0).ravel()
    c = np.roll(nodes, -1, axis=1).ravel()
    d = np.roll(np.roll(nodes, -1, axis=0), -1, axis=1).ravel()
    if unique:
        eu = _both_directions(a, b)
        ev = _both_directions(a, c)
    else:
        # Each cell's two axial edges appear in two neighbouring patches.
        eu = np.stack((np.column_stack((a, b)), np.column_stack((b, a)),
                       np.column_stack((c, d)), np.column_stack((d, c))), axis=1).reshape(-1, 2)
        ev = np.stack((np.column_stack((a, c)), np.column_stack((c, a)),
                       np.column_stack((b, d)), np.column_stack((d, b))), axis=1).reshape(-1, 2)
    ew = np.stack((np.column_stack((a, d)), np.column_stack((d, a)),
                   np.column_stack((b, c)), np.column_stack((c, b))), axis=1).reshape(-1, 2)
    return eu, ev, ew


def _easy_edges(modules: int, rng: np.random.Generator) -> np.ndarray:
    """A connected bounded-degree random graph, without an expander claim."""
    nodes = np.arange(modules, dtype=np.int64)
    cycles = [nodes, rng.permutation(nodes), rng.permutation(nodes)]
    pairs = np.concatenate(
        [np.column_stack((cycle, np.roll(cycle, -1))) for cycle in cycles], axis=0
    )
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    return _both_directions(pairs[:, 0], pairs[:, 1])


def _make_groups(
    family: str, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, int, dict[str, Any]]:
    grid = FAMILY_SPECS[family]["size_kind"] == "periodic_grid_side"
    modules = size * size if grid else size
    internal = (2 * np.arange(modules, dtype=np.int64)[:, None, None]
                + _CUBE_BITS[None, :, :]).reshape(-1, 3)
    empty = np.empty((0, 2), dtype=np.int64)
    if family == "kernel_grid":
        eu = ev = _grid_edges(size)
        ew = empty
        base_bound = 12 * modules
        bound_source = "Implicit Kernel Coupling, Theorem 5.1 / Section 5.1"
        spectrum_order = "Theta(modules) for unit weights"
    elif family.startswith("rigidity_grid"):
        unique = family == "rigidity_grid_unique"
        eu, ev, ew = _rigidity_edges(size, unique)
        base_bound = (24 if unique else 12) * modules
        bound_source = (
            "Rigidity Patches, Section 6.3; factor-two duplicate-removal comparison"
            if unique else "Rigidity Patches, Section 6.3"
        )
        spectrum_order = "Theta(modules) for unit weights"
    elif family == "kernel_cycle":
        source = np.arange(modules, dtype=np.int64)
        eu = ev = _both_directions(source, np.roll(source, -1))
        ew = empty
        base_bound = 6 * modules * modules
        bound_source = "FAMILY_DESIGN.md: affine cycle symbol and exact 6x6 minor certificate"
        spectrum_order = "Theta(modules**2) for unit weights"
    else:
        eu = ev = _easy_edges(modules, rng)
        ew = empty
        base_bound = None
        bound_source = "none; empirical easy control, no certified graph expansion"
        spectrum_order = "not asserted"

    ru = np.column_stack((2 * eu[:, 0], 2 * eu[:, 1], 2 * eu[:, 1]))
    rv = np.column_stack((2 * ev[:, 1], 2 * ev[:, 0], 2 * ev[:, 1]))
    rw = np.column_stack((2 * ew[:, 0], 2 * ew[:, 0], 2 * ew[:, 1]))
    groups = np.ascontiguousarray(np.concatenate((internal, ru, rv, rw), axis=0))
    unique_rows = 16 * modules if family == "rigidity_grid" else len(groups)
    details = {
        "modules": modules,
        "grid_side": size if grid else None,
        "core_rows": 8 * modules,
        "connector_rows_by_kind": {"u": len(eu), "v": len(ev), "u_plus_v": len(ew)},
        "base_unique_tuples": unique_rows,
        "base_kappa_upper": base_bound,
        "kappa_bound_source": bound_source,
        "spectrum_order": spectrum_order,
    }
    return groups, modules, details


def _mode_frequencies(size: int, grid: bool) -> list[tuple[int, ...]]:
    # Include the smallest frequency and a growing range of spatial scales.
    # Only one +/- representative of a real Fourier pair is needed.
    max_frequency = max(1, size // 2)
    frequencies = [1, 2, 3]
    frequency = 4
    while frequency <= max_frequency:
        frequencies.append(frequency)
        frequency *= 2
    if max_frequency not in frequencies:
        frequencies.append(max_frequency)
    frequencies = sorted(set(k for k in frequencies if k <= max_frequency))
    if not grid:
        return [(k,) for k in frequencies[:24]]
    result: list[tuple[int, ...]] = []
    for k in frequencies:
        result.extend(((k, 0), (0, k), (k, k)))
    # Keep all low modes and representatives spread over the remaining scales.
    if len(result) > 24:
        indices = np.linspace(0, len(result) - 1, 24, dtype=int)
        result = [result[int(i)] for i in indices]
    return result


def _slow_coefficients(
    family: str, size: int, modules: int, rhs: int, rng: np.random.Generator
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[list[int]]]:
    grid = FAMILY_SPECS[family]["size_kind"] == "periodic_grid_side"
    frequencies = _mode_frequencies(size, grid)
    if grid:
        x, z = np.divmod(np.arange(modules), size)
    else:
        x = np.arange(modules)
        z = np.zeros(modules, dtype=np.int64)
    u = np.zeros((modules, rhs))
    v = np.zeros((modules, rhs))
    for mode in frequencies:
        p = mode[0]
        q = mode[1] if grid else 0
        phase = (2 * np.pi / size) * (p * x + q * z)
        tau = 4 * math.sin(math.pi * p / size) ** 2
        if grid:
            tau += 4 * math.sin(math.pi * q / size) ** 2
        # The fit is a gradient of these local-kernel fields.  Compensating
        # its small gradient keeps the slowest spatial modes observable.
        amplitude = 1.0 / math.sqrt(tau)
        phase_u = rng.uniform(-np.pi, np.pi, size=rhs)
        phase_v = rng.uniform(-np.pi, np.pi, size=rhs)
        mix_u = rng.choice(np.array([-1.0, 1.0]), size=rhs)
        mix_v = rng.choice(np.array([-1.0, 1.0]), size=rhs)
        u += amplitude * np.cos(phase[:, None] + phase_u) * mix_u
        v += amplitude * np.cos(phase[:, None] + phase_v) * mix_v
    coefficients = (
        np.repeat(u, 2, axis=0),
        np.repeat(v, 2, axis=0),
        np.repeat(-u - v, 2, axis=0),
    )
    return coefficients, [list(mode) for mode in frequencies]


def _broadband_coefficients(
    family: str, size: int, modules: int, rhs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-square-root graph filtering with all nonzero Fourier modes.

    Only the generated grid/cycle coordinates are used.  This is target
    construction, not a solver or a learned preconditioner.  The random-graph
    control uses vertex ordering and makes no slow-mode claim for that graph.
    """
    grid = FAMILY_SPECS[family]["size_kind"] == "periodic_grid_side"
    phase = 2 * np.pi * np.arange(size) / size
    axis_tau = 2 - 2 * np.cos(phase)
    if grid:
        tau = axis_tau[:, None] + axis_tau[None, :]
        shape, axes = (size, size, rhs), (0, 1)
    else:
        tau = axis_tau
        shape, axes = (size, rhs), (0,)
    tau = tau.copy()
    tau.flat[0] = 1.0
    fields = []
    for _ in range(2):
        spectrum = np.fft.fftn(rng.standard_normal(shape), axes=axes)
        spectrum /= np.sqrt(tau)[..., None]
        spectrum[(0,) * len(axes)] = 0.0
        fields.append(np.fft.ifftn(spectrum, axes=axes).real.reshape(modules, rhs))
    u, v = fields
    return (
        np.repeat(u, 2, axis=0),
        np.repeat(v, 2, axis=0),
        np.repeat(-u - v, 2, axis=0),
    )


def _scaled_weights(weights: np.ndarray | None, n: int) -> np.ndarray | None:
    return None if weights is None else weights / np.max(weights)


def _column_rms(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    # Scale both weights and values before squaring, so moderate stress weights
    # do not turn a finite oracle into an overflowed normalization calculation.
    magnitude = np.max(np.abs(values), axis=0)
    safe = np.where(magnitude > 0, magnitude, 1.0)
    scaled = values / safe
    if weights is None:
        mean_square = np.mean(scaled * scaled, axis=0)
    else:
        sw = _scaled_weights(weights, len(values))
        mean_square = np.sum(sw[:, None] * scaled * scaled, axis=0) / np.sum(sw)
    return magnitude * np.sqrt(mean_square)


def _unit_fit(
    groups: np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: np.ndarray | None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    fit = fitted_values(groups, coefficients)
    scales = _column_rms(fit, weights)
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("The constructed fit has zero or nonfinite weighted norm.")
    coefficients = tuple(np.ascontiguousarray(a / scales) for a in coefficients)
    # Recompute from the actual returned coefficients, avoiding an unnecessary
    # feasibility discrepancy from two different floating-point evaluation paths.
    return coefficients, fitted_values(groups, coefficients)


def _fitted_rhs(
    family: str,
    size: int,
    modules: int,
    groups: np.ndarray,
    rhs: int,
    rhs_kind: str,
    weights: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, dict[str, Any]]:
    modes: list[list[int]] = []
    if rhs_kind in ("mixed", "slow", "broadband"):
        slow, modes = _slow_coefficients(family, size, modules, rhs, rng)
        slow, slow_fit = _unit_fit(groups, slow, weights)
    if rhs_kind == "slow":
        coefficients, fit = slow, slow_fit
    elif rhs_kind == "random":
        coefficients = tuple(rng.standard_normal((2 * modules, rhs)) for _ in range(3))
        coefficients, fit = _unit_fit(groups, coefficients, weights)
    else:
        fast = (
            _broadband_coefficients(family, size, modules, rhs, rng)
            if rhs_kind == "broadband" else
            tuple(rng.standard_normal((2 * modules, rhs)) for _ in range(3))
        )
        fast, fast_fit = _unit_fit(groups, fast, weights)
        sw = _scaled_weights(weights, len(groups))
        if sw is None:
            cross = np.sum(fast_fit * slow_fit, axis=0) / np.sum(slow_fit * slow_fit, axis=0)
        else:
            cross = np.sum(sw[:, None] * fast_fit * slow_fit, axis=0) / np.sum(
                sw[:, None] * slow_fit * slow_fit, axis=0
            )
        fast = tuple(fast[k] - cross * slow[k] for k in range(3))
        fast, _ = _unit_fit(groups, fast, weights)
        targeted_fraction = 0.5 if rhs_kind == "broadband" else 0.9
        coefficients = tuple(math.sqrt(targeted_fraction) * slow[k]
                             + math.sqrt(1 - targeted_fraction) * fast[k]
                             for k in range(3))
        coefficients, fit = _unit_fit(groups, coefficients, weights)
    rhs_details = {
        "kind": rhs_kind,
        "spatial_frequencies": modes,
        "spatial_mode_count": len(modes),
        "full_nonzero_fourier_support": rhs_kind == "broadband",
        "broadband_frequency_count": modules - 1 if rhs_kind == "broadband" else None,
        "finite_mode_warning": (
            "pure slow has at most 24 spatial frequencies; not a worst-case RHS scaling claim"
            if rhs_kind == "slow" else None
        ),
        "slow_component_fraction_of_fit_energy": (
            0.9 if rhs_kind == "mixed" else 0.5 if rhs_kind == "broadband"
            else 1.0 if rhs_kind == "slow" else 0.0
        ),
        "fit_fraction_of_target_energy": 0.5,
        "truth_fraction_of_target_energy": 0.5,
        "slow_mode_scope": (
            "local-kernel multi-frequency fields; not supplied exact global eigenvectors"
            if family != "easy_crossed" else
            "vertex-order Fourier fields are not claimed to be slow for the random graph"
        ),
    }
    return coefficients, fit, rhs_details


def make_case(
    family: str,
    size: int,
    rhs: int = 2,
    seed: int = 0,
    repeat: int = 1,
    weight_ratio: float = 1.0,
    rhs_kind: str = "mixed",
) -> FECase:
    """Construct a reproducible case without an iterative projection oracle.

    For grid families ``size`` is the side length, and modules = size**2.
    For ``kernel_cycle`` and ``easy_crossed`` it is the number of modules.
    The returned metadata records actual N, G and the unique tuple count U.

    ``repeat`` uniformly repeats the complete base row set.  It is a duplicate
    compression stress parameter, not a rank-growing scaling experiment.
    ``weight_ratio`` creates positive log-spread weights in [1, weight_ratio];
    the residual oracle is reconstructed for those actual weights.
    """
    if family not in FAMILY_SPECS:
        raise ValueError(f"Unknown family {family!r}; choose from {FAMILY_NAMES}.")
    size = _positive_int(size, "size", FAMILY_SPECS[family]["min_size"])
    rhs = _positive_int(rhs, "rhs")
    repeat = _positive_int(repeat, "repeat")
    if rhs_kind not in RHS_KINDS:
        raise ValueError(f"rhs_kind must be one of {RHS_KINDS}.")
    weight_ratio = float(weight_ratio)
    if not np.isfinite(weight_ratio) or weight_ratio < 1:
        raise ValueError("weight_ratio must be finite and >= 1.")
    streams = np.random.SeedSequence(seed).spawn(4)
    topology_rng, fit_rng, residual_rng, weight_rng = [np.random.default_rng(s) for s in streams]
    groups, modules, details = _make_groups(family, size, topology_rng)
    base_n = len(groups)
    if repeat != 1:
        groups = np.tile(groups, (repeat, 1))
    n = len(groups)
    weights = None
    if weight_ratio > 1:
        weights = np.exp(weight_rng.uniform(0.0, math.log(weight_ratio), size=n))
        # The advertised ratio is an actual ratio, not just a sampling limit.
        weights[0], weights[-1] = 1.0, weight_ratio

    # Each cube contrast cancels within every category of every FE dimension.
    # Dividing by actual row weights preserves that exact weighted identity.
    amplitudes = residual_rng.standard_normal((modules, 4, rhs))
    core_residual = np.einsum("bi,mip->mbp", _WALSH, amplitudes).reshape(8 * modules, rhs)
    truth = np.zeros((n, rhs))
    for copy in range(repeat):
        start = copy * base_n
        truth[start:start + 8 * modules] = core_residual
    if weights is not None:
        truth /= weights[:, None]
    truth_scale = _column_rms(truth, weights)
    if np.any(truth_scale <= 0) or not np.all(np.isfinite(truth_scale)):
        raise ValueError("The constructed residual has zero or nonfinite weighted norm.")
    truth /= truth_scale

    coefficients, fit, rhs_details = _fitted_rhs(
        family, size, modules, groups, rhs, rhs_kind, weights, fit_rng
    )
    y = fit + truth
    base_bound = details.pop("base_kappa_upper")
    metadata: dict[str, Any] = {
        "family": family,
        "size": size,
        "size_kind": FAMILY_SPECS[family]["size_kind"],
        "seed": int(seed),
        "rhs_count": rhs,
        "N": n,
        "G": 6 * modules,
        "U": details["base_unique_tuples"],
        "categories_per_dimension": [2 * modules] * 3,
        "expected_rank": 6 * modules - 2,
        "normal_rank": True,
        "base_N": base_n,
        "repeat": repeat,
        "weight_ratio": weight_ratio,
        "weighted": weights is not None,
        "row_order": "complete base copies; each base: cubes, u, v, u_plus_v connectors",
        "dimension_growth": "G proportional to N when repeat is fixed",
        "repeat_scope": "duplicates only; increasing repeat does not increase G or rank",
        "rhs": rhs_details,
        "oracle": {
            "kind": "constructed_fe_plus_weighted_cube_contrast",
            "certification": "algebraic",
            "coefficients_available": True,
            "orthogonality_mechanism": "local_cube_walsh_interactions_divided_by_weights",
            "finite_precision_note": "rounded numpy arrays; validate numerical cancellation independently",
        },
        "condition_number": {
            "definition": "positive_spectrum_frequency_normalized_gram",
            "analytic_upper_bound": None if base_bound is None else base_bound * weight_ratio,
            "base_analytic_upper_bound": base_bound,
            "weighted_comparison_factor": weight_ratio,
            "uniform_repeat_changes_unweighted_spectrum": False,
            "weighted_repeat_note": "independent weights on row copies may change the numeric spectrum",
            "status": "analytic_family_bound" if base_bound is not None else "not_certified",
            "weight_scope": "arbitrary positive row weights within the stated global ratio",
        },
        **details,
    }
    return FECase(
        name=f"{family}-s{size}-p{rhs}-{rhs_kind}-r{repeat}-w{weight_ratio:g}",
        groups=groups,
        y=np.ascontiguousarray(y),
        truth=np.ascontiguousarray(truth),
        weights=weights,
        metadata=metadata,
        coefficients=coefficients,
    )
