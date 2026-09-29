from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import numpy as np
from .frontend.validate import require_numeric, require_identifier
from .frontend.roles import VariableRole
from .factorvars import FactorVariableExpression, _FVAtom
from .prediction import PredictionInput, CategoricalEncodingState, DesignTermState

from .design_structure import StructuralTerm, StructuralCollinearityPlan, plan_structural_collinearity


@dataclass(frozen=True, slots=True)
class _FactorBaseSelector:
    mode: str


@dataclass(frozen=True, slots=True)
class Factor:
    """Categorical regressor expanded to observed-level indicators."""

    variable: Any
    base: Any | None = None
    drop_base: bool = True
    name: str | None = None


@dataclass(frozen=True, slots=True)
class RegressorInteraction:
    """Multiplicative interaction among continuous/factor regressor terms."""

    parts: tuple[Any, ...]
    name: str | None = None

    def __init__(self, *parts: Any, name: str | None = None):
        if len(parts) < 2:
            raise ValueError("a regressor interaction requires at least two parts")
        object.__setattr__(self, "parts", tuple(parts))
        object.__setattr__(self, "name", name)




@dataclass(frozen=True, slots=True)
class OmitSpec:
    """Explicitly omit a requested expanded design column or factor level.

    ``name`` matches one exact expanded column name. ``term`` matches a whole
    term (or narrows a component-level selector). ``component``/``level``
    matches factor components such as event_time[-1] inside an interaction.
    """

    name: str | None = None
    term: str | None = None
    component: str | None = None
    level: str | None = None


@dataclass(frozen=True, slots=True)
class UserOmission:
    index: int
    name: str
    term: str
    kind: str
    components: tuple[str, ...]
    level: str | None
    selector: OmitSpec

    def as_dict(self) -> dict:
        return {
            "index": int(self.index), "name": self.name, "term": self.term,
            "kind": self.kind, "components": self.components, "level": self.level,
            "reason": "user_reference", "dependent_on": (), "proof_type": "user_selection",
            "selected_by_user": True,
        }


def omit_column(name: str) -> OmitSpec:
    return OmitSpec(name=str(name))


def omit_term(term: str) -> OmitSpec:
    return OmitSpec(term=str(term))


def omit_level(component: str, level, *, term: str | None = None) -> OmitSpec:
    return OmitSpec(term=None if term is None else str(term), component=str(component), level=str(level))


def _normalize_omit_specs(specs) -> tuple[OmitSpec, ...]:
    if specs is None:
        return ()
    if isinstance(specs, (str, OmitSpec)):
        specs = [specs]
    out = []
    for spec in specs:
        if isinstance(spec, OmitSpec):
            out.append(spec)
        elif isinstance(spec, str):
            out.append(OmitSpec(name=spec))
        else:
            raise TypeError("omit entries must be column names or OmitSpec objects")
    return tuple(out)


def _omit_matches(spec: OmitSpec, origin: ColumnOrigin) -> bool:
    if spec.name is not None and origin.name != spec.name:
        return False
    if spec.term is not None and origin.term != spec.term:
        return False
    if spec.component is not None:
        target = spec.component if spec.level is None else f"{spec.component}[{spec.level}]"
        if target not in origin.components:
            return False
    elif spec.level is not None and str(origin.level) != spec.level:
        return False
    return any(v is not None for v in (spec.name, spec.term, spec.component, spec.level))

@dataclass(frozen=True, slots=True)
class ColumnOrigin:
    name: str
    term: str
    kind: str
    components: tuple[str, ...] = ()
    level: str | None = None


@dataclass(slots=True)
class DesignMatrix:
    values: np.ndarray
    names: tuple[str, ...]
    origins: tuple[ColumnOrigin, ...]
    requested_names: tuple[str, ...] = ()
    requested_origins: tuple[ColumnOrigin, ...] = ()
    structural_plan: StructuralCollinearityPlan | None = None
    structural_terms: tuple[StructuralTerm, ...] = ()
    user_omissions: tuple[UserOmission, ...] = ()
    prediction_terms: tuple[DesignTermState, ...] = ()

    @property
    def ncols(self) -> int:
        return int(self.values.shape[1])


def factor(variable: Any, *, base: Any | None = None, drop_base: bool = True,
           name: str | None = None) -> Factor:
    return Factor(variable, base=base, drop_base=drop_base, name=name)


def reg_interaction(*parts: Any, name: str | None = None) -> RegressorInteraction:
    return RegressorInteraction(*parts, name=name)


def _factor_from_fv_atom(atom: _FVAtom) -> Any:
    if atom.kind == "continuous":
        return atom.variable
    if atom.kind != "factor":
        raise ValueError(f"unknown factor-variable atom kind {atom.kind!r}")
    if atom.base_mode == "none":
        return Factor(atom.variable, drop_base=False)
    if atom.base_mode == "exact":
        return Factor(atom.variable, base=atom.base_value, drop_base=True)
    if atom.base_mode == "first":
        return Factor(atom.variable, base=_FactorBaseSelector("first"), drop_base=True)
    if atom.base_mode in {"last", "freq"}:
        return Factor(atom.variable, base=_FactorBaseSelector(atom.base_mode), drop_base=True)
    raise ValueError(f"unknown factor-variable base mode {atom.base_mode!r}")


def _expand_factor_variable_spec(spec: FactorVariableExpression) -> list[Any]:
    out: list[Any] = []
    for term in spec.terms:
        parts = tuple(_factor_from_fv_atom(atom) for atom in term.atoms)
        if len(parts) == 1:
            out.append(parts[0])
        else:
            out.append(RegressorInteraction(*parts))
    return out


def _column(data, spec, row_mask=None):
    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"column name {spec!r} requires data")
        a = np.asarray(data[spec])
    else:
        a = np.asarray(spec)
    return a if row_mask is None else a[row_mask]


def _label(spec: Any, fallback: str) -> str:
    return spec if isinstance(spec, str) else fallback


def _token(spec: Any, fallback: str) -> str:
    if isinstance(spec, str):
        return f"column:{spec}"
    return f"array:{id(spec)}:{fallback}"


def _flatten_parts(spec) -> list[Any]:
    if isinstance(spec, RegressorInteraction):
        out = []
        for p in spec.parts:
            out.extend(_flatten_parts(p))
        return out
    return [spec]


@dataclass(slots=True)
class _FactorEncoding:
    root: str
    token: str
    levels: np.ndarray
    codes: np.ndarray
    selected: np.ndarray


@dataclass(slots=True)
class _Blueprint:
    index: int
    name: str
    kind: str
    codes: np.ndarray
    n_levels: int
    active: np.ndarray
    level_names: tuple[str, ...]
    origins: tuple[ColumnOrigin, ...]
    continuous_specs: tuple[Any, ...]
    continuous_labels: tuple[str, ...]
    continuous_tokens: tuple[str, ...]
    continuous_signature: tuple[str, ...]
    categorical_tokens: tuple[str, ...]
    categorical_codes: tuple[np.ndarray, ...]
    categorical_encodings: tuple[_FactorEncoding, ...]
    categorical_cells: tuple[tuple[int, ...], ...]


def _factor_encoding(data, term: Factor, n_active: int, fallback: str, row_mask, cache) -> _FactorEncoding:
    key = (_token(term.variable, fallback), term.base, bool(term.drop_base), term.name)
    if key in cache:
        return cache[key]
    raw = require_identifier(_column(data, term.variable, row_mask), name=str(term.name or _label(term.variable, fallback)), role=VariableRole.FACTOR)
    if len(raw) != n_active:
        raise ValueError("regressor length mismatch")
    levels, codes = np.unique(raw, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int32)
    selected = np.ones(len(levels), dtype=bool)
    if term.drop_base and len(levels):
        if term.base is None:
            base_idx = 0
        elif isinstance(term.base, _FactorBaseSelector):
            if term.base.mode == "first":
                base_idx = 0
            elif term.base.mode == "last":
                base_idx = len(levels) - 1
            elif term.base.mode == "freq":
                counts = np.bincount(codes.astype(np.int64), minlength=len(levels))
                base_idx = int(np.flatnonzero(counts == counts.max())[0])
            else:
                raise ValueError(f"unknown factor base selector {term.base.mode!r}")
        else:
            hits = np.flatnonzero(levels == term.base)
            if not len(hits):
                raise ValueError(f"base level {term.base!r} is not observed")
            base_idx = int(hits[0])
        selected[base_idx] = False
    root = str(term.name or _label(term.variable, fallback))
    token = _token(term.variable, fallback)
    enc = _FactorEncoding(root, token, levels, codes, selected)
    cache[key] = enc
    return enc


def _joint_partition(encs: list[_FactorEncoding], n: int):
    if not encs:
        return np.zeros(n, dtype=np.int32), np.ones(1, dtype=bool), [()], [()]
    if len(encs) == 1:
        e = encs[0]
        tuples = [(j,) for j in range(len(e.levels))]
        labels = [(str(e.levels[j]),) for j in range(len(e.levels))]
        return e.codes, e.selected.copy(), tuples, labels

    # Fast mixed-radix key for observed categorical cells.  This avoids
    # materializing string/MultiIndex interactions and preserves lexicographic
    # component order. Fall back to a structured unique key if uint64 would
    # overflow for extremely high-order interactions.
    max_u64 = np.iinfo(np.uint64).max
    prod = 1
    safe = True
    for e in encs:
        base = max(len(e.levels), 1)
        if prod > max_u64 // base:
            safe = False; break
        prod *= base
    if safe:
        key = np.zeros(n, dtype=np.uint64)
        for e in encs:
            key = key * np.uint64(max(len(e.levels), 1)) + e.codes.astype(np.uint64)
        uniq, inv = np.unique(key, return_inverse=True)
        level_tuples = []
        for u in uniq.tolist():
            vals = [0] * len(encs)
            z = int(u)
            for j in range(len(encs)-1, -1, -1):
                base = max(len(encs[j].levels), 1)
                vals[j] = z % base
                z //= base
            level_tuples.append(tuple(vals))
    else:
        rec = np.empty(n, dtype=[(f"f{j}", np.int32) for j in range(len(encs))])
        for j, e in enumerate(encs):
            rec[f"f{j}"] = e.codes
        uniq, inv = np.unique(rec, return_inverse=True)
        level_tuples = [tuple(int(row[f"f{j}"]) for j in range(len(encs))) for row in uniq]

    active = np.array([
        all(encs[j].selected[t[j]] for j in range(len(encs)))
        for t in level_tuples
    ], dtype=bool)
    labels = [tuple(str(encs[j].levels[t[j]]) for j in range(len(encs))) for t in level_tuples]
    return np.asarray(inv, dtype=np.int32), active, level_tuples, labels


def _root_name(spec, parts, fallback):
    if isinstance(spec, Factor):
        return str(spec.name or _label(spec.variable, fallback))
    if isinstance(spec, RegressorInteraction) and spec.name:
        return str(spec.name)
    labels = []
    for j, p in enumerate(parts):
        if isinstance(p, Factor):
            labels.append(str(p.name or _label(p.variable, f"{fallback}_{j+1}")))
        else:
            labels.append(_label(p, f"{fallback}_{j+1}"))
    return "#".join(labels) if labels else fallback


def _blueprint(data, spec, n_active: int, fallback: str, row_mask, cache, index: int) -> _Blueprint:
    atoms = _flatten_parts(spec)
    factors = [p for p in atoms if isinstance(p, Factor)]
    continuous = [p for p in atoms if not isinstance(p, Factor)]
    encs = [
        _factor_encoding(data, p, n_active, f"{fallback}_f{j+1}", row_mask, cache)
        for j, p in enumerate(factors)
    ]
    codes, active, tuples, labels = _joint_partition(encs, n_active)
    root = _root_name(spec, atoms, fallback)
    cont_labels = tuple(_label(p, f"{fallback}_c{j+1}") for j, p in enumerate(continuous))
    cont_tokens = tuple(_token(p, f"{fallback}_c{j+1}") for j, p in enumerate(continuous))
    cont_sig = tuple(sorted(cont_tokens))

    level_names = []
    origins = []
    if encs:
        for lev, labs in enumerate(labels):
            comps = tuple(f"{encs[j].root}[{labs[j]}]" for j in range(len(encs))) + cont_labels
            name = "#".join(comps)
            level_names.append(name)
            kind = "factor" if isinstance(spec, Factor) else "interaction"
            origins.append(ColumnOrigin(name, root, kind, comps, "#".join(labs)))
    else:
        # Preserve the expanded-column name independently of a custom term
        # label: reg_interaction("x1", "x2", name="prod") still exposes the
        # numeric column as ``x1#x2`` while provenance.term is ``prod``.
        name = "#".join(cont_labels) if isinstance(spec, RegressorInteraction) else root
        level_names = [name]
        kind = "continuous" if not isinstance(spec, RegressorInteraction) else "interaction"
        origins = [ColumnOrigin(name, root, kind, cont_labels)]

    return _Blueprint(
        index=index, name=root, kind=origins[0].kind if origins else "empty",
        codes=np.asarray(codes, dtype=np.int32), n_levels=len(active), active=np.asarray(active, dtype=bool),
        level_names=tuple(level_names), origins=tuple(origins),
        continuous_specs=tuple(continuous), continuous_labels=cont_labels,
        continuous_tokens=cont_tokens, continuous_signature=cont_sig,
        categorical_tokens=tuple(e.token for e in encs),
        categorical_codes=tuple(e.codes for e in encs),
        categorical_encodings=tuple(encs), categorical_cells=tuple(tuple(int(v) for v in cell) for cell in tuples),
    )


def _materialize(data, bp: _Blueprint, row_mask, n_active: int):
    active_levels = np.flatnonzero(bp.active)
    if not len(active_levels):
        return [], [], []
    monomial = np.ones(n_active, dtype=np.float64)
    for p in bp.continuous_specs:
        arr = require_numeric(_column(data, p, row_mask), name=_label(p, bp.name), role=VariableRole.REGRESSOR, ndim=1)
        if len(arr) != n_active:
            from .errors import ShapeError
            raise ShapeError("each scalar regressor term must match y", details={"term": _label(p, bp.name), "expected": int(n_active), "actual": int(len(arr))})
        monomial *= arr

    cols, names, origins = [], [], []
    if bp.n_levels == 1 and not bp.continuous_specs:
        # A global constant is allowed as an explicit array-like term, but a
        # Factor with one omitted base simply has no active levels above.
        monomial = np.ones(n_active, dtype=np.float64)
    for lev in active_levels:
        if bp.n_levels == 1 and not bp.level_names[lev].endswith("]"):
            col = monomial.copy()
        else:
            col = np.zeros(n_active, dtype=np.float64)
            hit = bp.codes == lev
            col[hit] = monomial[hit]
        cols.append(col)
        names.append(bp.level_names[lev])
        origins.append(bp.origins[lev])
    return cols, names, origins


@dataclass(slots=True)
class _CompiledStructuredDesign:
    n_total: int
    n_active: int
    row_mask: np.ndarray | None
    blueprints: tuple[_Blueprint, ...]
    requested_names: tuple[str, ...]
    requested_origins: tuple[ColumnOrigin, ...]
    structural_plan: StructuralCollinearityPlan | None
    terms: tuple[StructuralTerm, ...]
    user_omissions: tuple[UserOmission, ...]

    @property
    def names(self) -> tuple[str, ...]:
        out = []
        for bp in self.blueprints:
            out.extend(bp.level_names[int(lev)] for lev in np.flatnonzero(bp.active))
        return tuple(out)

    @property
    def origins(self) -> tuple[ColumnOrigin, ...]:
        out = []
        for bp in self.blueprints:
            out.extend(bp.origins[int(lev)] for lev in np.flatnonzero(bp.active))
        return tuple(out)


def _normalize_structured_specs(specs):
    if isinstance(specs, (str, Factor, RegressorInteraction, FactorVariableExpression)) or (
        isinstance(specs, np.ndarray) and specs.ndim == 1
    ):
        specs = [specs]
    specs = list(specs)
    expanded_specs: list[Any] = []
    for spec in specs:
        if isinstance(spec, FactorVariableExpression):
            expanded_specs.extend(_expand_factor_variable_spec(spec))
        else:
            expanded_specs.append(spec)
    return expanded_specs


def is_heterogeneous_spec_candidate(specs) -> bool:
    """Cheap syntax-level gate for the heterogeneous-spec execution planner.

    Plain continuous column lists stay on the established dense path without
    paying a topology-analysis pass. Factor expansions and explicit regressor
    interactions are the common empirical cases where structural zeros can be
    large enough to justify pre-materialization planning.
    """
    if isinstance(specs, (Factor, RegressorInteraction, FactorVariableExpression)):
        return True
    if specs is None or isinstance(specs, str):
        return False
    if isinstance(specs, np.ndarray):
        return False
    try:
        return any(isinstance(s, (Factor, RegressorInteraction, FactorVariableExpression)) for s in specs)
    except TypeError:
        return False


def _compile_structured_design(
    data,
    specs,
    n: int,
    *,
    prefix: str,
    row_mask,
    absorbed_groups,
    absorbed_names,
    structural: bool,
    protected_terms,
    omit,
) -> _CompiledStructuredDesign:
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if len(row_mask) != n:
            raise ValueError("row_mask length must match design rows")
        n_active = int(np.count_nonzero(row_mask))
    else:
        n_active = int(n)

    specs = _normalize_structured_specs(specs)
    if not specs:
        return _CompiledStructuredDesign(
            int(n), n_active, row_mask, (), (), (), None, (), (),
        )

    cache = {}
    blueprints = [
        _blueprint(data, spec, n_active, f"{prefix}{j+1}", row_mask, cache, j)
        for j, spec in enumerate(specs)
    ]
    requested_names = []
    requested_origins = []
    terms = []
    for j, bp in enumerate(blueprints):
        # StructuralTerm index must match list position for order-preserving
        # comparisons even if a prior term has zero requested columns.
        bp.index = j
        active_levels = np.flatnonzero(bp.active)
        requested_names.extend(bp.level_names[lev] for lev in active_levels)
        requested_origins.extend(bp.origins[lev] for lev in active_levels)
        terms.append(StructuralTerm(
            j, bp.name, bp.codes, bp.n_levels, bp.active,
            bp.level_names, bp.continuous_signature, bp.kind,
            bp.categorical_tokens, bp.categorical_codes,
        ))

    selectors = _normalize_omit_specs(omit)
    matched = [False] * len(selectors)
    user_omissions = []
    requested_idx = 0
    for bp in blueprints:
        for lev in np.flatnonzero(bp.active):
            origin = bp.origins[int(lev)]
            for q, sel in enumerate(selectors):
                if _omit_matches(sel, origin):
                    matched[q] = True
                    bp.active[int(lev)] = False
                    user_omissions.append(UserOmission(
                        requested_idx, origin.name, origin.term, origin.kind,
                        origin.components, origin.level, sel,
                    ))
                    break
            requested_idx += 1
    if selectors and not all(matched):
        missing = [selectors[i] for i, hit in enumerate(matched) if not hit]
        raise ValueError(f"omit selector(s) matched no requested columns: {missing}")

    structural_plan = None
    if structural and terms:
        structural_plan = plan_structural_collinearity(
            terms, absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
            protected_terms=protected_terms,
        )
        structural_plan = replace(structural_plan, requested_columns=len(requested_names))
        # The planner mutates each term.active in place; blueprints share those
        # masks by construction.

    return _CompiledStructuredDesign(
        int(n), n_active, row_mask, tuple(blueprints),
        tuple(requested_names), tuple(requested_origins), structural_plan,
        tuple(terms), tuple(user_omissions),
    )


def _prediction_source(token: str) -> str | None:
    token = str(token)
    return token[len("column:"):] if token.startswith("column:") else None


def _python_level(value):
    return value.item() if isinstance(value, np.generic) else value


def _prediction_terms(compiled: _CompiledStructuredDesign) -> tuple[DesignTermState, ...]:
    out = []
    for bp in compiled.blueprints:
        categorical = tuple(
            CategoricalEncodingState(
                name=str(enc.root),
                source=_prediction_source(enc.token),
                levels=enc.levels,
                selected=enc.selected,
            )
            for enc in bp.categorical_encodings
        )
        continuous = tuple(
            PredictionInput(str(label), _prediction_source(token))
            for label, token in zip(bp.continuous_labels, bp.continuous_tokens, strict=False)
        )
        cell_codes = (
            np.asarray(bp.categorical_cells, dtype=np.int32)
            if bp.categorical_encodings else None
        )
        out.append(DesignTermState(
            name=str(bp.name),
            kind=str(bp.kind),
            column_names=tuple(str(x) for x in bp.level_names),
            active_mask=tuple(bool(x) for x in bp.active.tolist()),
            categorical=categorical,
            continuous=continuous,
            cell_codes=cell_codes,
        ))
    return tuple(out)


def _materialize_compiled_dense(data, compiled: _CompiledStructuredDesign) -> DesignMatrix:
    cols, names, origins = [], [], []
    for bp in compiled.blueprints:
        c, nm, org = _materialize(data, bp, compiled.row_mask, compiled.n_active)
        cols.extend(c); names.extend(nm); origins.extend(org)
    values = np.column_stack(cols) if cols else np.empty((compiled.n_active, 0), dtype=np.float64)
    return DesignMatrix(
        values, tuple(names), tuple(origins), compiled.requested_names,
        compiled.requested_origins, compiled.structural_plan, compiled.terms,
        compiled.user_omissions, _prediction_terms(compiled),
    )


def _materialize_compiled_block(data, compiled: _CompiledStructuredDesign, structure):
    """Materialize only certified component-local dense payloads.

    This is intentionally private while the representation contract is being
    validated.  Crucially, it never creates the global N x K expanded matrix.
    """
    from .compute.block_design import BlockDesign, DenseDesignBlock

    if not structure.certified_block_separable:
        raise ValueError("block materialization requires a certified structural plan")
    if structure.nobs != compiled.n_active:
        raise ValueError("structural plan row count does not match compiled design")

    # Global output-column order exactly matches dense materialization order.
    sources = []
    for bp_index, bp in enumerate(compiled.blueprints):
        sources.extend((bp_index, int(lev)) for lev in np.flatnonzero(bp.active))
    if len(sources) != structure.ncols:
        raise ValueError("structural term column count does not match block plan")

    if compiled.row_mask is None:
        active_source_rows = np.arange(compiled.n_active, dtype=np.int64)
    else:
        active_source_rows = np.flatnonzero(compiled.row_mask).astype(np.int64, copy=False)

    numeric_cache = {}

    def numeric_source(spec, bp_name):
        key = _token(spec, bp_name)
        hit = numeric_cache.get(key)
        if hit is not None:
            return hit
        arr = require_numeric(
            _column(data, spec), name=_label(spec, bp_name),
            role=VariableRole.REGRESSOR, ndim=1,
        )
        if len(arr) != compiled.n_total:
            from .errors import ShapeError
            raise ShapeError(
                "each scalar regressor term must match y",
                details={"term": _label(spec, bp_name), "expected": compiled.n_total, "actual": int(len(arr))},
            )
        numeric_cache[key] = arr
        return arr

    blocks = []
    rb = np.asarray(structure.row_blocks, dtype=np.int32)
    for meta in structure.blocks:
        rows = np.flatnonzero(rb == int(meta.index)).astype(np.int64, copy=False)
        cols = np.fromiter(meta.columns, dtype=np.int64, count=len(meta.columns))
        vals = np.zeros((len(rows), len(cols)), dtype=np.float64)
        source_rows = active_source_rows[rows]
        monomial_cache = {}
        for local_j, global_j in enumerate(cols.tolist()):
            bp_index, lev = sources[int(global_j)]
            bp = compiled.blueprints[bp_index]
            monomial = monomial_cache.get(bp_index)
            if monomial is None:
                monomial = np.ones(len(rows), dtype=np.float64)
                for spec in bp.continuous_specs:
                    monomial *= numeric_source(spec, bp.name)[source_rows]
                monomial_cache[bp_index] = monomial
            if bp.categorical_tokens:
                hit = bp.codes[rows] == int(lev)
                vals[hit, local_j] = monomial[hit]
            else:
                vals[:, local_j] = monomial
        blocks.append(DenseDesignBlock(int(meta.index), rows, cols, vals))
    return BlockDesign._from_trusted(compiled.n_active, structure.ncols, tuple(blocks))


@dataclass(slots=True)
class _BlockDesignMatrix:
    values: Any
    names: tuple[str, ...]
    origins: tuple[ColumnOrigin, ...]
    requested_names: tuple[str, ...]
    requested_origins: tuple[ColumnOrigin, ...]
    structural_plan: StructuralCollinearityPlan | None
    structural_terms: tuple[StructuralTerm, ...]
    user_omissions: tuple[UserOmission, ...]
    execution_structure: Any
    prediction_terms: tuple[DesignTermState, ...] = ()

    @property
    def ncols(self) -> int:
        return int(self.values.ncols)


def _build_heterogeneous_design(
    data,
    specs,
    n: int,
    *,
    groups=(),
    prefix: str = "x",
    row_mask=None,
    absorbed_groups=(),
    absorbed_names=(),
    structural: bool = True,
    protected_terms=(),
    omit=None,
):
    """Compile a certified heterogeneous specification directly into BlockDesign.

    Factor/interactions retain symbolic support until the physical storage
    representation is selected. Plain caller-supplied 2D arrays intentionally
    remain on the established dense path.
    """
    if specs is None or (isinstance(specs, np.ndarray) and specs.ndim == 2):
        raise ValueError("pre-materialization block compilation requires structured regressor specs")
    compiled = _compile_structured_design(
        data, specs, n, prefix=prefix, row_mask=row_mask,
        absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
        structural=structural, protected_terms=protected_terms, omit=omit,
    )
    from .compute.design_plan import analyze_execution_structure
    structure = analyze_execution_structure(compiled.terms, groups=groups)
    values = _materialize_compiled_block(data, compiled, structure)
    return _BlockDesignMatrix(
        values, compiled.names, compiled.origins, compiled.requested_names,
        compiled.requested_origins, compiled.structural_plan, compiled.terms,
        compiled.user_omissions, structure, _prediction_terms(compiled),
    )


@dataclass(slots=True)
class _ExecutionDesignMatrix:
    values: Any
    names: tuple[str, ...]
    origins: tuple[ColumnOrigin, ...]
    requested_names: tuple[str, ...]
    requested_origins: tuple[ColumnOrigin, ...]
    structural_plan: StructuralCollinearityPlan | None
    structural_terms: tuple[StructuralTerm, ...]
    user_omissions: tuple[UserOmission, ...]
    execution_structure: Any
    storage_plan: Any
    prediction_terms: tuple[DesignTermState, ...] = ()

    @property
    def ncols(self) -> int:
        if hasattr(self.values, "ncols"):
            return int(self.values.ncols)
        return int(self.values.shape[1])


def _compile_execution_design(
    data,
    specs,
    n: int,
    *,
    groups=(),
    expected_passes: int = 1,
    memory_budget_mb: float | None = None,
    prefix: str = "x",
    row_mask=None,
    absorbed_groups=(),
    absorbed_names=(),
    structural: bool = True,
    protected_terms=(),
    omit=None,
):
    """Internal heterogeneous-specification compiler + storage planner.

    Structure is certified before physical materialization. The planner selects
    dense or row-partitioned storage without estimator-specific parsing rules;
    unsupported or unprofitable specifications remain dense automatically.
    """
    if specs is None or (isinstance(specs, np.ndarray) and specs.ndim == 2):
        dense = build_design(
            data, specs, n, prefix=prefix, row_mask=row_mask,
            absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
            structural=structural, protected_terms=protected_terms, omit=omit,
        )
        from .compute.execution import DesignStoragePlan
        payload = int(dense.values.nbytes)
        storage = DesignStoragePlan(
            "dense", max(1, int(expected_passes)), payload, payload, 0.0,
            "caller_supplied_dense_or_empty_design",
        )
        return _ExecutionDesignMatrix(
            dense.values, dense.names, dense.origins, dense.requested_names,
            dense.requested_origins, dense.structural_plan, dense.structural_terms,
            dense.user_omissions, None, storage, dense.prediction_terms,
        )

    compiled = _compile_structured_design(
        data, specs, n, prefix=prefix, row_mask=row_mask,
        absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
        structural=structural, protected_terms=protected_terms, omit=omit,
    )
    from .compute.execution import DesignStoragePlan, plan_design_storage

    # Topology analysis is itself O(N x terms). For a small mixed design
    # with only a handful of localized columns and several globally supported
    # controls, the maximum block-storage payoff is too small to amortize a
    # full connectivity scan over the estimation sample. This is common after
    # structural collinearity removes absorbed factor/event-study controls.
    # Keep the shortcut deliberately narrow: purely localized designs, designs
    # with only one global bridge, wider designs, repeated-pass workloads, and
    # memory-pressured designs still reach the exact block planner.
    active_cols = len(compiled.names)
    dense_bytes = int(compiled.n_active) * int(active_cols) * 8
    group_count = len(tuple(groups or ()))
    budget_bytes = None if memory_budget_mb is None else int(float(memory_budget_mb) * 1024**2)
    global_cols = sum(
        int(np.count_nonzero(term.active))
        for term in compiled.terms
        if not term.categorical_components
    )
    localized_cols = active_cols - global_cols
    small_mixed_dense = (
        group_count >= 2 and active_cols <= 8 and global_cols >= 2
        and localized_cols <= 4 and max(1, int(expected_passes)) <= 2
        and dense_bytes <= 32 * 1024**2
        and (budget_bytes is None or dense_bytes <= budget_bytes)
    )
    if small_mixed_dense:
        dense = _materialize_compiled_dense(data, compiled)
        storage = DesignStoragePlan(
            "dense", max(1, int(expected_passes)), dense_bytes, dense_bytes, 0.0,
            "small_mixed_design_avoids_topology_setup",
        )
        return _ExecutionDesignMatrix(
            dense.values, compiled.names, compiled.origins, compiled.requested_names,
            compiled.requested_origins, compiled.structural_plan, compiled.terms,
            compiled.user_omissions, None, storage, _prediction_terms(compiled),
        )

    from .compute.design_plan import analyze_execution_structure
    structure = analyze_execution_structure(compiled.terms, groups=groups)
    storage = plan_design_storage(
        structure, expected_passes=expected_passes,
        memory_budget_mb=memory_budget_mb,
    )
    if storage.representation == "block_dense":
        values = _materialize_compiled_block(data, compiled, structure)
    else:
        values = _materialize_compiled_dense(data, compiled).values
    return _ExecutionDesignMatrix(
        values, compiled.names, compiled.origins, compiled.requested_names,
        compiled.requested_origins, compiled.structural_plan, compiled.terms,
        compiled.user_omissions, structure, storage, _prediction_terms(compiled),
    )


# Backward-private aliases retained for older internal tests/checkpoints.


def build_design(
    data,
    specs,
    n: int,
    *,
    prefix: str = "x",
    row_mask=None,
    absorbed_groups=(),
    absorbed_names=(),
    structural: bool = True,
    protected_terms=(),
    omit=None,
) -> DesignMatrix:
    """Compile regressors with optional exact structural rank reduction.

    The compiler first creates compact factor/interaction partition metadata,
    applies conservative symbolic dependencies, and only then materializes the
    surviving N x K numeric columns. A post-absorption Gram-rank resolver still
    provides the final numerical safety net.
    """
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if len(row_mask) != n:
            raise ValueError("row_mask length must match design rows")
        n_active = int(np.count_nonzero(row_mask))
    else:
        n_active = int(n)

    if specs is None:
        return DesignMatrix(np.empty((n_active, 0), dtype=np.float64), (), ())
    if isinstance(specs, np.ndarray) and specs.ndim == 2:
        a = np.asarray(specs if row_mask is None else specs[row_mask], dtype=np.float64)
        names = tuple(f"{prefix}{j+1}" for j in range(a.shape[1]))
        origins = tuple(ColumnOrigin(name, name, "array") for name in names)
        selectors = _normalize_omit_specs(omit)
        matched = [False] * len(selectors)
        keep = np.ones(len(names), dtype=bool)
        user_omissions = []
        for j, origin in enumerate(origins):
            for q, sel in enumerate(selectors):
                if _omit_matches(sel, origin):
                    keep[j] = False; matched[q] = True
                    user_omissions.append(UserOmission(j, origin.name, origin.term, origin.kind, origin.components, origin.level, sel))
                    break
        if selectors and not all(matched):
            missing = [selectors[i] for i, hit in enumerate(matched) if not hit]
            raise ValueError(f"omit selector(s) matched no requested columns: {missing}")
        active_names = tuple(n for n, k in zip(names, keep) if k)
        active_origins = tuple(o for o, k in zip(origins, keep) if k)
        # Boolean column indexing always copies. Preserve the caller's dense
        # NumPy matrix when no column was actually omitted; this matters at
        # 10M+ rows where an unnecessary X copy can approach a gigabyte.
        values = a if bool(np.all(keep)) else a[:, keep]
        prediction_terms = tuple(
            DesignTermState(
                name=name,
                kind="array",
                column_names=(name,),
                active_mask=(bool(keep[j]),),
                continuous=(PredictionInput(name, None),),
            )
            for j, name in enumerate(names)
        )
        return DesignMatrix(
            values, active_names, active_origins, names, origins, None, (),
            tuple(user_omissions), prediction_terms,
        )
    compiled = _compile_structured_design(
        data, specs, n, prefix=prefix, row_mask=row_mask,
        absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
        structural=structural, protected_terms=protected_terms, omit=omit,
    )
    return _materialize_compiled_dense(data, compiled)
