from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..compute.block_design import BlockDesign, DenseDesignBlock
from ..compute.design_plan import StructuralDesignPlan
from .plan import FEPlan
from .weighted_projection import WeightedFEProjector
from .projection import resolve_threads
from ..planner import scheduler


@dataclass(slots=True)
class BlockProjectionResult:
    response: np.ndarray
    design: BlockDesign
    iterations: int
    converged: bool = True
    max_update: float = 0.0
    criterion: str = "component_projection"


class BlockWeightedFEProjector:
    """Exact FE projection over certified disconnected design components.

    The projector is keyed by the *row partition*, not by a particular set of
    surviving regressor columns.  This lets structural column filtering retain
    response-only components and reuse the same FE topology after columns are
    omitted.
    """

    def __init__(
        self,
        structure: StructuralDesignPlan,
        plan: FEPlan,
        *,
        engine: str,
        method: str = "map",
        max_iter: int = 16_000,
        require_convergence: bool = False,
        absorb_threads="auto",
        projection_memory_budget_mb: float = 512,
    ):
        if not structure.certified_block_separable:
            raise ValueError("block FE projection requires an exact block-separability certificate")
        rb = np.asarray(structure.row_blocks)
        parts = []
        for block in structure.blocks:
            rows = np.flatnonzero(rb == int(block.index)).astype(np.int64, copy=False)
            parts.append((int(block.index), rows))
        self._init_parts(
            parts, plan, engine=engine, method=method, max_iter=max_iter, absorb_threads=absorb_threads,
            projection_memory_budget_mb=projection_memory_budget_mb,
            require_convergence=require_convergence,
        )

    @classmethod
    def from_design(
        cls,
        design: BlockDesign,
        plan: FEPlan,
        *,
        engine: str,
        method: str = "map",
        max_iter: int = 16_000,
        require_convergence: bool = False,
        absorb_threads="auto",
        projection_memory_budget_mb: float = 512,
    ):
        """Compile directly from a previously certified BlockDesign layout.

        ``BlockDesign`` preserves its certified row components when columns are
        filtered, including components with zero surviving explicit columns.
        This constructor is therefore the safe path after rank/separation sample
        transformations where the original StructuralDesignPlan is stale.
        """
        design._ensure_layout()
        obj = cls.__new__(cls)
        parts = [(int(b.index), np.asarray(b.rows, dtype=np.int64)) for b in design.blocks]
        seen = np.zeros(design.nobs, dtype=bool)
        for _, rows in parts:
            if np.any(rows < 0) or np.any(rows >= design.nobs) or np.any(seen[rows]):
                raise ValueError("BlockDesign rows must form disjoint in-range components")
            seen[rows] = True
        if design.nobs and not np.all(seen):
            raise ValueError("BlockDesign must cover every observation")
        obj._init_parts(
            parts, plan, engine=engine, method=method, max_iter=max_iter, absorb_threads=absorb_threads,
            projection_memory_budget_mb=projection_memory_budget_mb,
            require_convergence=require_convergence,
        )
        return obj

    def _init_parts(
        self, parts, plan: FEPlan, *, engine: str, method: str, max_iter: int, absorb_threads,
        projection_memory_budget_mb: float, require_convergence: bool,
    ) -> None:
        if plan.groups:
            max_row = max((int(np.max(rows)) for _, rows in parts if len(rows)), default=-1)
            total_rows = sum(len(rows) for _, rows in parts)
            if max_row >= plan.nobs or total_rows != plan.nobs:
                raise ValueError("FE plan and block row partition have different observation counts")
        self.structure = None
        self.plan = plan
        self.engine = str(engine)
        self.method = str(method)
        self._parts = []
        for index, rows in parts:
            local_plan = plan.take(rows) if plan.groups else plan
            projector = None
            if local_plan.groups:
                projector = WeightedFEProjector(
                    local_plan.groups, engine=engine, method=method, max_iter=max_iter,
                    require_convergence=require_convergence,
                    absorb_threads=absorb_threads,
                    projection_memory_budget_mb=projection_memory_budget_mb,
                )
            self._parts.append((int(index), np.asarray(rows, dtype=np.int64), projector))

    def _validate_design(self, design: BlockDesign) -> None:
        if design.nobs != sum(len(rows) for _, rows, _ in self._parts):
            raise ValueError("BlockDesign observation count does not match projector layout")
        if len(design.blocks) != len(self._parts):
            raise ValueError("BlockDesign component count does not match projector layout")
        for b, (index, rows, _) in zip(design.blocks, self._parts, strict=True):
            if int(b.index) != index or not np.array_equal(b.rows, rows):
                raise ValueError("BlockDesign row layout does not match projector topology")

    def _prewarm_threads(self):
        largest = max(((len(rows), projector) for _, rows, projector in self._parts
                       if projector is not None), key=lambda item: item[0], default=None)
        if largest is not None:
            nobs, projector = largest
            resolve_threads(projector.absorb_threads_requested, nobs=nobs)

    def residualize_design(self, design: BlockDesign, weights, *, tol: float) -> BlockDesign:
        """Residualize block-local columns while preserving the row partition."""
        self._validate_design(design)
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != design.nobs:
            raise ValueError("weights must have one value per observation")
        # All block payloads remain live until the result is returned. Hold
        # their reservation across the loop, including local weight selection.
        output_bytes = sum(b.values.size * 8 for b in design.blocks)
        weight_bytes = max((len(rows) * 8 for _, rows, _ in self._parts), default=0)
        self._prewarm_threads()
        with scheduler.reserve_memory(output_bytes + weight_bytes, label="block_fe_outputs"):
            blocks = []
            for src, (_, rows, projector) in zip(design.blocks, self._parts, strict=True):
                work = np.array(src.values, dtype=np.float64, order="C", copy=True)
                if projector is not None and work.shape[1]:
                    work, _ = projector.residualize(
                        work, w[rows], tol=tol, return_info=True, copy=False
                    )
                blocks.append(DenseDesignBlock(src.index, rows, src.columns, work))
            return BlockDesign._from_trusted(design.nobs, design.ncols, tuple(blocks))

    def residualize_response_design(
        self, response, design: BlockDesign, weights, *, tol: float,
    ) -> BlockProjectionResult:
        """Residualize one response plus all block-local design columns."""
        self._validate_design(design)
        r = np.asarray(response, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if r.ndim != 1 or len(r) != design.nobs:
            raise ValueError("response must have one value per observation")
        if w.ndim != 1 or len(w) != design.nobs:
            raise ValueError("weights must have one value per observation")
        # Each returned design view retains its whole response+design work
        # array. Include those bases, the response output and row selections.
        output_bytes = design.nobs * 8 + sum(
            len(b.rows) * (b.values.shape[1] + 1) * 8 for b in design.blocks
        )
        selection_bytes = max((len(rows) * 16 for _, rows, _ in self._parts), default=0)
        self._prewarm_threads()
        with scheduler.reserve_memory(output_bytes + selection_bytes, label="block_fe_outputs"):
            out_r = np.empty(design.nobs, dtype=np.float64)
            blocks = []
            total_iterations = 0
            converged = True
            max_update = 0.0
            criteria = set()
            for src, (_, rows, projector) in zip(design.blocks, self._parts, strict=True):
                work = np.empty((len(rows), src.values.shape[1] + 1), dtype=np.float64)
                work[:, 0] = r[rows]
                if src.values.shape[1]:
                    work[:, 1:] = src.values
                if projector is not None:
                    work, info = projector.residualize(
                        work, w[rows], tol=tol, return_info=True, copy=False
                    )
                    total_iterations += 0 if info is None else int(info.iterations)
                    if info is not None:
                        converged = converged and bool(getattr(info, "converged", True))
                        max_update = max(max_update, float(getattr(info, "max_update", 0.0)))
                        criteria.add(str(getattr(info, "criterion", "component_projection")))
                out_r[rows] = work[:, 0]
                blocks.append(DenseDesignBlock(src.index, rows, src.columns, work[:, 1:]))
            return BlockProjectionResult(
                out_r, BlockDesign._from_trusted(design.nobs, design.ncols, tuple(blocks)),
                total_iterations, converged, max_update,
                next(iter(criteria)) if len(criteria) == 1 else "component_projection",
            )

    @property
    def resource_info(self) -> dict:
        details = []
        for index, rows, projector in self._parts:
            details.append({
                "block": index,
                "nobs": int(len(rows)),
                "projector": None if projector is None else projector.resource_info,
            })
        return {"block_count": len(details), "blocks": tuple(details)}
