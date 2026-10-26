from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math
import pandas as pd

from ..planner import plan_execution
from ..errors import SpecificationError
from .source import DataSource, as_data_source


def estimate_frame_bytes_per_row(frame: pd.DataFrame) -> float:
    n = max(len(frame), 1)
    # deep=True includes Python/string payload in the preview.  This is a
    # planning estimate, not an allocation guarantee.
    return max(float(frame.memory_usage(index=False, deep=True).sum()) / n, 1.0)


@dataclass(frozen=True, slots=True)
class DataIngestionPlan:
    required_columns: tuple[str, ...]
    memory_budget_bytes: int
    target_batch_bytes: int
    estimated_bytes_per_row: float
    batch_rows: int
    source_label: str
    source_type: str
    strategy: str = "scan"

    @property
    def projected_column_count(self) -> int:
        return len(self.required_columns)

    def as_dict(self, *, include_sensitive: bool = False) -> dict:
        out = {
            "required_column_count": len(self.required_columns),
            "memory_budget_bytes": self.memory_budget_bytes,
            "target_batch_bytes": self.target_batch_bytes,
            "estimated_bytes_per_row": self.estimated_bytes_per_row,
            "batch_rows": self.batch_rows,
            "source_type": self.source_type,
            "strategy": self.strategy,
        }
        if include_sensitive:
            out["required_columns"] = self.required_columns
            out["source_label"] = self.source_label
        return out


def plan_ingestion(
    source: DataSource,
    required_columns: Sequence[str],
    *,
    memory_budget_mb: float = 512,
    batch_fraction: float = 0.20,
    preview_rows: int = 2048,
    min_batch_rows: int = 4_096,
    max_batch_rows: int = 1_000_000,
) -> DataIngestionPlan:
    source = as_data_source(source)
    cols = tuple(dict.fromkeys(map(str, required_columns)))
    requested_budget = max(int(float(memory_budget_mb) * 1024**2), 1)
    execution = plan_execution(requested_budget_bytes=requested_budget)
    budget = int(execution.memory.effective_budget_bytes)
    frac = min(max(float(batch_fraction), 0.01), 0.80)
    target = max(int(budget * frac), 1)
    preview = source.preview(cols, nrows=max(int(preview_rows), 1))
    bpr = estimate_frame_bytes_per_row(preview)
    if bpr > budget:
        raise SpecificationError(
            "One estimated input row exceeds the available ingestion budget.",
            code="resource.memory", stage="data",
            details={"estimated_bytes_per_row": bpr, "budget_bytes": budget},
        )
    raw_rows = max(int(math.floor(target / bpr)), 1)
    # The preferred minimum is soft: it must never override the byte budget.
    batch_rows = min(max(int(max_batch_rows), 1), raw_rows)
    return DataIngestionPlan(
        required_columns=cols,
        memory_budget_bytes=budget,
        target_batch_bytes=target,
        estimated_bytes_per_row=float(bpr),
        batch_rows=int(batch_rows),
        source_label=source.label,
        source_type=type(source).__name__,
        strategy="scan",
    )
