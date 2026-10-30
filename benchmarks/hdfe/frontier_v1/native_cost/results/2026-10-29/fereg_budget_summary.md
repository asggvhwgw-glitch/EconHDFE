# Native cost study summary

Times are seconds: median [min, max] across passed calls. Counts retain their native units.

## Full call timing and accuracy

| Family | Backend | L | Diag | Call | Statuses | Total s | Max forward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rigidity_grid | fereg | 320 | on | cold | {"passed": 1} | 8.34595 | 5.57674e-13 |
| rigidity_grid | fereg | 320 | on | warm | {"passed": 1} | 6.07391 | 4.33474e-13 |

## Within vector-call work (diagnostics on)

| Backend | L | Call | D apply | D adjoint | P apply | P adjoint | Deferred build s | Escalated | Local factors | Max factor dim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Fereg batch work and factor state (diagnostics on)

| L | Call | H batches | P batches | Exact P | Joint allocated entries | CSR nnz | Build outcome | Joint build / outer total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | cold | 70 | 71 | 0 | — | 12697600 | over the fill budget | 0.12675 |
| 320 | warm | 58 | 59 | 0 | — | 12697600 | over the fill budget | 0 |

## Fereg inclusive phase times; columns must not be added

| L | Call | Joint build | Aggregation | Pattern | Assembly | Analyze incl. ordering | Numeric | Refactor | Refresh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | cold | 1.05785 | 0.0297429 | 0.0328391 | 0.223474 | 0.761728 | 0 | 0 | 0 |
| 320 | warm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Cold total / warm total, paired by worker

| Backend | L | diagnostics | Paired ratio | Excluded pairs |
| --- | --- | --- | --- | --- |
| fereg | 320 | True | 1.37407 | 0 |

## Diagnostics on / off total, paired by replicate

| Backend | L | mode | Paired ratio | Excluded pairs |
| --- | --- | --- | --- | --- |
| fereg | 320 | cold | — | 1 |
| fereg | 320 | warm | — | 1 |

## Scope and repeat checks

- Worker/call counts: `{"call_statuses": {"passed": 2}, "recorded_calls": 2, "represented_call_outcomes": 2, "worker_cases": 1, "worker_statuses": {"passed": 1}}`.
- Factor entries are allocated values, including padding; symbolic flops are estimates.
- Snapshot constructor/cached-build times are metadata, not new costs on each warm call.
- Ratios describe these trials; diagnostics on/off does not measure the entire patch overhead.
- One exploratory trial per configuration: repeat consistency and timing variability cannot be assessed.
