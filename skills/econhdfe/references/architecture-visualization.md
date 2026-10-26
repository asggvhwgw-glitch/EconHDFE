# Architecture visualization and code maps

Use this reference when a user asks to visualize the econhdfe codebase, explain package architecture, build a dependency diagram, create contributor-onboarding material, or verify that architecture documentation still matches source code.

## Contents

- Evidence-first workflow
- Generated outputs
- Static versus interactive views
- Fidelity rules
- Regeneration and release checks

## Evidence-first workflow

Do not draw the architecture from memory. Start from the source tree and recover the real implementation topology.

For an econhdfe source checkout:

```bash
python skills/econhdfe/scripts/architecture_map.py
```

When this skill is installed separately from the repository, point it at the source root explicitly:

```bash
python scripts/architecture_map.py --root /path/to/econhdfe-source
```

The generator uses Python AST import analysis. Package `__init__.py` re-export imports are recorded in the machine evidence but excluded from collapsed dependency counts so the visualization represents implementation coupling rather than namespace wiring.

## Generated outputs

The canonical repository outputs live under `docs/development/architecture-map/`:

- `architecture.json` — machine-readable modules, collapsed AST edges, public exports, repository-area counts, invariants and reviewed execution-flow overlays;
- `architecture.md` — a compact codemap plus Mermaid dependency and execution-flow diagrams suitable for GitHub and grep/search;
- `architecture.html` — a self-contained interactive view with flow filters, node details, dependency inspection and a repository-area summary.

The three outputs must describe the same architecture. Do not edit one output manually to make it look better; change the source classification/layout/flow declaration in `architecture_map.py` and regenerate all three.

## Static versus interactive views

Use `architecture.md` when the artifact must remain diffable, reviewable in a pull request, printable, or readable without JavaScript. It should answer “where does X live?”, “what depends on Y?”, and “which boundary owns this behavior?” before adding visual decoration.

Use `architecture.html` for onboarding, workshops, architecture reviews and exploratory browsing. The interactive view may highlight semantic execution flows, but those flows must be labeled separately from import dependencies.

Do not replace the static Markdown companion with HTML alone. A visualization is not a substitute for searchable architecture documentation.

## Fidelity rules

1. Recover topology from code/configuration before drawing it. Never invent a module, dependency, datastore or execution hop to fill visual space.
2. Keep import topology and execution/data flow semantically separate. Imports are AST-derived evidence; execution flows are reviewed overlays.
3. Prefer stable architectural groups over file-by-file spaghetti. File-level evidence stays available in JSON/node details.
4. Keep role categories small and meaningful: public/interface surface, estimator model, reusable infrastructure, low-level compute kernel.
5. Preserve architectural invariants in prose as well as diagrams, especially deliberate absences such as generic `iv` not depending on outcome models.
6. If a flow cannot be verified from code, mark it partial or omit it. Do not infer a framework-default path.
7. The HTML must remain self-contained: no remote scripts, fonts, analytics or network fetches.

## Visual layout requirements

The generated diagram is documentation, not decoration. Keep the visual layer readable at normal browser zoom and on narrower screens:

- size architecture cards from declared layout metadata rather than assuming one fixed box width;
- wrap node titles to at most two balanced lines and never let SVG text overflow the card boundary;
- route dependency edges to card boundaries/ports instead of drawing center-to-center lines through node text;
- allocate separate edge ports per node side, use rounded orthogonal corridors, and keep a small terminal gap so arrowheads remain fully visible outside cards;
- use restrained open-chevron arrowheads and neutral edge halos at crossings; reserve stronger strokes for the selected execution flow;
- use stable layer bands (surface, models, infrastructure, kernel) to create visual hierarchy and whitespace;
- make the default `Overview` intentionally sparse, with the complete AST dependency graph available as a separate control;
- keep long file paths in the detail panel breakable with `overflow-wrap`, rather than widening the whole layout;
- prefer larger cards and fewer default edges over shrinking typography to fit more information.

Any change to these rules belongs in `architecture_map.py` and should include a regression test for bounding/layout semantics.

## Regeneration and release checks

After changing package boundaries/imports:

```bash
python scripts/generate_architecture_map.py
python scripts/generate_architecture_map.py --check
```

`--check` regenerates the expected outputs in memory and exits nonzero if committed architecture artifacts are stale.

For a documentation/tooling-only architecture-map update, a `MAJOR.MINOR.PATCH.REVISION` release is appropriate only when the public runtime contract is unchanged. The normal release compatibility gate still decides whether a revision is allowed.
