#!/usr/bin/env python3
"""Generate an evidence-backed architecture map for an econhdfe source tree.

The import topology is recovered from Python ASTs. Higher-level execution flows are
small, explicit semantic overlays whose nodes map to the same recovered module groups.
The generated Markdown, JSON, and HTML are deterministic and suitable for release
artifacts and contributor onboarding.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


GROUPS = {
    "compat": {
        "label": "Compatibility shim",
        "role": "compat",
        "layer": "surface",
        "description": "Preserves older reghdfe-style Python workflows so empirical projects can migrate without changing the econometric specification.",
    },
    "interface": {
        "label": "Public interface & orchestration",
        "role": "surface",
        "layer": "surface",
        "description": "Turns an empirical specification into the appropriate OLS, IV, PPML or IV-PPML workflow and manages repeated-specification research workflows.",
    },
    "frontend": {
        "label": "Frontend validation",
        "role": "surface",
        "layer": "surface",
        "description": "Checks that outcomes, regressors, instruments, fixed effects, clusters and weights have economically valid roles before estimation.",
    },
    "data": {
        "label": "Econometric data layer",
        "role": "infra",
        "layer": "infrastructure",
        "description": "Projects, encodes and reuses only the raw observations/columns needed by the empirical specification while preserving sample identity.",
    },
    "planner": {
        "label": "Execution planner",
        "role": "infra",
        "layer": "infrastructure",
        "description": "Separates exactness certificates from resource/cost policy and chooses memory, representation and parallel execution without changing the estimand.",
    },
    "contracts": {
        "label": "Shared contracts",
        "role": "surface",
        "layer": "surface",
        "description": "Keeps estimator choices, inference conventions and paper-facing results consistent across model families.",
    },
    "ols": {
        "label": "OLS model",
        "role": "model",
        "layer": "models",
        "description": "Estimates linear partial effects while controlling for high-dimensional observed heterogeneity; supplies OLS-specific resampling inference.",
    },
    "linear_iv": {
        "label": "Linear IV model",
        "role": "model",
        "layer": "models",
        "description": "Handles endogenous regressors in linear models and reports whether excluded instruments provide credible identifying variation.",
    },
    "ppml": {
        "label": "PPML model",
        "role": "model",
        "layer": "models",
        "description": "Estimates nonnegative outcomes such as trade flows with multiplicative conditional means, high-dimensional FE and separation-safe inference.",
    },
    "ppml_iv": {
        "label": "IV-PPML model",
        "role": "model",
        "layer": "models",
        "description": "Addresses endogeneity in multiplicative/PPML models and finite-panel bias while retaining high-dimensional fixed effects.",
    },
    "model_shared": {
        "label": "Shared model primitives",
        "role": "model",
        "layer": "models",
        "description": "Provides the common Poisson mean/link calculations used by PPML and IV-PPML so both models implement the same economic outcome process.",
    },
    "hdfe": {
        "label": "HDFE infrastructure",
        "role": "infra",
        "layer": "infrastructure",
        "description": "Controls for large sets of fixed effects and heterogeneous FE structures without estimating/reporting every nuisance coefficient explicitly.",
    },
    "iv": {
        "label": "Generic IV primitives",
        "role": "infra",
        "layer": "infrastructure",
        "description": "Represents the exclusion restrictions and orthogonality conditions that identify endogenous effects, independently of the outcome model.",
    },
    "resampling": {
        "label": "Resampling engine",
        "role": "infra",
        "layer": "infrastructure",
        "description": "Runs bootstrap/resampling procedures used to quantify sampling uncertainty without embedding estimator-specific economic assumptions.",
    },
    "cluster_inference": {
        "label": "Cluster inference",
        "role": "infra",
        "layer": "infrastructure",
        "description": "Supports credible inference when shocks are correlated within economic groups and cluster counts/balance make asymptotics fragile.",
    },
    "compute": {
        "label": "Compute/runtime kernels",
        "role": "kernel",
        "layer": "kernel",
        "description": "Executes the sufficient-statistic and matrix operations behind estimation while exploiting data structure so large empirical designs remain feasible.",
    },
}

LAYOUT = {
    "compat": (40, 72, 190),
    "interface": (300, 72, 220),
    "frontend": (590, 72, 200),
    "contracts": (860, 72, 200),
    "data": (35, 165, 205),
    "planner": (300, 690, 205),
    "ols": (35, 270, 185),
    "linear_iv": (275, 270, 190),
    "ppml": (515, 270, 185),
    "ppml_iv": (755, 270, 190),
    "model_shared": (995, 270, 205),
    "hdfe": (210, 490, 210),
    "iv": (515, 490, 200),
    "resampling": (790, 490, 190),
    "cluster_inference": (1025, 490, 190),
    "compute": (665, 690, 220),
}
NODE_HEIGHT = 90


def _wrap_label(label: str, max_chars: int = 21) -> list[str]:
    """Wrap a node title into at most two balanced SVG-safe lines."""
    if len(label) <= max_chars:
        return [label]
    words = label.split()
    best = None
    for cut in range(1, len(words)):
        left = " ".join(words[:cut])
        right = " ".join(words[cut:])
        if max(len(left), len(right)) > max_chars + 4:
            continue
        score = abs(len(left) - len(right))
        if best is None or score < best[0]:
            best = (score, left, right)
    if best:
        return [best[1], best[2]]
    return [
        label[: max_chars - 1] + "…",
        label[max_chars - 1 : max_chars * 2 - 2] + ("…" if len(label) > max_chars * 2 - 2 else ""),
    ]


OVERVIEW_EDGES = [
    ["compat", "interface"],
    ["interface", "frontend"],
    ["interface", "data"],
    ["interface", "contracts"],
    ["interface", "ols"],
    ["interface", "linear_iv"],
    ["interface", "ppml"],
    ["interface", "ppml_iv"],
    ["ols", "hdfe"],
    ["linear_iv", "hdfe"],
    ["linear_iv", "iv"],
    ["ppml", "hdfe"],
    ["ppml", "model_shared"],
    ["ppml_iv", "ppml"],
    ["ppml_iv", "hdfe"],
    ["ppml_iv", "iv"],
    ["ppml_iv", "resampling"],
    ["interface", "cluster_inference"],
    ["cluster_inference", "resampling"],
    ["cluster_inference", "compute"],
    ["cluster_inference", "contracts"],
    ["data", "planner"],
    ["hdfe", "planner"],
    ["compute", "planner"],
    ["hdfe", "compute"],
    ["iv", "compute"],
    ["compute", "contracts"],
]

FLOWS = [
    {
        "id": "ols",
        "label": "OLS-HDFE",
        "description": "Estimate linear partial effects net of high-dimensional observed heterogeneity, then construct the requested inference.",
        "nodes": ["interface", "frontend", "ols", "hdfe", "compute", "contracts"],
        "steps": [
            ["interface", "frontend", "Validate variable roles and data contract"],
            ["frontend", "ols", "Dispatch the linear model specification"],
            ["ols", "hdfe", "Compile/absorb requested fixed effects"],
            ["hdfe", "compute", "Use grouped projection and numerical kernels"],
            ["compute", "contracts", "Assemble covariance and stable result/reporting contracts"],
            ["contracts", "interface", "Expose the public result"],
        ],
    },
    {
        "id": "linear_iv",
        "label": "Linear IV-HDFE",
        "description": "Identify linear causal/structural effects with excluded instruments while partialling out high-dimensional confounders.",
        "nodes": ["interface", "frontend", "linear_iv", "iv", "hdfe", "compute", "contracts"],
        "steps": [
            ["interface", "frontend", "Validate outcome/exogenous/endogenous/instrument roles"],
            ["frontend", "linear_iv", "Enter linear-IV model orchestration"],
            ["linear_iv", "iv", "Build instrument design and weighted IV solve"],
            ["linear_iv", "hdfe", "Concentrate fixed effects and compute absorbed DoF"],
            ["iv", "compute", "Reuse weighted WLS/linear algebra primitives"],
            ["compute", "contracts", "Assemble estimates and reportable IV diagnostics"],
            ["contracts", "interface", "Expose the public result"],
        ],
    },
    {
        "id": "ppml",
        "label": "PPML-HDFE",
        "description": "Estimate multiplicative conditional-mean effects for nonnegative outcomes while absorbing high-dimensional heterogeneity and handling separation.",
        "nodes": ["interface", "frontend", "ppml", "model_shared", "hdfe", "compute", "contracts"],
        "steps": [
            ["interface", "frontend", "Validate Poisson outcome, exposure and FE inputs"],
            ["frontend", "ppml", "Initialize PPML state/separation contract"],
            ["ppml", "model_shared", "Use shared Poisson numerical primitives"],
            ["ppml", "hdfe", "Weighted FE concentration inside IRLS"],
            ["hdfe", "compute", "Run projection/WLS/VCE kernels"],
            ["compute", "contracts", "Assemble PPML result and publication output"],
            ["contracts", "interface", "Expose the public result"],
        ],
    },
    {
        "id": "ivppml",
        "label": "IV-PPML-HDFE",
        "description": "Identify endogenous effects in multiplicative/PPML models using excluded instruments and high-dimensional fixed effects.",
        "nodes": ["interface", "frontend", "ppml_iv", "ppml", "model_shared", "hdfe", "iv", "compute", "resampling", "contracts"],
        "steps": [
            ["interface", "frontend", "Validate nonlinear IV specification"],
            ["frontend", "ppml_iv", "Enter IV-PPML orchestration"],
            ["ppml_iv", "ppml", "Reuse PPML separation/config primitives"],
            ["ppml_iv", "hdfe", "Concentrate weighted high-dimensional FE"],
            ["ppml_iv", "iv", "Evaluate additive IV moments / weighted 2SLS step"],
            ["iv", "compute", "Reuse WLS and covariance kernels"],
            ["ppml_iv", "resampling", "Optional model-specific bootstrap delegates draw mechanics"],
            ["compute", "contracts", "Assemble IV-PPML/SPJ inference output"],
            ["contracts", "interface", "Expose the public result"],
        ],
    },
    {
        "id": "heterogeneous_spec",
        "label": "Heterogeneous specifications",
        "description": "Accelerate event-study, group-specific-slope and interaction-rich specifications without changing the requested heterogeneous coefficients.",
        "nodes": ["interface", "ols", "linear_iv", "ppml", "ppml_iv", "hdfe", "compute", "contracts"],
        "steps": [
            ["interface", "compute", "Compile explicit coefficient heterogeneity before dense expansion"],
            ["compute", "hdfe", "Certify FE row topology and retain only locally active coefficient support"],
            ["hdfe", "compute", "Project nuisance FE within certified components"],
            ["compute", "ols", "Feed compact design to OLS when economically equivalent"],
            ["compute", "linear_iv", "Feed compact role designs to linear IV when economically equivalent"],
            ["compute", "ppml", "Feed compact design to PPML while retaining global convergence/separation semantics"],
            ["compute", "ppml_iv", "Feed compact exogenous/endogenous/instrument designs to IV-PPML"],
            ["contracts", "interface", "Fallback to pooled dense execution whenever exactness or benefit is not certified"],
        ],
    },
    {
        "id": "execution_planning",
        "label": "Execution planning",
        "description": "Choose a feasible physical execution for a fixed econometric specification without allowing performance heuristics to alter identification or the estimand.",
        "nodes": ["interface", "data", "planner", "hdfe", "compute", "contracts"],
        "steps": [
            ["interface", "data", "Compile required raw columns and reusable data roles"],
            ["data", "planner", "Report projected payload and memory requirements"],
            ["hdfe", "planner", "Report exact FE topology and parallel work geometry"],
            ["compute", "planner", "Report certified physical representations and workspace candidates"],
            ["planner", "compute", "Choose memory/representation/thread policy among exact candidates"],
            ["compute", "contracts", "Expose explainable execution metadata without changing statistical output"],
        ],
    },
    {
        "id": "session",
        "label": "Repeated specifications",
        "description": "Accelerate common empirical robustness workflows that repeatedly change outcomes, controls or FE sets across regression-table columns.",
        "nodes": ["interface", "ols", "linear_iv", "hdfe", "compute", "contracts"],
        "steps": [
            ["interface", "hdfe", "Cache encoded FE combinations and reusable absorber state"],
            ["interface", "ols", "Dispatch repeated OLS specifications when requested"],
            ["interface", "linear_iv", "Dispatch repeated linear-IV specifications when requested"],
            ["hdfe", "compute", "Reuse within-transformed columns and execution context"],
            ["compute", "contracts", "Assemble each specification result with shared-state metadata"],
            ["contracts", "interface", "Return the repeated-spec result"],
        ],
    },
]

INVARIANTS = [
    "compute is estimator-agnostic and must not import HDFE, IV, or model implementations.",
    "hdfe may use compute, but numerical FE canonicalization/core reduction must not rewrite the requested inference topology.",
    "generic iv remains outcome-model agnostic; linear-IV and IV-PPML model layers consume it rather than importing one another.",
    "resampling owns draw mechanics only; estimator/inference-specific statistics remain outside resampling.",
    "standard CRV1 covariance remains in compute/VCE; advanced cluster diagnostics and WCR/WCU live in inference/cluster.",
    "public boundaries expose structured EconHDFEError codes without masking programmer failures.",
    "pyreghdfe is a compatibility surface; canonical implementation lives under econhdfe.",
]

CODEMAP = [
    ["Choose the estimator implied by the empirical specification", "econhdfe/api.py; econhdfe/__init__.py"],
    ["Validate economic variable roles before estimation", "econhdfe/frontend/"],
    ["Compile factor interactions / event-study / group-specific slopes", "econhdfe/design.py; factorvars.py; compute/design_plan.py"],
    ["Reuse work across regression-table robustness specifications", "econhdfe/sessions.py"],
    ["Partial out high-dimensional observed heterogeneity", "econhdfe/hdfe/absorber.py; plan.py; numerical_core.py; projection.py"],
    ["Count absorbed nuisance parameters / residual degrees of freedom", "econhdfe/hdfe/rank.py; dof.py"],
    ["Represent exclusion restrictions and IV orthogonality conditions", "econhdfe/iv/"],
    ["Estimate endogenous linear effects and diagnose weak instruments", "econhdfe/models/linear_iv/"],
    ["Estimate multiplicative effects for nonnegative outcomes / trade flows", "econhdfe/models/ppml/"],
    ["Estimate endogenous multiplicative effects and panel-bias corrections", "econhdfe/models/ppml_iv/"],
    ["Construct robust/clustered uncertainty and weighted least-squares sufficient statistics", "econhdfe/compute/"],
    ["Run bootstrap/resampling uncertainty calculations", "econhdfe/resampling/"],
    ["Diagnose cluster quality and few-cluster sensitivity", "econhdfe/inference/cluster/"],
    ["Return stable paper-facing estimates, diagnostics and structured failures", "econhdfe/errors.py; config.py; results.py; reporting.py; support_reports.py"],
]


@dataclass(frozen=True)
class ModuleRecord:
    module: str
    path: str
    group: str
    is_init: bool


def _version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("project version not found")
    return m.group(1)


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _group(module: str) -> str:
    if module == "pyreghdfe" or module.startswith("pyreghdfe."):
        return "compat"
    if module in {"econhdfe.errors", "econhdfe.config", "econhdfe.results", "econhdfe.reporting", "econhdfe.support_reports"}:
        return "contracts"
    if module.startswith("econhdfe.frontend"):
        return "frontend"
    if module.startswith("econhdfe.data"):
        return "data"
    if module.startswith("econhdfe.planner"):
        return "planner"
    if module.startswith("econhdfe.compute"):
        return "compute"
    if module.startswith("econhdfe.hdfe") or module.startswith("econhdfe.effects"):
        return "hdfe"
    if module.startswith("econhdfe.iv"):
        return "iv"
    if module.startswith("econhdfe.resampling"):
        return "resampling"
    if module.startswith("econhdfe.inference.cluster"):
        return "cluster_inference"
    if module.startswith("econhdfe.models.linear_iv"):
        return "linear_iv"
    if module.startswith("econhdfe.models.ppml_iv"):
        return "ppml_iv"
    if module.startswith("econhdfe.models.ppml"):
        return "ppml"
    if module in {"econhdfe.models.ols", "econhdfe.models.ols_bootstrap"}:
        return "ols"
    if module.startswith("econhdfe.models"):
        return "model_shared"
    return "interface"


def _records(root: Path) -> list[ModuleRecord]:
    out: list[ModuleRecord] = []
    for package in ("econhdfe", "pyreghdfe"):
        base = root / package
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            mod = _module_name(root, path)
            out.append(
                ModuleRecord(
                    module=mod,
                    path=path.relative_to(root).as_posix(),
                    group=_group(mod),
                    is_init=path.name == "__init__.py",
                )
            )
    return out


def _resolve_import(record: ModuleRecord, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = record.module.split(".") if record.is_init else record.module.split(".")[:-1]
    ascend = node.level - 1
    if ascend > len(package_parts):
        return ""
    base = package_parts[: len(package_parts) - ascend]
    if node.module:
        base += node.module.split(".")
    return ".".join(base)


def _import_edges(root: Path, records: list[ModuleRecord]) -> tuple[list[dict], list[dict]]:
    by_module = {r.module: r for r in records}
    raw: list[dict] = []
    for record in records:
        path = root / record.path
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_import(record, node)
                if target:
                    targets.append(target)
            for target in targets:
                if not (target == "econhdfe" or target.startswith("econhdfe.") or target == "pyreghdfe" or target.startswith("pyreghdfe.")):
                    continue
                # Map imports of symbols/subpaths to the deepest known module prefix.
                resolved = target
                while resolved and resolved not in by_module:
                    if "." not in resolved:
                        break
                    resolved = resolved.rsplit(".", 1)[0]
                target_group = _group(resolved or target)
                raw.append(
                    {
                        "source_module": record.module,
                        "source_path": record.path,
                        "source_group": record.group,
                        "target_module": target,
                        "target_group": target_group,
                        "reexport_surface": record.is_init,
                    }
                )

    counts: Counter[tuple[str, str]] = Counter()
    examples: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for edge in raw:
        if edge["source_group"] == edge["target_group"] or edge["reexport_surface"]:
            continue
        key = (edge["source_group"], edge["target_group"])
        counts[key] += 1
        example = f'{edge["source_module"]} → {edge["target_module"]}'
        if len(examples[key]) < 5 and example not in examples[key]:
            examples[key].append(example)

    aggregated = [
        {
            "source": source,
            "target": target,
            "count": count,
            "examples": examples[(source, target)],
        }
        for (source, target), count in sorted(counts.items())
    ]
    return raw, aggregated


def _public_exports(root: Path) -> list[str]:
    tree = ast.parse((root / "econhdfe" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                values = ast.literal_eval(node.value)
            except Exception:
                return []
            if isinstance(values, list):
                return [str(v) for v in values]
    return []


def _repo_areas(root: Path) -> list[dict]:
    areas = []
    for name in ["econhdfe", "pyreghdfe", "tests", "validation", "benchmarks", "docs", "scripts", "skills", "compatibility"]:
        base = root / name
        if not base.exists():
            continue
        files = []
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if "docs/development/architecture-map" in rel.as_posix():
                continue
            if any(part in {"__pycache__", ".pytest_cache", "build", "dist", ".venv", "venv"} or part.endswith(".egg-info") for part in rel.parts):
                continue
            if p.suffix in {".pyc", ".pyo", ".nbc", ".nbi", ".aux", ".log", ".out"} or p.name == ".coverage":
                continue
            files.append(p)
        areas.append(
            {
                "name": name,
                "files": len(files),
                "python_files": sum(p.suffix == ".py" for p in files),
                "markdown_files": sum(p.suffix.lower() == ".md" for p in files),
            }
        )
    return areas


def build_ir(root: Path) -> dict:
    records = _records(root)
    raw_edges, aggregate_edges = _import_edges(root, records)
    files_by_group: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        files_by_group[record.group].append(record.path)
    nodes = []
    for group_id, spec in GROUPS.items():
        x, y, width = LAYOUT[group_id]
        nodes.append(
            {
                "id": group_id,
                **spec,
                "x": x,
                "y": y,
                "width": width,
                "height": NODE_HEIGHT,
                "label_lines": _wrap_label(spec["label"]),
                "files": files_by_group[group_id],
                "module_count": len(files_by_group[group_id]),
            }
        )
    return {
        "schema_version": 1,
        "project": "econhdfe",
        "version": _version(root),
        "evidence": {
            "topology": "Python AST import analysis; package __init__ re-export imports excluded from aggregated dependency counts.",
            "execution_flows": "Explicit semantic overlays reviewed against estimator orchestration modules; not presented as import edges.",
        },
        "nodes": nodes,
        "aggregate_import_edges": aggregate_edges,
        "raw_import_edges": raw_edges,
        "overview_edges": [{"source": source, "target": target} for source, target in OVERVIEW_EDGES],
        "flows": FLOWS,
        "public_exports": _public_exports(root),
        "repo_areas": _repo_areas(root),
        "invariants": INVARIANTS,
        "codemap": [{"task": task, "location": location} for task, location in CODEMAP],
    }


def _mermaid_id(group_id: str) -> str:
    return "n_" + re.sub(r"[^a-zA-Z0-9_]", "_", group_id)


def render_markdown(ir: dict) -> str:
    node_by_id = {n["id"]: n for n in ir["nodes"]}
    lines = [
        f'# econhdfe architecture map — {ir["version"]}',
        "",
        "> Generated by `skills/econhdfe/scripts/architecture_map.py`. Do not hand-edit generated topology; update the generator or source code and regenerate.",
        "",
        "This document is the compact, evidence-backed code map for contributor onboarding. The import graph below is recovered from Python ASTs. The execution flows are semantic overlays verified against the estimator orchestration modules; they are intentionally separated from import dependency claims.",
        "",
        "## Bird's-eye view",
        "",
        "```mermaid",
        "flowchart TB",
    ]
    for node in ir["nodes"]:
        label = node["label"].replace('"', "'")
        lines.append(f'  {_mermaid_id(node["id"])}["{label}"]')
    for edge in ir["aggregate_import_edges"]:
        lines.append(
            f'  {_mermaid_id(edge["source"])} -->|"{edge["count"]} imports"| {_mermaid_id(edge["target"])}'
        )
    lines.extend(["```", "", "The diagram collapses files into stable architectural groups. `__init__.py` re-export imports are excluded from the edge counts so the graph reflects implementation dependencies rather than public namespace wiring.", ""])

    lines.extend(["## Codemap", "", "| If you need to change… | Start here |", "| --- | --- |"]) 
    for item in ir["codemap"]:
        lines.append(f'| {item["task"]} | `{item["location"].replace("; ", "` / `")}` |')
    lines.append("")

    lines.extend(["## Architectural groups", ""])
    for node in ir["nodes"]:
        preview = ", ".join(node["files"][:5])
        if len(node["files"]) > 5:
            preview += f", … (+{len(node['files']) - 5})"
        lines.extend(
            [
                f'### {node["label"]}',
                "",
                node["description"],
                "",
                f'Files/modules: {node["module_count"]}. {preview or "No Python modules found."}',
                "",
            ]
        )

    lines.extend(["## Observed cross-group imports", "", "| Source | Target | AST imports | Examples |", "| --- | --- | ---: | --- |"]) 
    for edge in ir["aggregate_import_edges"]:
        examples = "<br>".join(f'`{e}`' for e in edge["examples"])
        lines.append(
            f'| {node_by_id[edge["source"]]["label"]} | {node_by_id[edge["target"]]["label"]} | {edge["count"]} | {examples} |'
        )
    lines.append("")

    lines.extend(["## Representative execution flows", ""])
    for flow in ir["flows"]:
        lines.extend([f'### {flow["label"]}', "", flow["description"], "", "```mermaid", "flowchart LR"])
        seen = set()
        for source, target, label in flow["steps"]:
            for group_id in (source, target):
                if group_id not in seen:
                    seen.add(group_id)
                    lines.append(f'  {_mermaid_id(group_id)}["{node_by_id[group_id]["label"]}"]')
            lines.append(f'  {_mermaid_id(source)} -->|"{label.replace(chr(34), chr(39))}"| {_mermaid_id(target)}')
        lines.extend(["```", ""])

    lines.extend(["## Architectural invariants", ""])
    lines.extend(f"- {item}" for item in ir["invariants"])
    lines.append("")

    lines.extend(["## Repository map", "", "| Area | Files | Python | Markdown |", "| --- | ---: | ---: | ---: |"]) 
    for area in ir["repo_areas"]:
        lines.append(f'| `{area["name"]}/` | {area["files"]} | {area["python_files"]} | {area["markdown_files"]} |')
    lines.extend(
        [
            "",
            "## Visual companion",
            "",
            "Open `architecture.html` for an interactive version of the same evidence: select an estimator flow to highlight participating layers, or click a node to inspect its files and AST-derived incoming/outgoing dependencies.",
            "",
            "Machine-readable evidence is in `architecture.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(ir: dict) -> str:
    payload = json.dumps(ir, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    title = f'econhdfe {html.escape(ir["version"])} architecture'
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#0b1017;--panel:#111821;--panel2:#151e29;--card:#172231;--card-hi:#1c2a3b;--text:#edf3f8;--muted:#94a3b8;--line:#74859a;--border:#29384a;--surface:#6aa7ff;--model:#c997ff;--infra:#68d391;--kernel:#f0c86a;--compat:#ff8a84;--active:#8bc3ff;--shadow:rgba(0,0,0,.32)}}
:root.light{{--bg:#f5f7fa;--panel:#fff;--panel2:#f8fafc;--card:#fff;--card-hi:#f8fafc;--text:#152033;--muted:#64748b;--line:#8997aa;--border:#d2dae5;--surface:#0969da;--model:#8250df;--infra:#1a7f37;--kernel:#9a6700;--compat:#cf222e;--active:#0550ae;--shadow:rgba(15,23,42,.10)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}button{{font:inherit}}code{{overflow-wrap:anywhere}}.shell{{max-width:1580px;margin:auto;padding:24px}}.top{{display:flex;gap:18px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap}}h1{{font-size:24px;line-height:1.2;margin:0 0 8px;letter-spacing:-.02em}}.sub{{color:var(--muted);max-width:920px}}.theme{{border:1px solid var(--border);background:var(--panel);color:var(--text);border-radius:9px;padding:9px 12px;cursor:pointer;min-height:40px}}.toolbar{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 14px}}.toolbar button{{border:1px solid var(--border);background:var(--panel);color:var(--text);border-radius:999px;padding:8px 13px;cursor:pointer;min-height:38px}}.toolbar button:hover{{background:var(--panel2)}}.toolbar button[aria-pressed="true"]{{background:var(--panel2);border-color:var(--active);box-shadow:inset 0 0 0 1px var(--active)}}.layout{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px;align-items:start}}.stage{{background:var(--panel);border:1px solid var(--border);border-radius:14px;overflow:auto;min-width:0;box-shadow:0 14px 36px var(--shadow)}}svg{{display:block;width:100%;min-width:1180px;height:auto}}.lane{{fill:var(--panel2);stroke:var(--border);stroke-width:1}}.lane-label{{fill:var(--muted);font-size:11px;font-weight:700;letter-spacing:.12em}}.edge-halo{{fill:none;stroke:var(--panel);stroke-width:5.8;stroke-linecap:round;stroke-linejoin:round;opacity:.90;pointer-events:none;vector-effect:non-scaling-stroke}}.edge{{fill:none;stroke:var(--line);stroke-width:1.35;stroke-linecap:round;stroke-linejoin:round;opacity:.44;vector-effect:non-scaling-stroke;transition:opacity .15s,stroke-width .15s,stroke .15s}}.edge.overview{{stroke-width:1.5;opacity:.60}}.edge.all{{stroke-width:1.0;opacity:.20}}.edge.on{{stroke:var(--active);stroke-width:2.4;opacity:.98}}.edge-halo.on{{stroke-width:6.8;opacity:.96}}.node{{cursor:pointer;opacity:.78;transition:opacity .15s}}.node:hover,.node:focus,.node.on{{opacity:1}}.node .card{{fill:var(--card);stroke:var(--border);stroke-width:1.4;filter:url(#nodeShadow)}}.node:hover .card,.node:focus .card{{fill:var(--card-hi)}}.node.on .card{{stroke:var(--active);stroke-width:2.3}}.node .rolebar{{stroke:none}}.node .title{{fill:var(--text);font-size:13.5px;font-weight:650;letter-spacing:-.01em;pointer-events:none}}.node .small{{fill:var(--muted);font-size:10.5px;pointer-events:none}}.node[data-role="surface"] .rolebar{{fill:var(--surface)}}.node[data-role="model"] .rolebar{{fill:var(--model)}}.node[data-role="infra"] .rolebar{{fill:var(--infra)}}.node[data-role="kernel"] .rolebar{{fill:var(--kernel)}}.node[data-role="compat"] .rolebar{{fill:var(--compat)}}.side{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:17px;min-width:0;box-shadow:0 10px 28px var(--shadow)}}.side h2{{font-size:17px;line-height:1.25;margin:0 0 7px}}.meta{{color:var(--muted);font-size:12px}}.filelist,.edgelist{{padding-left:18px;max-height:210px;overflow:auto;overflow-wrap:anywhere}}.filelist li,.edgelist li{{margin:3px 0}}.flowdesc{{margin:13px 0;padding:11px 12px;background:var(--panel2);border:1px solid var(--border);border-radius:10px}}.steps{{padding-left:20px}}.steps li{{margin:5px 0}}.legend{{display:flex;gap:13px;flex-wrap:wrap;margin:11px 0 4px;color:var(--muted);font-size:12px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}.repo{{margin-top:12px;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}.repo div{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px}}.repo b{{display:block;margin-bottom:4px}}.note{{margin-top:14px;color:var(--muted);font-size:12px}}@media(max-width:980px){{.layout{{grid-template-columns:1fr}}.side{{order:-1}}.shell{{padding:14px}}}}@media(prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style>
</head>
<body>
<main class="shell" id="app">
<div class="top"><div><h1>{title}</h1><div class="sub">AST-derived internal import topology with code-verified execution-flow overlays. Re-export-only <code>__init__.py</code> imports are excluded from dependency counts.</div></div><button class="theme" id="theme" type="button">Toggle theme</button></div>
<div class="legend"><span><i class="dot" style="background:var(--surface)"></i>surface</span><span><i class="dot" style="background:var(--model)"></i>model</span><span><i class="dot" style="background:var(--infra)"></i>infrastructure</span><span><i class="dot" style="background:var(--kernel)"></i>kernel</span><span><i class="dot" style="background:var(--compat)"></i>compatibility</span></div>
<div class="toolbar" id="flows"></div>
<div class="layout"><section class="stage" aria-label="Architecture graph"><svg id="graph" viewBox="0 0 1240 835" role="img" aria-label="econhdfe architecture dependency graph"><defs><filter id="nodeShadow" x="-15%" y="-20%" width="130%" height="150%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-opacity=".16"/></filter><marker id="arrow" viewBox="0 0 10 10" markerWidth="8" markerHeight="8" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" fill="none" stroke="var(--line)" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round"/></marker><marker id="arrowOn" viewBox="0 0 10 10" markerWidth="9" markerHeight="9" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 1.2 1.2 L 8.8 5 L 1.2 8.8" fill="none" stroke="var(--active)" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"/></marker></defs><g id="lanes"></g><g id="edges"></g><g id="nodes"></g></svg></section><aside class="side" id="details" aria-live="polite"></aside></div>
<section><h2 style="font-size:16px;margin:20px 0 8px">Repository areas</h2><div class="repo" id="repo"></div></section>
<div class="note">Generated by <code>skills/econhdfe/scripts/architecture_map.py</code>. Companion Markdown and JSON are stored beside this file.</div>
</main>
<script>
const data={payload};
const laneLayer=document.getElementById('lanes'), edgeLayer=document.getElementById('edges'), nodeLayer=document.getElementById('nodes'), flowBar=document.getElementById('flows'), details=document.getElementById('details');
const nodeMap=new Map(data.nodes.map(n=>[n.id,n]));
let activeMode='overview', activeFlow=null, selected='interface';
const NS='http://www.w3.org/2000/svg';
function svgEl(tag,attrs={{}}){{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e}}
function center(n){{return [n.x+n.width/2,n.y+n.height/2]}}
function sideFor(a,b){{const [ax,ay]=center(a),[bx,by]=center(b),dx=bx-ax,dy=by-ay;if(Math.abs(dy)>105)return dy>0?['bottom','top']:['top','bottom'];return dx>=0?['right','left']:['left','right']}}
function portPoint(n,side,slot,count){{const pad=22,span=(side==='left'||side==='right'?n.height:n.width)-pad*2,t=count<=1?.5:(slot+1)/(count+1);if(side==='top'||side==='bottom'){{const x=n.x+pad+span*t;return [x,side==='top'?n.y:n.y+n.height]}}const y=n.y+pad+span*t;return [side==='left'?n.x:n.x+n.width,y]}}
function outsidePoint(p,side,gap){{if(side==='top')return [p[0],p[1]-gap];if(side==='bottom')return [p[0],p[1]+gap];if(side==='left')return [p[0]-gap,p[1]];return [p[0]+gap,p[1]]}}
function roundedOrthogonal(points,r=12){{const clean=[];for(const p of points){{const q=[+p[0].toFixed(2),+p[1].toFixed(2)];if(!clean.length||clean.at(-1)[0]!==q[0]||clean.at(-1)[1]!==q[1])clean.push(q)}}if(clean.length<2)return '';let d=`M ${{clean[0][0]}} ${{clean[0][1]}}`;for(let i=1;i<clean.length-1;i++){{const p0=clean[i-1],p=clean[i],p1=clean[i+1],v0=[p0[0]-p[0],p0[1]-p[1]],v1=[p1[0]-p[0],p1[1]-p[1]],l0=Math.hypot(...v0),l1=Math.hypot(...v1),rr=Math.min(r,l0/2,l1/2);if(!rr){{d+=` L ${{p[0]}} ${{p[1]}}`;continue}}const a=[p[0]+v0[0]/l0*rr,p[1]+v0[1]/l0*rr],b=[p[0]+v1[0]/l1*rr,p[1]+v1[1]/l1*rr];d+=` L ${{a[0].toFixed(2)}} ${{a[1].toFixed(2)}} Q ${{p[0]}} ${{p[1]}} ${{b[0].toFixed(2)}} ${{b[1].toFixed(2)}}`}}d+=` L ${{clean.at(-1)[0]}} ${{clean.at(-1)[1]}}`;return d}}
function blockedSameRow(a,b){{const leftFirst=a.x<b.x,lo=leftFirst?a.x+a.width:b.x+b.width,hi=leftFirst?b.x:a.x;if(hi<=lo)return false;return data.nodes.some(n=>n.id!==a.id&&n.id!==b.id&&Math.abs(n.y-a.y)<30&&n.x<hi&&n.x+n.width>lo)}}
function makePortPlan(edges){{const requests=[];for(const e of edges){{const a=nodeMap.get(e.source),b=nodeMap.get(e.target);if(!a||!b)continue;const [ss,ts]=sideFor(a,b);requests.push({{e,a,b,ss,ts,key:e.source+'>'+e.target}})}}const bins=new Map;for(const q of requests){{for(const [node,side,other,kind] of [[q.a,q.ss,q.b,'s'],[q.b,q.ts,q.a,'t']]){{const k=node.id+'|'+side;if(!bins.has(k))bins.set(k,[]);bins.get(k).push({{key:q.key,other:center(other),kind,side}})}}}}for(const arr of bins.values())arr.sort((u,v)=>((u.side==='top'||u.side==='bottom')?u.other[0]-v.other[0]:u.other[1]-v.other[1])||u.key.localeCompare(v.key));const plan=new Map;for(const q of requests){{const point=(node,side,kind)=>{{const arr=bins.get(node.id+'|'+side),idx=arr.findIndex(x=>x.key===q.key&&x.kind===kind);return portPoint(node,side,idx,arr.length)}};plan.set(q.key,{{...q,start:point(q.a,q.ss,'s'),end:outsidePoint(point(q.b,q.ts,'t'),q.ts,8)}})}}return plan}}
function route(q,index){{const [x1,y1]=q.start,[x2,y2]=q.end,a=q.a,b=q.b,[acx,acy]=center(a),[bcx,bcy]=center(b),dy=bcy-acy,dx=bcx-acx,spread=((index%7)-3)*4;if(Math.abs(dy)<=105){{const sign=dx>=0?1:-1;if(blockedSameRow(a,b)){{const channel=a.y<210?Math.min(a.y,b.y)-22:Math.max(a.y+a.height,b.y+b.height)+24;return roundedOrthogonal([[x1,y1],[x1+sign*16,y1],[x1+sign*16,channel+spread],[x2-sign*16,channel+spread],[x2-sign*16,y2],[x2,y2]],11)}}const midY=(y1+y2)/2;return roundedOrthogonal([[x1,y1],[x1+sign*14,y1],[x1+sign*14,midY],[x2-sign*14,midY],[x2-sign*14,y2],[x2,y2]],9)}}if(dy< -240){{const useRight=(acx+bcx)/2>=620,gutter=useRight?1220:20,startOut=outsidePoint([x1,y1],q.ss,16),endOut=outsidePoint([x2,y2],q.ts,18);return roundedOrthogonal([[x1,y1],startOut,[gutter,startOut[1]],[gutter,endOut[1]],endOut,[x2,y2]],15)}}const channel=y1+(y2-y1)*.5+spread;return roundedOrthogonal([[x1,y1],[x1,channel],[x2,channel],[x2,y2]],12)}}
function drawLanes(){{laneLayer.textContent='';const lanes=[['SURFACE',25,28,1190,165],['MODELS',25,225,1190,175],['INFRASTRUCTURE',25,445,1190,170],['KERNEL',25,645,1190,150]];for(const [label,x,y,w,h] of lanes){{laneLayer.appendChild(svgEl('rect',{{x,y,width:w,height:h,rx:15,class:'lane'}}));const t=svgEl('text',{{x:x+14,y:y+22,class:'lane-label'}});t.textContent=label;laneLayer.appendChild(t)}}}}
function flowEdges(){{if(!activeFlow)return new Set;return new Set(activeFlow.steps.map(s=>s[0]+'>'+s[1]))}}
function draw(){{drawLanes();edgeLayer.textContent='';nodeLayer.textContent='';const hotEdges=flowEdges();const edges=activeMode==='overview'?data.overview_edges:(activeMode==='all'?data.aggregate_import_edges:activeFlow.steps.map(s=>({{source:s[0],target:s[1]}})));const plan=makePortPlan(edges);edges.forEach((e,i)=>{{const q=plan.get(e.source+'>'+e.target);if(!q)return;const hot=!!activeFlow&&hotEdges.has(e.source+'>'+e.target),d=route(q,i),modeClass=activeMode==='all'?' all':(activeMode==='overview'?' overview':'');edgeLayer.appendChild(svgEl('path',{{d,class:'edge-halo'+(hot?' on':'')}}));edgeLayer.appendChild(svgEl('path',{{d,class:'edge'+modeClass+(hot?' on':''),'marker-end':`url(#${{hot?'arrowOn':'arrow'}})`}}))}});for(const n of data.nodes){{const active=!activeFlow||activeFlow.nodes.includes(n.id);const g=svgEl('g',{{class:'node'+(active?' on':''),tabindex:'0','data-id':n.id,'data-role':n.role,role:'button','aria-label':n.label}});g.appendChild(svgEl('rect',{{x:n.x,y:n.y,width:n.width,height:n.height,rx:13,class:'card'}}));g.appendChild(svgEl('rect',{{x:n.x,y:n.y,width:6,height:n.height,rx:3,class:'rolebar'}}));const lines=n.label_lines||[n.label];const titleY=n.y+(lines.length===1?34:26);const t=svgEl('text',{{x:n.x+18,y:titleY,class:'title'}});lines.forEach((line,i)=>{{const span=svgEl('tspan',{{x:n.x+18,dy:i===0?0:18}});span.textContent=line;t.appendChild(span)}});g.appendChild(t);const s=svgEl('text',{{x:n.x+18,y:n.y+n.height-17,class:'small'}});s.textContent=n.module_count+' Python module'+(n.module_count===1?'':'s');g.appendChild(s);g.addEventListener('click',()=>{{selected=n.id;renderDetails()}});g.addEventListener('keydown',ev=>{{if(ev.key==='Enter'||ev.key===' '){{ev.preventDefault();selected=n.id;renderDetails()}}}});nodeLayer.appendChild(g)}}}}
function renderDetails(){{const n=nodeMap.get(selected)||nodeMap.get('interface');const incoming=data.aggregate_import_edges.filter(e=>e.target===n.id), outgoing=data.aggregate_import_edges.filter(e=>e.source===n.id);const esc=s=>String(s).replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));let h=`<h2>${{esc(n.label)}}</h2><div class="meta">${{esc(n.layer)}} · ${{n.module_count}} Python modules</div><p>${{esc(n.description)}}</p>`;if(activeFlow){{h+=`<div class="flowdesc"><b>${{esc(activeFlow.label)}}</b><br>${{esc(activeFlow.description)}}</div><ol class="steps">`+activeFlow.steps.map(s=>`<li>${{esc(nodeMap.get(s[0])?.label||s[0])}} → ${{esc(nodeMap.get(s[1])?.label||s[1])}}: ${{esc(s[2])}}</li>`).join('')+'</ol>'}}h+='<b>Files</b><ul class="filelist">'+n.files.map(f=>`<li><code>${{esc(f)}}</code></li>`).join('')+'</ul>';h+='<b>AST-derived dependencies</b><ul class="edgelist">'+outgoing.map(e=>`<li>→ ${{esc(nodeMap.get(e.target)?.label||e.target)}} (${{e.count}})</li>`).concat(incoming.map(e=>`<li>← ${{esc(nodeMap.get(e.source)?.label||e.source)}} (${{e.count}})</li>`)).join('')+'</ul>';details.innerHTML=h}}
function setFlow(id){{activeMode=id;activeFlow=(id==='overview'||id==='all')?null:data.flows.find(f=>f.id===id);for(const b of flowBar.querySelectorAll('button'))b.setAttribute('aria-pressed',String(b.dataset.flow===id));draw();renderDetails()}}
function addFlowButton(id,label){{const b=document.createElement('button');b.type='button';b.dataset.flow=id;b.textContent=label;b.setAttribute('aria-pressed','false');b.addEventListener('click',()=>setFlow(id));flowBar.appendChild(b)}}
addFlowButton('overview','Overview');addFlowButton('all','All AST dependencies');for(const f of data.flows)addFlowButton(f.id,f.label);setFlow('overview');
document.getElementById('theme').addEventListener('click',()=>document.documentElement.classList.toggle('light'));
const repo=document.getElementById('repo');for(const a of data.repo_areas){{const d=document.createElement('div');d.innerHTML=`<b>${{a.name}}/</b><span>${{a.files}} files · ${{a.python_files}} Python · ${{a.markdown_files}} Markdown</span>`;repo.appendChild(d)}}
</script>
</body>
</html>
'''


def render_json(ir: dict) -> str:
    return json.dumps(ir, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def generated_files(root: Path) -> dict[str, str]:
    ir = build_ir(root)
    return {
        "architecture.json": render_json(ir),
        "architecture.md": render_markdown(ir),
        "architecture.html": render_html(ir),
    }


def write_or_check(root: Path, output_dir: Path, check: bool) -> int:
    outputs = generated_files(root)
    if check:
        stale = []
        for name, content in outputs.items():
            path = output_dir / name
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        if stale:
            print("architecture map is stale: " + ", ".join(stale), file=sys.stderr)
            print("run: python skills/econhdfe/scripts/architecture_map.py", file=sys.stderr)
            return 1
        print(f"architecture map PASS: {output_dir}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_text(content)
        print(output_dir / name)
    return 0


def main() -> None:
    default_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=default_root, help="econhdfe source-tree root")
    ap.add_argument("--output-dir", type=Path, help="output directory (default: docs/development/architecture-map)")
    ap.add_argument("--check", action="store_true", help="fail if committed generated outputs are stale")
    args = ap.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "docs" / "development" / "architecture-map").resolve()
    raise SystemExit(write_or_check(root, output, args.check))


if __name__ == "__main__":
    main()
