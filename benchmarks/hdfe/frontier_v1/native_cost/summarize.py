#!/usr/bin/env python3
"""Summarize observed native costs without adding nested timers or mixing units.

Timing medians use passed calls; every returned timing, outcome and error is also
retained. Same-named deltas may be summed across disjoint attempts of one kernel.
Different kernels, phases, constructor snapshots and wrapper clocks stay separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = [v for v in values if numeric(v)]
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def variants(values):
    unique = {json.dumps(v, sort_keys=True, allow_nan=False): v for v in values}
    return {"n": len(values), "consistent": len(unique) == 1,
            "values": [unique[k] for k in sorted(unique)]}


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}.{key}" if prefix else key)
    elif numeric(value):
        yield prefix, value


def native_observation(call):
    diagnostic = call.get("native_cost_diagnostics", {})
    if (diagnostic.get("available") is not True or diagnostic.get("enabled") is not True
            or diagnostic.get("schema_version") != 1 or diagnostic.get("scope") != "call"):
        return None
    counts, times, metadata, structure = {}, {}, {}, {}
    snapshots = {}
    for attempt in diagnostic.get("attempts", []):
        cost = attempt.get("native_cost", attempt)
        kernel = cost.get("kernel", diagnostic.get("kernel", "unknown"))
        for path, value in leaves(cost.get("delta")):
            target = times if path.endswith("seconds") else counts
            key = f"{kernel}.{path}"
            target[key] = target.get(key, 0) + value
        if isinstance(cost.get("after"), dict):
            snapshots[kernel] = cost["after"]
    for path, value in leaves(diagnostic.get("wrapper_delta")):
        (times if path.endswith("seconds") else counts)[f"wrapper.{path}"] = value
    structure["wrapper.route"] = diagnostic.get("route", diagnostic.get("kernel"))
    structure["kernel_builds_in_call"] = len(diagnostic["kernel_builds"]) if isinstance(diagnostic.get("kernel_builds"), list) else None
    structure["reported_iterations"] = call.get("solver_details", {}).get("iterations")
    structure["shared_iterations_per_attempt"] = [
        {"iterations": a["core_report"]["shared_iterations"], "rhs_width": a.get("rhs_width")}
        for a in diagnostic.get("attempts", []) if "shared_iterations" in a.get("core_report", {})]
    for kernel, snapshot in snapshots.items():
        if "state" in snapshot:
            for key, value in snapshot["state"].items():
                structure[f"{kernel}.{key}"] = value
            for key, value in snapshot.get("totals", {}).items():
                if key.startswith("constructor_") and numeric(value):
                    metadata[f"{kernel}.{key}"] = value
        else:
            for key in ("preconditioner_kind", "has_escalated", "n_obs", "n_dofs", "local_factors"):
                structure[f"{kernel}.{key}"] = snapshot.get(key)
            for key in ("core_constructor_seconds", "initial_preconditioner_seconds"):
                if numeric(snapshot.get(key)):
                    metadata[f"{kernel}.{key}"] = snapshot[key]
            for key, value in leaves(snapshot.get("schwarz_build")):
                metadata[f"{kernel}.cached_schwarz_build.{key}"] = value
            inventory = snapshot.get("local_factors", [])
            structure[f"{kernel}.local_factor_count"] = sum(g["count"] for g in inventory) if "local_factors" in snapshot else None
            dimensions = [g["shape"]["factor_solve_dim"] for g in inventory]
            structure[f"{kernel}.local_factor_dim_min"] = min(dimensions, default=None)
            structure[f"{kernel}.local_factor_dim_max"] = max(dimensions, default=None)
    for path, value in leaves(diagnostic.get("constructor", {})):
        metadata[f"wrapper.constructor.{path}"] = value
    fractions = {}
    build = times.get("fereg.joint_build_seconds")
    for label, denominator in (("joint_build_fraction_of_outer_total", call.get("total_s")),
                               ("joint_build_fraction_of_core_residualize", times.get("fereg.residualize_seconds"))):
        if numeric(build) and numeric(denominator) and denominator > 0:
            fractions[label] = build / denominator
    return {"counters": counts, "phase_times_s": times, "snapshot_metadata_s": metadata,
            "structure": structure, "fractions": fractions,
            "counter_units": diagnostic.get("counter_units")}


def aggregate_native(observations):
    observations = [o for o in observations if o is not None]
    if not observations:
        return None
    result = {"observations": len(observations)}
    for field in ("counters", "phase_times_s", "snapshot_metadata_s", "fractions"):
        keys = sorted({key for o in observations for key in o[field]})
        result[field] = {key: distribution([o[field].get(key) for o in observations]) for key in keys}
        if field == "counters":
            for value in result[field].values():
                value["consistent"] = value["n"] == len(observations) and value["min"] == value["max"]
    keys = sorted({key for o in observations for key in o["structure"]})
    result["structure"] = {key: variants([o["structure"].get(key) for o in observations]) for key in keys}
    result["counter_units"] = variants([o["counter_units"] for o in observations])
    result["counters_consistent"] = all(s["consistent"] for s in result["counters"].values())
    result["structure_consistent"] = all(s["consistent"] for s in result["structure"].values())
    return result


def ratio_groups(records, kind):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    excluded = {"diagnostics", "calls", "mode_order"}
    for record in records:
        req = record["request"]
        base = (req["family"], req["backend"], req["size"],
                req["diagnostics"] if kind == "cold_over_warm" else record["mode"])
        pairing = json.dumps({k: v for k, v in req.items() if k not in excluded}, sort_keys=True)
        label = record["mode"] if kind == "cold_over_warm" else req["diagnostics"]
        grouped[base][pairing][label].append(record)
    output = []
    numerator, denominator = ("cold", "warm") if kind == "cold_over_warm" else (True, False)
    for base, pairs in sorted(grouped.items()):
        values, a_values, b_values = [], [], []
        for pair in pairs.values():
            a, b = pair.get(numerator, []), pair.get(denominator, [])
            if len(a) == len(b) == 1 and a[0]["status"] == b[0]["status"] == "passed":
                x, y = a[0].get("total_s"), b[0].get("total_s")
                if numeric(x) and numeric(y) and y > 0:
                    values.append(x / y)
                    a_values.append(x)
                    b_values.append(y)
        output.append({"family": base[0], "backend": base[1], "size": base[2],
                       "diagnostics" if kind == "cold_over_warm" else "mode": base[3],
                       "kind": kind, "paired_total_time_ratio": distribution(values),
                       "ratio_of_paired_time_medians": statistics.median(a_values) / statistics.median(b_values) if values else None,
                       "excluded_or_missing_pairs": len(pairs) - len(values)})
    return output


def summarize(report, *, expected_replicates=3):
    records, worker_errors = [], []
    for case_index, case in enumerate(report.get("cases", [])):
        req = case["request"]
        calls = case.get("calls", [])
        present = {c.get("mode") for c in calls}
        calls = calls + [{"mode": mode, "status": case["status"], "not_returned": True}
                         for mode in req.get("calls", []) if mode not in present]
        if case["status"] != "passed":
            worker_errors.append({"case_index": case_index, "request": req, "status": case["status"],
                                  "error": case.get("error"), "has_partial_result": "partial_result" in case})
        for call in calls:
            record = {"case_index": case_index, "request": req, "case": case.get("case", {}),
                      **call, "native_observation": native_observation(call)}
            records.append(record)
    grouped = defaultdict(list)
    for r in records:
        q = r["request"]
        grouped[(q["family"], q["backend"], q["size"], q["diagnostics"], r["mode"])].append(r)
    rows, warnings = [], []
    for key, samples in sorted(grouped.items()):
        passed = [s for s in samples if s["status"] == "passed"]
        replicates = [s["request"].get("replicate", 0) for s in samples]
        metadata = [s["case"].get("metadata", {}) for s in samples]
        control = [{k: v for k, v in s["request"].items() if k not in {"replicate", "mode_order"}} for s in samples]
        complete = sorted(replicates) == list(range(expected_replicates))
        native = aggregate_native([s["native_observation"] for s in samples])
        errors = [s.get("accuracy", {}).get("max_forward_relative_error") for s in samples]
        row = {"family": key[0], "backend": key[1], "size": key[2], "diagnostics": key[3], "mode": key[4],
               "replicates": replicates, "replicates_complete": complete,
               "statuses": dict(Counter(s["status"] for s in samples)), "controls": variants(control),
               "N": variants([m.get("N") for m in metadata]), "G": variants([m.get("G") for m in metadata]),
               "U": variants([m.get("U") for m in metadata]),
               "total_s": distribution([s.get("total_s") for s in passed]),
               "setup_s": distribution([s.get("setup_s") for s in passed]),
               "solve_s": distribution([s.get("solve_s") for s in passed]),
               "all_returned_total_s": distribution([s.get("total_s") for s in samples]),
               "max_forward_error": max((v for v in errors if numeric(v)), default=None), "native": native,
               "sample_refs": [{"case_index": s["case_index"], "mode": s["mode"], "replicate": s["request"].get("replicate", 0)} for s in samples]}
        rows.append(row)
        if not complete:
            warnings.append(f"{key}: expected replicate IDs 0..{expected_replicates - 1}, observed {replicates}")
        if not row["controls"]["consistent"]:
            warnings.append(f"{key}: controls differ; pooled times are not a controlled comparison")
        if len(passed) != len(samples):
            warnings.append(f"{key}: non-passing outcomes retained: {row['statuses']}")
        if key[3] and (native or {}).get("observations", 0) != len(samples):
            warnings.append(f"{key}: some requested native diagnostics are unavailable")
        if native and not (native["counters_consistent"] and native["structure_consistent"]):
            warnings.append(f"{key}: native counts or structure differ; inspect ranges and variants")
    errors = [r.get("accuracy", {}).get("max_forward_relative_error") for r in records]
    result = {
        "schema_version": 1, "expected_replicates": expected_replicates,
        "repeat_consistency_assessment": "single_trial_not_assessable" if expected_replicates == 1 else "compare_counter_ranges_and_structural_variants",
        "counts": {"worker_cases": len(report.get("cases", [])),
                   "worker_statuses": dict(Counter(c["status"] for c in report.get("cases", []))),
                   "recorded_calls": sum(len(c.get("calls", [])) for c in report.get("cases", [])),
                   "represented_call_outcomes": len(records),
                   "call_statuses": dict(Counter(r["status"] for r in records))},
        "max_forward_error": max((v for v in errors if numeric(v)), default=None),
        "rows": rows, "cold_over_warm": ratio_groups(records, "cold_over_warm"),
        "diagnostics_on_over_off": ratio_groups(records, "diagnostics_on_over_off"),
        "worker_errors": worker_errors,
        "call_errors": [{"case_index": r["case_index"], "mode": r["mode"], "status": r["status"],
                         "error": r.get("error"), "native": r["native_observation"]}
                        for r in records if r["status"] != "passed"],
        "warnings": warnings,
        "interpretation": {
            "timing_population": "total/setup/solve stats use passed calls; all returned timings and every outcome remain separate",
            "deltas": "sum only the same field across disjoint attempts of the same kernel; fallback namespaces remain separate",
            "timers": "inclusive/nested and sometimes concurrent; phase times must not be added together",
            "snapshot_metadata": "constructor and cached-build times describe retained objects; repeated warm snapshots are not new work",
            "factor_entries": "allocated sparse factor values including padding, not exact nnz(L); matrix nnz is full assembled CSR",
            "local_factors": "within local-factor shapes and cross-tab sparsity are not a factor-nnz or flop measurement",
            "symbolic_flops": "symbolic work estimate, not instrumented floating-point operations",
            "ratios": "descriptive paired ratios at equal recorded controls; no significance claim or strong ranking",
            "toggle": "hot-loop counters/timers only; both modes retain metadata and snapshot generation",
        },
    }
    return result


def number(value):
    return "—" if value is None else f"{value:.6g}" if numeric(value) else str(value)


def spread(value):
    if not value or not value.get("n"):
        return "—"
    if value["min"] == value["max"]:
        return number(value["median"])
    return f"{number(value['median'])} [{number(value['min'])}, {number(value['max'])}]"


def state(row, key):
    value = (row.get("native") or {}).get("structure", {}).get(key, {})
    choices = value.get("values", [])
    if len(choices) == 1:
        return choices[0]
    if choices and all(numeric(x) for x in choices):
        return f"{number(min(choices))}–{number(max(choices))}"
    return f"varies ({len(choices)})" if choices else None


def metric(row, field, key):
    return spread((row.get("native") or {}).get(field, {}).get(key))


def markdown_tables(summary):
    lines = ["# Native cost study summary", "", "Times are seconds: median [min, max] across passed calls. Counts retain their native units.", ""]

    def table(title, headers, rows):
        lines.extend([f"## {title}", "", "| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"])
        lines.extend("| " + " | ".join(("—" if v is None else str(v)).replace("|", "/") for v in row) + " |" for row in rows)
        lines.append("")

    table("Full call timing and accuracy", ["Family", "Backend", "L", "Diag", "Call", "Statuses", "Total s", "Max forward"],
          [[r["family"], r["backend"], r["size"], "on" if r["diagnostics"] else "off", r["mode"],
            json.dumps(r["statuses"], sort_keys=True), spread(r["total_s"]), number(r["max_forward_error"])] for r in summary["rows"]])
    within = [r for r in summary["rows"] if r["diagnostics"] and r["backend"] in {"within", "native_auto"}]
    table("Within vector-call work (diagnostics on)", ["Backend", "L", "Call", "D apply", "D adjoint", "P apply", "P adjoint", "Deferred build s", "Escalated", "Local factors", "Max factor dim"],
          [[r["backend"], r["size"], r["mode"], *[metric(r, "counters", "within." + k + ".calls") for k in ("design_apply", "design_adjoint", "preconditioner_apply", "preconditioner_adjoint")],
            metric(r, "phase_times_s", "within.deferred_build.seconds"), state(r, "within.has_escalated"),
            state(r, "within.local_factor_count"), state(r, "within.local_factor_dim_max")] for r in within])
    fereg = [r for r in summary["rows"] if r["diagnostics"] and r["backend"] == "fereg"]
    table("Fereg batch work and factor state (diagnostics on)", ["L", "Call", "H batches", "P batches", "Exact P", "Joint allocated entries", "CSR nnz", "Build outcome", "Joint build / outer total"],
          [[r["size"], r["mode"], *[metric(r, "counters", "fereg." + k) for k in ("matvec_calls", "precond_calls", "exact_precond_calls")],
            state(r, "fereg.joint_factor_entries"), state(r, "fereg.joint_matrix_nnz"),
            state(r, "fereg.joint_last_build_outcome"), metric(r, "fractions", "joint_build_fraction_of_outer_total")] for r in fereg])
    phases = ("joint_build_seconds", "joint_aggregation_seconds", "joint_pattern_seconds", "joint_assembly_seconds", "joint_analyze_seconds", "joint_numeric_seconds", "joint_refactor_seconds", "joint_refresh_seconds")
    table("Fereg inclusive phase times; columns must not be added", ["L", "Call", "Joint build", "Aggregation", "Pattern", "Assembly", "Analyze incl. ordering", "Numeric", "Refactor", "Refresh"],
          [[r["size"], r["mode"], *[metric(r, "phase_times_s", "fereg." + k) for k in phases]] for r in fereg])
    for key, title, other in (("cold_over_warm", "Cold total / warm total, paired by worker", "diagnostics"),
                              ("diagnostics_on_over_off", "Diagnostics on / off total, paired by replicate", "mode")):
        table(title, ["Backend", "L", other, "Paired ratio", "Excluded pairs"],
              [[r["backend"], r["size"], r[other], spread(r["paired_total_time_ratio"]), r["excluded_or_missing_pairs"]] for r in summary[key]])
    lines.extend(["## Scope and repeat checks", "", f"- Worker/call counts: `{json.dumps(summary['counts'], sort_keys=True)}`.",
                  "- Factor entries are allocated values, including padding; symbolic flops are estimates.",
                  "- Snapshot constructor/cached-build times are metadata, not new costs on each warm call.",
                  "- Ratios describe these trials; diagnostics on/off does not measure the entire patch overhead."])
    lines.extend("- " + warning for warning in summary["warnings"])
    if summary["expected_replicates"] == 1:
        lines.append("- One exploratory trial per configuration: repeat consistency and timing variability cannot be assessed.")
    elif not summary["warnings"]:
        lines.append("- Expected repeat coverage, controls, native counter values and structural snapshots agree across observed repeats.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-replicates", type=int, default=3)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--print-tables", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_replicates < 1:
        parser.error("expected replicates must be positive")
    stored = args.input.read_bytes()
    raw = gzip.decompress(stored) if args.input.suffix == ".gz" else stored
    report = json.loads(raw)
    summary = summarize(report, expected_replicates=args.expected_replicates)
    summary["source"] = {"path": args.input.name, "sha256": hashlib.sha256(raw).hexdigest(),
                         "stored_sha256": hashlib.sha256(stored).hexdigest(),
                         "compression": "gzip" if args.input.suffix == ".gz" else None,
                         "study_source_sha256": report.get("study_source_sha256")}
    summary["markdown_tables"] = markdown_tables(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(summary["markdown_tables"])
    print(json.dumps({"counts": summary["counts"], "max_forward_error": summary["max_forward_error"],
                      "rows": len(summary["rows"]), "warnings": len(summary["warnings"]), "source_sha256": summary["source"]["sha256"]}, indent=2))
    if args.print_tables:
        print(summary["markdown_tables"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
