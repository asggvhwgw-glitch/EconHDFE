#!/usr/bin/env python3
"""Render the recorded native-cost study with non-overlapping cost categories."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def load_report(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def render(source, output, budget_probe=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = load_report(source)
    rows = []
    if not report.get("cases") or any(case["status"] != "passed" for case in report["cases"]):
        raise ValueError("plot requires a completed study with every requested worker passing")
    for case in report["cases"]:
        if not case["request"]["diagnostics"]:
            continue
        for call in case.get("calls", []):
            if call["status"] != "passed":
                raise ValueError("plot requires complete passing evidence; inspect failed records first")
            rows.append((case["request"]["backend"], case["request"]["size"],
                         case["case"]["n"], call))
    sizes = sorted({size for _, size, _, _ in rows})
    colors = {"within": "#087F8C", "native_auto": "#6851A5", "fereg": "#D16B28"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for backend, color in colors.items():
        for mode, style in (("cold", "-"), ("warm", "--")):
            x, low, mid, high = [], [], [], []
            for size in sizes:
                sample = [(n, c["total_s"]) for b, s, n, c in rows
                          if b == backend and s == size and c["mode"] == mode]
                if not sample:
                    continue
                x.append(sample[0][0]); times = [t for _, t in sample]
                low.append(min(times)); mid.append(median(times)); high.append(max(times))
            axes[0].plot(x, mid, style, marker="o", markersize=4, color=color,
                         label=f"{backend} / {mode}")
            axes[0].fill_between(x, low, high, color=color, alpha=.08)
    axes[0].set(xscale="log", yscale="log", xlabel="Observations N", ylabel="Full projection time (s)",
                title="Cold construction and factor reuse")
    axes[0].grid(alpha=.15, which="both")
    axes[0].legend(fontsize=8, ncols=1)

    parts, entries, ns = [], [], []
    for size in sizes:
        sample = sorted([(n, c) for b, s, n, c in rows
                         if b == "fereg" and s == size and c["mode"] == "cold"],
                        key=lambda row: row[1]["total_s"])
        if not sample:
            continue
        # One actual median-time trial: component medians would not be additive.
        n, call = sample[len(sample) // 2]
        attempts = call["native_cost_diagnostics"]["attempts"]
        totals = {key: sum(a["native_cost"]["delta"][key] for a in attempts)
                  for key in ("joint_build_seconds", "joint_analyze_seconds", "joint_numeric_seconds")}
        build, analyze, numeric = (totals[key] for key in totals)
        segments = [call["total_s"] - build, build - analyze - numeric, analyze, numeric]
        if min(segments) < -1e-8:
            raise ValueError("nested time accounting is inconsistent; do not silently clamp")
        parts.append(np.maximum(segments, 0) / call["total_s"])
        state = call["native_cost_diagnostics"]["core_after"]["state"]
        entries.append(state["joint_factor_entries"])
        ns.append(n)
    x = np.arange(len(parts))
    bottom = np.zeros(len(parts))
    labels = ("Other outer-call work", "Other joint-build work", "Ordering + symbolic", "Numeric factorization")
    for i, (label, color) in enumerate(zip(labels, ("#BCC8D0", "#71B9BD", "#BD9BDA", "#D16B28"))):
        values = np.array(parts)[:, i]
        axes[1].bar(x, values * 100, bottom=bottom, label=label, color=color, width=.66)
        bottom += values * 100
    axes[1].set(xticks=x, xticklabels=[f"{n:,}" for n in ns], ylim=(0, 100),
                xlabel="Observations N", ylabel="Share of full cold-call wall time (%)",
                title="Fereg: median-time trial at each size")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(fontsize=8, loc="upper left")

    pairs = [(n, count) for n, count in zip(ns, entries) if count is not None]
    axes[2].plot([n for n, _ in pairs], [count / 1e6 for _, count in pairs],
                 "o-", color=colors["fereg"])
    for n, count in pairs:
        axes[2].annotate(f"{count / 1e6:.2f}M", (n, count / 1e6),
                         xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    if budget_probe is not None:
        probe = load_report(budget_probe)
        for case in probe["cases"]:
            if case["status"] != "passed":
                raise ValueError("budget probe did not pass")
            cold = next(call for call in case["calls"] if call["mode"] == "cold")
            state = cold["native_cost_diagnostics"]["core_after"]["state"]
            planned = state["joint_analyzed_factor_entries"]
            if state["joint_factor_present"] or planned is None:
                raise ValueError("this plot annotation requires a rejected analyzed candidate")
            n = case["case"]["n"]
            axes[2].plot([n], [planned / 1e6], "x", color="#A52D34", markersize=8,
                         label="Rejected analyzed candidate")
            axes[2].annotate(f"{planned / 1e6:.2f}M: rejected", (n, planned / 1e6),
                             xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=9)
        axes[2].axhline(100, color="#686868", linestyle="--", linewidth=1,
                       label="Pinned 100M-value fill limit")
        axes[2].legend(fontsize=8, loc="upper left")
    axes[2].set(xscale="log", xlabel="Observations N", ylabel="Factor value entries (millions)",
                title="Joint factor storage and fill budget")
    axes[2].grid(alpha=.15)
    axes[2].margins(y=.18)
    fig.suptitle("FE frontier: native cost diagnostics | rigidity grid | two RHS, two native threads", fontsize=14)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path, facecolor="white")
    # Matplotlib emits optional spaces at path-data line endings.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-probe", type=Path)
    args = parser.parse_args()
    render(args.source, args.output, args.budget_probe)
