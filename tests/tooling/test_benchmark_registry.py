from pathlib import Path
import json


def test_real_world_20260912_benchmark_is_archived_immutably():
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "real_world" / "2026-09-12"
    expected = {
        "benchmark_complete_table_20260912.md",
        "mobility_ppml_speedup_diagnosis_20260912.md",
        "provenance.json",
        "README.md",
    }
    assert expected <= {p.name for p in root.iterdir()}
    prov = json.loads((root / "provenance.json").read_text())
    assert prov["status"]["ppml"].startswith("pre-0.4.5 baseline")
    assert len(prov["source_files"]) == 2
    assert all(len(v) == 64 for v in prov["source_files"].values())
