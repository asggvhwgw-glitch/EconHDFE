from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from econhdfe import PPMLConfig
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.separation import detect_separation

BASE = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/separation_datasets/{:02d}.csv"


def download_suite(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(1, 18):
        path = directory / f"{i:02d}.csv"
        if not path.exists():
            urllib.request.urlretrieve(BASE.format(i), path)


def validate(directory: Path) -> int:
    failures = []
    cfg = PPMLConfig(engine="replica", separation=("fe", "simplex", "relu"))
    for i in range(1, 18):
        path = directory / f"{i:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; use --download on an internet-enabled host")
        df = pd.read_csv(path)
        xcols = [c for c in df if c.startswith("x")]
        idcols = [c for c in df if c.startswith("id")]
        X = df[xcols].to_numpy(dtype=float) if xcols else np.empty((len(df), 0))
        plan = FEPlan.from_dataframe(df, idcols)
        got = detect_separation(df["y"].to_numpy(float), X, plan, np.ones(len(df)), cfg).separated
        expected = df["separated"].to_numpy(bool)
        bad = np.flatnonzero(got != expected)
        print(f"{i:02d}: expected={expected.sum():4d} got={got.sum():4d} mismatches={len(bad):4d}")
        if len(bad):
            failures.append((i, bad[:20].tolist()))
    if failures:
        print("FAILED:", failures)
        return 1
    print("all 17 upstream separation masks matched")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "separation_datasets")
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download_suite(args.data_dir)
    raise SystemExit(validate(args.data_dir))


if __name__ == "__main__":
    main()
