"""Create deterministic fixtures shared by Stata and econhdfe golden tests."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def make_fixture(n_firms: int = 400, n_periods: int = 50, seed: int = 20260910) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(1, n_firms + 1, dtype=np.int32), n_periods)
    year = np.tile(np.arange(1, n_periods + 1, dtype=np.int16), n_firms)
    n = len(firm)

    # Nested geography for v0.8 FE-canonicalization golden cases.
    city = (((firm - 1) % 80) + 1).astype(np.int16)
    province = (((city - 1) // 10) + 1).astype(np.int16)

    w = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    v = rng.normal(size=n)
    ux = rng.normal(size=n)
    eps = rng.normal(size=n)
    a_firm = rng.normal(scale=0.8, size=n_firms)[firm - 1]
    a_year = rng.normal(scale=0.5, size=n_periods)[year - 1]
    trend = (year.astype(float) - year.mean()) / year.std()
    slope = rng.normal(scale=0.15, size=n_firms)[firm - 1]
    x = 0.65 * z1 + 0.30 * z2 + 0.25 * w + 0.55 * v + 0.35 * ux

    # Exact composite collinearity used by explicit/automatic omission cases.
    x3 = x + w

    # Non-negative event codes keep the fixture portable to Stata factor-variable
    # syntax. code=2 is the event-time -1 reference; code=7 marks never-treated
    # rows and interacts to an exact zero because ever==0.
    cohort = rng.integers(6, max(7, n_periods - 5), size=n_firms, dtype=np.int16)
    ever_firm = (rng.random(n_firms) > 0.20).astype(np.int8)
    raw_event = np.clip(year.astype(np.int32) - cohort[firm - 1], -3, 3)
    event_code = (raw_event + 3).astype(np.int16)
    ever = ever_firm[firm - 1]
    event_code = np.where(ever == 1, event_code, 7).astype(np.int16)
    dyn = np.array([-0.10, -0.05, 0.00, 0.20, 0.35, 0.50, 0.60, 0.00])
    event_effect = dyn[event_code] * ever

    y = (
        1.50 * x + 0.35 * w + a_firm + a_year + slope * trend
        + event_effect + 0.55 * v + eps
    )

    # Weight columns are generated only after the structural fixture so adding
    # weight validation cannot alter the preceding DGP draws.
    fw = rng.integers(1, 5, size=n, dtype=np.int16)
    aw = np.exp(rng.normal(scale=0.55, size=n))
    pw = np.exp(rng.normal(scale=0.40, size=n))
    return pd.DataFrame({
        "firm": firm,
        "year": year,
        "province": province,
        "city": city,
        "c1": ((firm - 1) % 80 + 1).astype(np.int16),
        "c2": ((year - 1) % 10 + 1).astype(np.int16),
        "c3": (((firm * 7 + year * 11) % 60) + 1).astype(np.int16),
        "trend": trend,
        "event_code": event_code,
        "ever": ever,
        "w": w,
        "z1": z1,
        "z2": z2,
        "x": x,
        "x3": x3,
        "y": y,
        "fw": fw,
        "aw": aw,
        "pw": pw,
    })


def make_group_individual_fixture(n_groups: int = 1200, n_individuals: int = 300, seed: int = 20260911) -> pd.DataFrame:
    """Long membership fixture for reghdfe group()/individual() golden tests."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 6, size=n_groups, dtype=np.int16)
    group0 = np.repeat(np.arange(n_groups, dtype=np.int32), sizes)
    within = np.concatenate([np.arange(s, dtype=np.int32) for s in sizes])
    inventor0 = (group0 * 11 + within * 53) % n_individuals
    year_g = np.arange(n_groups, dtype=np.int32) % 24
    w_g = rng.normal(size=n_groups)
    z1_g = rng.normal(size=n_groups)
    z2_g = rng.normal(size=n_groups)
    v_g = rng.normal(size=n_groups)
    x_g = 0.70*z1_g + 0.25*z2_g + 0.20*w_g + 0.55*v_g + rng.normal(scale=.4, size=n_groups)
    alpha = rng.normal(scale=.8, size=n_individuals)
    member_alpha = alpha[inventor0]
    sums = np.bincount(group0, weights=member_alpha, minlength=n_groups)
    mean_alpha = sums / sizes
    tau = rng.normal(scale=.45, size=24)
    y_g = 1.45*x_g + .30*w_g + mean_alpha + tau[year_g] + .35*v_g + rng.normal(scale=.6, size=n_groups)
    return pd.DataFrame({
        "patent": group0 + 1,
        "inventor": inventor0 + 1,
        "year": year_g[group0] + 1,
        "w": w_g[group0],
        "z1": z1_g[group0],
        "z2": z2_g[group0],
        "x": x_g[group0],
        "y": y_g[group0],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures" / "golden_data.csv")
    ap.add_argument("--firms", type=int, default=400)
    ap.add_argument("--periods", type=int, default=50)
    ap.add_argument("--group-out", type=Path, default=Path(__file__).parent / "fixtures" / "golden_group_individual.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = make_fixture(args.firms, args.periods)
    df.to_csv(args.out, index=False, float_format="%.17g")
    print(f"wrote {len(df):,} rows to {args.out}")
    gdf = make_group_individual_fixture()
    gdf.to_csv(args.group_out, index=False, float_format="%.17g")
    print(f"wrote {len(gdf):,} membership rows to {args.group_out}")


if __name__ == "__main__":
    main()
