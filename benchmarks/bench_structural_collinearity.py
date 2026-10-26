"""Benchmark structural pruning of nested interaction controls."""
import sys, time, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from econhdfe import factor, reg_interaction
from econhdfe.design import build_design


def main(n=150_000, seed=733):
    rng=np.random.default_rng(seed)
    year=rng.integers(0,10,n,dtype=np.int32)
    city=rng.integers(0,200,n,dtype=np.int32)
    province=city//10
    region=province//4
    x=rng.normal(size=n)
    df=pd.DataFrame({'year':year,'city':city,'province':province,'region':region,'x':x})
    city_year=(city.astype(np.int64)*10+year).astype(np.int32)
    specs=[
        'x', factor('year',drop_base=False),
        reg_interaction(factor('region',drop_base=False),factor('year',drop_base=False),name='region_year'),
        reg_interaction(factor('province',drop_base=False),factor('year',drop_base=False),name='province_year'),
    ]
    t=time.perf_counter(); a=build_design(
        df,specs,n,absorbed_groups=[city_year],absorbed_names=['city#year'],structural=True
    ); ta=time.perf_counter()-t
    print(f'structural: {ta:.3f}s requested={len(a.requested_names)} materialized={a.values.shape[1]} memory={a.values.nbytes/2**20:.1f} MiB')
    del a; gc.collect()
    t=time.perf_counter(); b=build_design(df,specs,n,structural=False); tb=time.perf_counter()-t
    print(f'full:       {tb:.3f}s requested={len(b.requested_names)} materialized={b.values.shape[1]} memory={b.values.nbytes/2**20:.1f} MiB')

if __name__ == '__main__':
    main()
