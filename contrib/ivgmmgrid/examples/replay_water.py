"""Replay real public water-demand data entirely in Python, with frozen preselection.

Download Usage_July15_May18.dta from https://doi.org/10.5281/zenodo.10965745,
then run: python examples/replay_water.py --data /path/Usage_July15_May18.dta
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ivgmmgrid import select_water_grid

DATA_SHA256 = '13f63946ea39f462f1f43057ab3d79ecc5149d2a77367af042626fe6d9539142'
FIXTURES = Path(__file__).resolve().parents[1]/'validation/water'


def prepare_data(path, seed):
    """Recreate float-stored differences from raw public data and household draws."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError('Expected the pinned v2 water data; SHA256 does not match')
    metadata = json.loads((FIXTURES/f'seed_{seed}.json').read_text())
    archive = np.load(FIXTURES/f'seed_{seed}.npz')
    raw = pd.read_stata(path)[['id','t','lnC','lnC_1','p','Nd','lnP']].sort_values(['id','t'])
    raw['Nd'] = (raw.Nd.astype(float)/1000).astype(raw.Nd.dtype)
    for j in range(2,5):
        raw[f'lnC_{j}'] = raw.groupby('id',sort=False).lnC.shift(j).astype(np.float32)
    draw = pd.DataFrame(archive['household_draw'],columns=['id','bsid'])
    d = draw.merge(raw,on='id',how='left',validate='many_to_many').sort_values(['bsid','t']).reset_index(drop=True)
    lag = metadata['difference_lag']
    for col in ['lnC','lnC_1','lnC_2','lnC_3','lnC_4','p','Nd','lnP']:
        prior=d.groupby('bsid',sort=False)[col].shift(lag)
        d['D'+col]=(d[col].astype(float)-prior.astype(float)).astype(np.float32)
    for j,spec in enumerate(metadata['instruments'],1):
        lag,variable=spec
        prior=d.groupby('bsid',sort=False)[variable].shift(lag)
        time_prior=d.groupby('bsid',sort=False).t.shift(lag)
        d[f'iv_{j}']=prior.where(d.t-time_prior==lag).astype(float)
    return d,archive,metadata


def model_vector(m):
    return np.r_[m.nobs,m.nclusters,m.rmse,m.rss,m.df_resid,m.coefficients,
                 m.covariance[np.triu_indices(3)]]


def replay(path,seed):
    d,archive,metadata=prepare_data(path,seed)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        result=select_water_grid(d.DlnC.to_numpy(),d[['Dp','DNd']].to_numpy(),
            d[[f'DlnC_{j}' for j in range(1,5)]].to_numpy(),d.filter(regex='^iv_').to_numpy(),
            d.t.to_numpy(),d.bsid.to_numpy(),archive['gammas'])
    seconds=time.perf_counter()-start
    actual=np.array([model_vector(m) for grid in result.grids for m in grid.models])
    expected=archive['grid']
    np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=1e-9)
    np.testing.assert_allclose(model_vector(result.model),archive['final'],rtol=2e-7,atol=1e-9)
    assert result.order==metadata['selected_order']
    assert np.float32(result.gamma)==np.float32(metadata['selected_gamma'])
    keys=d.iloc[result.model.sample_indices][['id','bsid','t']].sort_values(['id','bsid','t']).to_numpy()
    np.testing.assert_array_equal(keys,archive['final_sample_keys'])
    return dict(seed=seed,grid_points=len(actual),order=result.order,gamma=result.gamma,
                nobs=result.model.nobs,nclusters=result.model.nclusters,seconds=seconds,
                max_coefficient_abs_error=float(np.max(abs(actual[:,5:8]-expected[:,5:8]))),
                max_covariance_abs_error=float(np.max(abs(actual[:,8:]-expected[:,8:]))),
                final_sample_keys_exact=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    a=p.parse_args()
    for seed in (1,7,2026):
        print(json.dumps(replay(a.data,seed)),flush=True)
