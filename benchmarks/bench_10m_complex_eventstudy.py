from __future__ import annotations
import argparse, json, resource, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from econhdfe import reghdfe, factor, reg_interaction, omit_level, omit_column


def max_rss_mib():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r / 1024.0 if sys.platform != 'darwin' else r / (1024.0 * 1024.0))


def make_dgp(n: int, seed: int = 7510):
    rng = np.random.default_rng(seed)
    nfirm = min(559_951, max(5_000, n // 18))
    nyear, ncity = 16, 347
    firm_city = rng.integers(0, ncity, nfirm, dtype=np.int16)
    firm = rng.integers(0, nfirm, n, dtype=np.int32)
    year = rng.integers(0, nyear, n, dtype=np.int8)
    city = firm_city[firm]
    province = (city.astype(np.int32) * 31 // ncity).astype(np.int8)
    region = (province.astype(np.int16) * 7 // 31).astype(np.int8)
    metro = (city.astype(np.int32) * 80 // ncity).astype(np.int8)
    city_year = (city.astype(np.int32) * nyear + year.astype(np.int32)).astype(np.int32)

    # Firm-level staggered adoption, 20% never treated.
    h = np.arange(nfirm, dtype=np.uint64) * np.uint64(11400714819323198485)
    ever_firm = ((h >> np.uint64(8)) % np.uint64(10) < 8).astype(np.int8)
    cohort_firm = (4 + ((h >> np.uint64(17)) + firm_city.astype(np.uint64) * 3) % 8).astype(np.int8)
    ever = ever_firm[firm]
    et = np.clip(year.astype(np.int16) - cohort_firm[firm].astype(np.int16), -2, 2).astype(np.int8)
    et[ever == 0] = -1

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    xsum = x1 + x2       # exact composite collinearity

    dyn = np.array([-.01, 0.0, .16, .32, .48], dtype=np.float64)  # k=-2..2, ref=-1
    alpha_f = rng.normal(scale=.8, size=nfirm).astype(np.float32)
    alpha_cy = rng.normal(scale=.5, size=ncity*nyear).astype(np.float32)
    y = np.empty(n, dtype=np.float64)
    y[:] = .35*x1; y += -.20*x2
    y += dyn[et.astype(np.int16)+2] * ever
    y += alpha_f[firm]; y += alpha_cy[city_year]
    firm_shock = rng.normal(scale=.10, size=nfirm).astype(np.float32)
    y += firm_shock[firm]
    y += rng.normal(scale=.50, size=n) * (0.85 + .15*np.abs(x1))
    del alpha_f, alpha_cy, firm_shock, city_year, h, cohort_firm, ever_firm

    data = pd.DataFrame(dict(y=y, firm=firm, year=year, city=city, province=province, region=region, metro=metro,
                event_time=et, ever_treated=ever, x1=x1, x2=x2, xsum=xsum))
    truth = {'x1':.35,'x2':-.20}
    for k,b in zip(range(-2,3),dyn):
        if k != -1: truth[f'event_time[{k}]#ever_treated'] = float(b)
    return data, truth, nfirm


def run(n: int, threads: int, memory_mb: int, seed: int):
    t0=time.perf_counter(); data,truth,nfirm=make_dgp(n,seed); dgp=time.perf_counter()-t0
    event = reg_interaction(factor('event_time',drop_base=False,name='event_time'),'ever_treated',name='event_study')
    controls=[
        'x1','x2','xsum',
        factor('year',drop_base=False,name='year_control'),
        reg_interaction(factor('region',drop_base=False,name='region'),factor('year',drop_base=False,name='year'),name='region_year_control'),
        reg_interaction(factor('province',drop_base=False,name='province'),factor('year',drop_base=False,name='year'),name='province_year_control'),
        reg_interaction(factor('metro',drop_base=False,name='metro'),factor('year',drop_base=False,name='year'),name='metro_year_control'),
        event,
    ]
    absorb=['firm','year',('region','year'),('province','year'),('metro','year'),('city','year')]
    omissions=[omit_column('xsum'), omit_level('event_time',-1,term='event_study')]
    t1=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res=reghdfe(data,y='y',x=controls,absorb=absorb,omit=omissions,
                    cluster=['firm','city'],vce='cluster',method='auto',absorb_threads=threads,
                    memory_budget_mb=memory_mb,pool_size='auto',tol=1e-8,collinearity='warn',structural_collinearity=True)
    elapsed=time.perf_counter()-t1
    est=dict(zip(res.names,map(float,res.params))); errors={k:est[k]-v for k,v in truth.items() if k in est}
    canon=res.absorb_info.get('canonicalization',{})
    automatic=res.automatic_omitted_variables
    return {
        'n':n,'nfirm_potential':nfirm,'threads':threads,'memory_budget_mb':memory_mb,
        'dgp_seconds':dgp,'estimate_seconds':elapsed,'peak_rss_mib':max_rss_mib(),
        'nobs':res.nobs,'dropped_singletons':res.dropped_singletons,'rank':res.rank,
        'df_absorbed':res.df_absorbed,'df_resid':res.df_resid,'solver':res.absorb_info.get('method'),
        'iterations':res.iterations,'requested_fe':canon.get('requested'),'effective_fe':canon.get('effective'),
        'n_dropped_fe':len(canon.get('dropped') or ()),'dropped_fe':canon.get('dropped'),
        'active_names':res.names,'user_omitted':[o['name'] for o in res.user_omitted_variables],
        'automatic_omitted_count':len(automatic),'automatic_omitted_sample':[o['name'] for o in automatic[:20]],
        'warnings':[str(w.message) for w in caught],
        'max_abs_beta_error':max(map(abs,errors.values())) if errors else None,'beta_errors':errors,
    }

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=10_000_000); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--memory-mb',type=int,default=768); ap.add_argument('--seed',type=int,default=7510); ap.add_argument('--out',default='')
    a=ap.parse_args(); result=run(a.n,a.threads,a.memory_mb,a.seed); text=json.dumps(result,indent=2,default=str); print(text)
    if a.out: Path(a.out).write_text(text+'\n')
