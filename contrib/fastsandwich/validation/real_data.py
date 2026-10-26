"""Public macro and wage-panel replay; run normally and with run_optimized.py."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac as baseline_hac
from linearmodels.datasets import wage_panel
from linearmodels.panel import PanelOLS
from threadpoolctl import threadpool_limits
from fastsandwich import cov_hac


def timed(call, repeats=7, inner=1):
    call()
    times=[]
    for _ in range(repeats):
        start=time.perf_counter()
        for _ in range(inner):
            result=call()
        times.append((time.perf_counter()-start)/inner)
    return result,median(times)


def run():
    with threadpool_limits(limits=1):
        macro=sm.datasets.macrodata.load_pandas().data
        growth=np.log(macro[['realcons','realdpi','realgdp']]).diff().dropna()
        fit=sm.OLS(growth.realcons,sm.add_constant(growth[['realdpi','realgdp']])).fit()
        hac=[]
        for lag in (4,100):
            ref,base_time=timed(lambda:baseline_hac(fit,nlags=lag),inner=100)
            actual,fast_time=timed(lambda:cov_hac(fit,nlags=lag),inner=100)
            np.testing.assert_allclose(actual,ref,rtol=1e-10,atol=1e-12)
            hac.append(dict(nlags=lag,baseline_seconds=base_time,fastsandwich_seconds=fast_time,
                            max_covariance_abs_error=float(np.max(abs(actual-ref)))))
        data=wage_panel.load().set_index(['nr','year'])
        model=PanelOLS(data.lwage,sm.add_constant(data[['expersq','union','married']]),
                       entity_effects=True,time_effects=True)
        def panel_fit():
            fitted=model.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
            _=fitted.cov  # materialize lazy covariance within timing
            return fitted
        panel,panel_time=timed(panel_fit)
    return dict(macro=dict(nobs=int(fit.nobs),results=hac),panel=dict(nobs=int(panel.nobs),
                coefficients=panel.params.to_list(),covariance=panel.cov.to_numpy().tolist(),
                standard_errors=panel.std_errors.to_list(),seconds=panel_time),
                repeats=7,blas_threads=1,
                macro_sha256=hashlib.sha256(macro.to_csv(index=False).encode()).hexdigest(),
                wage_panel_sha256=hashlib.sha256(data.to_csv().encode()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)
