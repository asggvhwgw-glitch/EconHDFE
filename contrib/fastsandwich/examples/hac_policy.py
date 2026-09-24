"""Public US macro data: identical OLS coefficients and HAC standard errors."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac as baseline_hac
from fastsandwich import cov_hac

data=sm.datasets.macrodata.load_pandas().data
growth=np.log(data[['realcons','realdpi','realgdp']]).diff().dropna()
x=sm.add_constant(growth[['realdpi','realgdp']])
fit=sm.OLS(growth['realcons'],x).fit()
reference=baseline_hac(fit,nlags=4)
accelerated=cov_hac(fit,nlags=4)
np.testing.assert_allclose(accelerated,reference,rtol=1e-10,atol=1e-12)
print('Descriptive growth regression; no causal interpretation. T=',fit.nobs)
for name,coef,se in zip(x.columns,fit.params,np.sqrt(np.diag(accelerated))):
    print(f'{name:10s} coefficient={coef: .8f} HAC_se={se: .8f}')
print('Max covariance difference:',np.max(np.abs(accelerated-reference)))
