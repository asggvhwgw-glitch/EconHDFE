"""Run normally or through tools/run_optimized.py; estimator API is unchanged."""
import json
import numpy as np
import statsmodels.api as sm
from linearmodels.datasets import wage_panel
from linearmodels.panel import PanelOLS
from linearmodels.shared import covariance

data=wage_panel.load().set_index(['nr','year'])
model=PanelOLS(data.lwage,sm.add_constant(data[['expersq','union','married']]),entity_effects=True,time_effects=True)
fit=model.fit(cov_type='clustered',cluster_entity=True,cluster_time=True)
print('Public wage panel; descriptive association, not a policy treatment effect.')
print('Covariance implementation:',covariance.__file__)
print(fit.summary)
print(json.dumps({'nobs':int(fit.nobs),'params':fit.params.to_dict(),'std_errors':fit.std_errors.to_dict(),'covariance':fit.cov.to_numpy().tolist()}))
