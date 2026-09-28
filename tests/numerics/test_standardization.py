import numpy as np
from pyreghdfe.ppml.standardize import Standardization
from pyreghdfe import ppmlhdfe, PPMLConfig


def test_reghdfe_scale_is_sample_sd_without_demeaning():
    X=np.array([[1.,3.],[2.,3.],[4.,3.],[8.,3.]])
    y=np.array([1.,2.,5.,9.])
    s=Standardization.fit(y,X)
    assert np.allclose(s.x_scale[0],np.std(X[:,0],ddof=1))
    assert s.x_scale[1]==1e-3
    assert np.allclose(s.y_scale,np.std(y,ddof=1))


def test_extreme_regressor_scaling_matches_stable_reparameterization():
    rng=np.random.default_rng(32); n=2500
    z=rng.normal(size=(n,2)); scales=np.array([1e-7,1e7])
    X=z*scales; beta=np.array([2.0e6,-2.0e-8])
    y=rng.poisson(np.exp(.3+X@beta))
    c=PPMLConfig(standardize=True,separation=(),tolerance=1e-9,target_inner_tol=1e-10)
    extreme=ppmlhdfe(y,X,vce='model',config=c)
    stable=ppmlhdfe(y,z,vce='model',config=c)
    mapped=np.r_[stable.coef[:2]/scales, stable.coef[2]]
    assert np.allclose(extreme.coef,mapped,rtol=5e-7,atol=1e-7)
