"""Small deterministic data for public workflow contracts; no shared mutable caches."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(640)
    n = 320
    firm = np.repeat(np.arange(8), 40)
    period = np.tile(np.arange(5), 64)
    c, z, noise = rng.normal(size=(3, n))
    endog = .8*z + .2*c + .4*noise
    fe = .12*firm - .05*period
    y = .4*c + 1.2*endog + fe + .2*noise + rng.normal(scale=.3, size=n)
    offset = .1*rng.normal(size=n)
    count = rng.poisson(np.exp(.5 + .12*c + .2*endog + .1*fe + offset)).astype(float)
    return pd.DataFrame(dict(y=y, count=count, c=c, endog=endog, z=z,
                             firm=np.array([f'firm-{j}' for j in firm]), period=period,
                             offset=offset, exposure=np.exp(offset)))
