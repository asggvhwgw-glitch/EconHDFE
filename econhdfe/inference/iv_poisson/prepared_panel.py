"""Compiled one-way null tests for large-sample confidence-set inversion."""
from __future__ import annotations
import numpy as np
from .panel import _matrix, _sum_cols, panel_null_scores
from .statistic import moment_test


class PreparedPanelS:
    """Cache unit encoding and outcome sufficient statistics, not fitted beta.

    Construction copies X/Z and compiles the sample once. Per-null work is
    O(N(K+L)+G L^2), with O(N+G L) additional workspace. Initial factorization
    uses numpy.unique and can cost O(N log N). No N-by-N pair construction.
    """
    def __init__(self,y,x,instruments,unit,*,method='cross_product',offset=None,
                 split=None,assume_long_panel=False,assume_cross_time_independence=False):
        y=np.asarray(y,dtype=float);n=len(y)
        X=_matrix(x,n,'x');Z=_matrix(instruments,n,'instruments')
        # Reuse the reference path's complete validation/assumption gates once.
        _,info=panel_null_scores(y,X,Z,unit,np.zeros(X.shape[1]),method=method,
            offset=offset,split=split,assume_long_panel=assume_long_panel,
            assume_cross_time_independence=assume_cross_time_independence)
        self.X=X.copy();self.Z=Z.copy();self.offset=np.zeros(n) if offset is None else np.array(offset,dtype=float,copy=True)
        self.method=method;self.info=info
        _,g=np.unique(np.asarray(unit),return_inverse=True);self.g=g;G=int(g.max())+1;self.G=G
        self.T=np.bincount(g,minlength=G)
        if method=='spj':
            self.codes=2*g+np.asarray(split,dtype=int);self.count=2*G
            self.half_T=np.bincount(self.codes,minlength=2*G).reshape(G,2)
        else:self.codes=g;self.count=G
        self.Y=_sum_cols(y,self.codes,self.count)
        self.ZY=np.column_stack([_sum_cols(y*self.Z[:,j],self.codes,self.count)
                                for j in range(self.Z.shape[1])])

    def scores(self,beta):
        b=np.asarray(beta,dtype=float).reshape(-1)
        if b.shape!=(self.X.shape[1],) or not np.all(np.isfinite(b)):
            raise ValueError('beta must specify all slopes')
        eta=self.offset+self.X@b
        if not np.all(np.isfinite(eta)) or np.ptp(eta)>600:
            raise ValueError('unsupported null predictor range')
        r=np.exp(eta-np.max(eta))
        R=_sum_cols(r,self.codes,self.count)
        ZR=np.column_stack([_sum_cols(r*self.Z[:,j],self.codes,self.count)
                            for j in range(self.Z.shape[1])])
        if self.method=='spj':
            RH=R.reshape(self.G,2);YH=self.Y.reshape(self.G,2)
            ZRH=ZR.reshape(self.G,2,-1);ZYH=self.ZY.reshape(self.G,2,-1)
            half=(ZYH-(YH/RH)[:,:,None]*ZRH)/self.half_T[:,:,None]
            full=(ZYH.sum(axis=1)-(YH.sum(axis=1)/RH.sum(axis=1))[:,None]*ZRH.sum(axis=1))/self.T[:,None]
            out=2*full-.5*half.sum(axis=1)
        elif self.method=='profile':
            out=(self.ZY-(self.Y/R)[:,None]*ZR)/self.T[:,None]
        else:
            out=(R[:,None]*self.ZY-self.Y[:,None]*ZR)/(self.T*(self.T-1))[:,None]
        info=dict(self.info,prepared=True,topology_reused=True,
                  per_null_extra_workspace='O(N+G*L)',per_null_work='O(N*(K+L)+G*L^2)')
        return out,info

    def test(self,beta):
        scores,info=self.scores(beta);out=moment_test(scores,diagnostics=info)
        out.null_value=np.asarray(beta,dtype=float).reshape(-1)
        return out
