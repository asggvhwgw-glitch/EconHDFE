"""Deterministic weighted projection oracles including mostly-unique FE tuples.

For each four-edge bipartite cycle use weighted flow (+e,-e,-e,+e).
The third factor is constant on that cycle. Hence D' W r = 0 exactly in
real arithmetic, without duplicating complete FE tuples. All cycles are
randomly superimposed, then observations are shuffled. y = D a + r.
Changed weights multiply all four edges by the same positive cycle factor.
"""
import numpy as np


def corpus(case, n, rhs, seed):
    rng = np.random.default_rng(seed)
    n = n // 16 * 16
    paired = case.startswith('paired') or case=='repeated'
    if paired:
        repeats = 16 if case=='repeated' else 2
        m=n//repeats
        dims=(max(2,n//10), max(2,n//12), max(2,n//16)) if case=='paired_three' else (max(2,n//12),max(2,n//20),16)
        base=np.column_stack([rng.integers(g,size=m,dtype=np.uint32) for g in dims])
        g=np.repeat(base,repeats,axis=0)
        unit=np.repeat(np.arange(m,dtype=np.uint32),repeats)
        w=np.repeat(np.exp(rng.normal(scale=.5,size=m)),repeats)
        e=rng.normal(size=(n//2,rhs))
        residual=np.empty((n,rhs)); residual[0::2]=e;residual[1::2]=-e
    else:
        m=n//4
        if case=='unique_one':
            ga,gb,gc=max(2,n//8),48,16
        elif case=='unique_two':
            ga,gb,gc=max(2,n//8),max(2,n//12),16
        else:
            ga,gb,gc=max(2,n//10),max(2,n//12),max(2,n//16)
        if case=='weak_cycle':
            communities=32
            la,lb=max(2,n//24//communities),max(2,n//32//communities)
            c=rng.integers(communities,size=m,dtype=np.uint32)
            a0=c*la+rng.integers(la,size=m,dtype=np.uint32)
            a1=c*la+(a0%la+rng.integers(1,la,size=m,dtype=np.uint32))%la
            b0=c*lb+rng.integers(lb,size=m,dtype=np.uint32)
            b1=c*lb+(b0%lb+rng.integers(1,lb,size=m,dtype=np.uint32))%lb
            bridges=rng.random(m)<.001
            a1[bridges]=(a1[bridges]+la)%(la*communities)
            g=np.column_stack([np.column_stack([a0,a0,a1,a1]).ravel(),np.column_stack([b0,b1,b0,b1]).ravel()])
        else:
            a0=rng.integers(ga,size=m,dtype=np.uint32)
            a1=(a0+rng.integers(1,ga,size=m,dtype=np.uint32))%ga
            b0=rng.integers(gb,size=m,dtype=np.uint32)
            b1=(b0+rng.integers(1,gb,size=m,dtype=np.uint32))%gb
            c=rng.integers(gc,size=m,dtype=np.uint32)
            g=np.column_stack([np.column_stack([a0,a0,a1,a1]).ravel(),np.column_stack([b0,b1,b0,b1]).ravel(),np.repeat(c,4)])
        unit=np.repeat(np.arange(m,dtype=np.uint32),4)
        w=np.exp(rng.normal(scale=.5,size=n))
        e=rng.normal(size=(m,rhs))
        residual=np.repeat(e,4,axis=0)*np.tile([1.,-1.,-1.,1.],m)[:,None]/w[:,None]
    y=residual.copy()
    # Missing numeric category labels are deliberately compacted outside timing
    # to satisfy the unchanged legacy benchmark's input contract.
    for j in range(g.shape[1]):
        values,inverse=np.unique(g[:,j],return_inverse=True)
        g[:,j]=inverse.astype(np.uint32)
        y+=rng.normal(scale=.3,size=(len(values),rhs))[g[:,j]]
    order=rng.permutation(n)
    return g[order],y[order],w[order],residual[order],unit[order]


def updated_weights(w, unit, step):
    if step==0:return w
    return w*np.exp(.2*np.sin(unit*(.019+.003*step)))


def forward_error(out, truth, w, chunk=65536):
    num=np.zeros(out.shape[1]);den=np.zeros_like(num)
    for i in range(0,len(w),chunk):
        a,b,ww=out[i:i+chunk],truth[i:i+chunk],w[i:i+chunk,None]
        num+=np.sum(ww*(a-b)**2,axis=0)
        den+=np.sum(ww*b*b,axis=0)
    return float(np.sqrt(num/np.maximum(den,np.finfo(float).tiny)).max())
