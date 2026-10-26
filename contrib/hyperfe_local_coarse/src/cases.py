"""Synthetic stress designs, not empirical estimates. Three RHS including an exact oracle.

N is genuinely large; unique tuples and level counts are reported. First two
observations of each worker pair are patched into a four-row contrast to supply
an exactly known residual. No identical complete FE tuples are added to pad N.
"""
import numpy as np

def make_case(name,n=1_000_000,seed=20261026):
    if n%100:raise ValueError('n must be a multiple of 100')
    rng=np.random.default_rng(seed)
    if name=='hub4':
        hub=n//10;t=4;k=(n-hub)//t
        g=np.r_[np.zeros(hub,np.int32),np.repeat(np.arange(1,k+1,dtype=np.int32),t)]
        b=rng.integers(0,1500,n);p=rng.integers(0,500,n);year=rng.integers(0,20,n)
        codes=np.column_stack([g,b,p,year]).astype(np.int32);w=np.ones(n)
        base=hub;kk=k;pairlabels=np.zeros(k,dtype=int)
    else:
        t=5 if name=='high5' else 10;k=n//t
        g=np.repeat(np.arange(k,dtype=np.int32),t);year=np.tile(np.arange(t),k)
        base=0;kk=k
        if name in ('joint4','equal_sparse','smooth_sparse'):
            blocks=64;lev=20
            community=rng.integers(0,blocks,k);comm=np.repeat(community,t)
            a=rng.integers(0,lev,n);b=rng.integers(0,lev,n);c=b-a+lev-1
            weak=rng.random(n)<.001
            # Joint relation is mostly preserved, with sparse rows that cross communities
            # and break the additive relation. Only joint4 gives those rows tiny weights.
            cb=comm.copy();cc=comm.copy()
            cb[weak]=rng.integers(0,blocks,weak.sum());cc[weak]=rng.integers(0,blocks,weak.sum())
            c[weak]=rng.integers(0,2*lev-1,weak.sum())
            codes=np.column_stack([g,comm*lev+a,cb*lev+b,cc*(2*lev-1)+c]).astype(np.int32)
            if name=='joint4':
                w=np.ones(n);w[weak]=1e-4
            elif name=='equal_sparse':
                w=np.ones(n)
            else:
                # Smooth, continuous weights with no separate low-weight class.
                w=np.exp(np.clip(rng.normal(0,.55,n),-1.5,1.5));w/=w.mean()
            pairlabels=community
        elif name=='irregular':
            # Unequal community sizes and non-aligned FE partitions; no exact additive relation.
            raw=np.minimum(rng.zipf(1.8,k),64)
            _,community=np.unique(raw,return_inverse=True)
            blocks=int(community.max())+1
            comm=np.repeat(community,t)
            bcomm=comm.copy();ccomm=comm.copy();dcomm=comm.copy()
            # Cross-community reassignment rates differ by FE and by observation.
            mb=rng.random(n)<.004;mc=rng.random(n)<.008;md=rng.random(n)<.012
            bcomm[mb]=rng.integers(0,blocks,mb.sum())
            ccomm[mc]=rng.integers(0,blocks,mc.sum())
            dcomm[md]=rng.integers(0,blocks,md.sum())
            bcat=rng.integers(0,24,n);ccat=rng.integers(0,18,n);dcat=rng.integers(0,12,n)
            codes=np.column_stack([g,bcomm*24+bcat,ccomm*18+ccat,dcomm*12+dcat]).astype(np.int32)
            w=np.exp(np.clip(rng.normal(0,.7,n),-2,2));w/=w.mean()
            pairlabels=community
        elif name=='nonadditive':
            # Worker-like groups, firms, city-year and occupation; ordinary mobility, no planted
            # multiway additive relation or clean weight split.
            nf=100000;nc=200;per=nf//nc
            city=rng.integers(0,nc,k);firm=city*per+rng.integers(0,per,k)
            firms=np.repeat(firm,t).astype(np.int32)
            move=rng.random(n)<.025
            firms[move]=rng.integers(0,nf,move.sum())
            year=np.tile(np.arange(t),k)
            cityobs=firms//per
            reloc=rng.random(n)<.01;cityobs[reloc]=rng.integers(0,nc,reloc.sum())
            cityyear=cityobs*t+year
            occn=500
            occ=np.repeat(rng.integers(0,occn,k),t)
            shift=rng.random(n)<.025;occ[shift]=rng.integers(0,occn,shift.sum())
            codes=np.column_stack([g,firms,cityyear,occ]).astype(np.int32)
            w=np.exp(np.clip(rng.normal(0,.6,n),-2,2));w/=w.mean()
            pairlabels=city
        else:
            nf=20000 if name=='high5' else (5000 if name=='weighted4' else 2500)
            nc=100;per=nf//nc
            city=rng.integers(0,nc,k);f=city*per+rng.integers(0,per,k)
            firms=np.empty((k,t),np.int32);firms[:,0]=f
            for j in range(1,t):
                move=rng.random(k)<(.2 if name=='high5' else .08)
                cross=move&(rng.random(k)<.04)
                f[move]=(f[move]//per)*per+rng.integers(0,per,move.sum())
                f[cross]=rng.integers(0,nf,cross.sum());firms[:,j]=f
            firms=firms.ravel();pairlabels=firms.reshape(k,t)[:,0]//per
            if name=='mobility3':
                codes=np.column_stack([g,firms,year]).astype(np.int32);w=np.ones(n)
            else:
                city=firms//per
                reloc=rng.random(n)<.02;city[reloc]=rng.integers(0,nc,reloc.sum())
                cy=city*t+year
                occn=1000 if name=='high5' else 120
                occ=np.repeat(rng.integers(0,occn,k),t)
                shift=rng.random(n)<.04;occ[shift]=rng.integers(0,occn,shift.sum())
                if name=='high5':
                    prod=firms%4000
                    ch=rng.random(n)<.15;prod[ch]=rng.integers(0,4000,ch.sum())
                    codes=np.column_stack([g,firms,cy,occ,prod]).astype(np.int32);w=np.ones(n)
                else:
                    codes=np.column_stack([g,firms,cy,occ]).astype(np.int32)
                    w=np.exp(np.clip(rng.normal(0,2,n),-6,6));w/=w.mean()
    # Exact oracle: weighted signed four-row cycles with no repeated full tuple.
    order=np.argsort(pairlabels,kind='stable')
    aa=order[0:(kk//2)*2:2];bb=order[1:(kk//2)*2:2]
    good=pairlabels[aa]==pairlabels[bb];aa=aa[good];bb=bb[good]
    first=base+aa*t
    i0=first;i1=first+1;i2=base+bb*t;i3=i2+1
    codes[i2,1:]=codes[i0,1:];codes[i3,1:]=codes[i1,1:]
    w[i2]=w[i0];w[i3]=w[i1]
    # Dense observed codes, including any unobserved level caused by sampling.
    for q in range(codes.shape[1]):
        _,v=np.unique(codes[:,q],return_inverse=True);codes[:,q]=v
    oracle=np.zeros(n)
    amp=rng.normal(size=len(first))/np.sqrt(1/w[i0]+1/w[i1]+1/w[i2]+1/w[i3])
    for ix,sig in [(i0,1),(i1,-1),(i2,-1),(i3,1)]:oracle[ix]=sig*amp/w[ix]
    sizes=[int(codes[:,q].max())+1 for q in range(codes.shape[1])]
    x=rng.normal(size=n);y=1.5*x+rng.normal(size=n);yt=oracle.copy()
    for q,sz in enumerate(sizes):
        y+=rng.normal(size=sz)[codes[:,q]]
        x+=.3*rng.normal(size=sz)[codes[:,q]]
        yt+=rng.normal(size=sz)[codes[:,q]]
    Y=np.column_stack([y,x,yt])
    perm=rng.permutation(n)
    return codes[perm],w[perm],Y[perm],oracle[perm]

def metadata(codes,w):
    sizes=[int(codes[:,q].max())+1 for q in range(codes.shape[1])]
    # Exact full-tuple distinct count, not a sample/extrapolation.
    e=np.unique(codes,axis=0).shape[0]
    mins=[int(np.bincount(codes[:,q]).min()) for q in range(codes.shape[1])]
    return dict(n=int(len(codes)),q=codes.shape[1],levels=sizes,total_levels=sum(sizes),
                unique_tuples=int(e),min_level_counts=mins,
                weight_min=float(w.min()),weight_max=float(w.max()),weight_sum=float(w.sum()))
