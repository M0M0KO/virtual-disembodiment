"""Numerical primitives for matched development representation diagnostics."""
from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, svd
from scipy.stats import rankdata

DIMENSIONS = (2,4,8,16,32)
LAMBDA = 1.0

def normalized(cal, held):
    mean=cal.mean(axis=0)
    std=cal.std(axis=0)
    std=np.where(std>1e-8,std,1.)
    return (cal-mean)/std,(held-mean)/std

def pca_views(cal, held):
    cal,held=normalized(cal.astype(float),held.astype(float))
    _,s,vt=svd(cal,full_matrices=False,check_finite=False)
    rank=int(np.sum(s>s[0]*max(cal.shape)*np.finfo(float).eps))
    if rank==0:
        raise ValueError('Zero calibration rank')
    spectrum=s**2/(s**2).sum()
    views={}
    c,h=cal@vt[:rank].T,held@vt[:rank].T
    for d in sorted(set(min(d,rank) for d in DIMENSIONS)):
        views[str(d)]=(c[:,:d],h[:,:d])
    views['full']=(cal,held)
    return views,rank,spectrum

def scale_predictors(cal, held):
    c,h=normalized(cal,held)
    return c/np.sqrt(cal.shape[1]),h/np.sqrt(cal.shape[1])

def cosine(x,y):
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator>1e-15 else np.nan

def geometry(x,y):
    if x.shape[0]<8 or y.shape[0]!=x.shape[0]:
        return np.nan,np.nan
    def centered_gram(a):
        b=a-a.mean(axis=0)
        return (b@b.T).ravel()
    def rsa_vector(a):
        a=a-a.mean(axis=1,keepdims=True)
        norms=np.linalg.norm(a,axis=1,keepdims=True)
        if np.any(norms<1e-12):
            return None
        unit=a/norms
        corr=unit@unit.T
        return rankdata(1-np.clip(corr[np.triu_indices(len(a),1)],-1,1))
    cka=cosine(centered_gram(x),centered_gram(y))
    a,b=rsa_vector(x),rsa_vector(y)
    rsa=np.nan if a is None or b is None else cosine(a-a.mean(),b-b.mean())
    return cka,rsa

def spectrum(x):
    if len(x)<8:
        return (np.nan,np.nan,np.full(6,np.nan))
    z=x-x.mean(axis=0)
    gram=z@z.T if len(z)<z.shape[1] else z.T@z
    vals=np.maximum(eigh(gram,eigvals_only=True,check_finite=False),0)[::-1]
    if vals.sum()<=1e-14:
        return (np.nan,np.nan,np.full(6,np.nan))
    p=vals/vals.sum()
    positive=p[p>0]
    return 1/(p@p),np.exp(-(positive*np.log(positive)).sum()),np.array([p[:d].sum() for d in DIMENSIONS]+[1.])

def fit_groups(mask, min_rows=32):
    buckets={}
    for j in range(mask.shape[1]):
        valid=mask[:,j]
        if valid.sum()>=min_rows:
            buckets.setdefault(valid.tobytes(),(valid,[]))[1].append(j)
    return list(buckets.values())

def person_weights(person):
    _,ind,counts=np.unique(person,return_inverse=True,return_counts=True)
    return np.sqrt(len(person)/(len(counts)*counts[ind]))

def diagnostic_w(z,v,y,mask,people):
    """Fixed ridge, one solve per unique calibration target mask."""
    out=np.full((len(v),y.shape[1]),np.nan)
    z,v=scale_predictors(z,v)
    for keep,cols in fit_groups(mask):
        w=person_weights(people[keep])
        yy=y[keep][:,cols]
        loc=np.average(yy,axis=0,weights=w*w)
        features=z[keep]*w[:,None]
        a=features.T@features
        a.flat[::a.shape[0]+1]+=LAMBDA
        coef=cho_solve(cho_factor(a,check_finite=False),features.T@((yy-loc)*w[:,None]),check_finite=False)
        out[:,cols]=v@coef+loc
    return out

class Conditional:
    """Identical full-history/dose/availability baseline for all paired arms."""
    def __init__(self,x,z,y,mask,people):
        x,z=scale_predictors(x,z)
        self.groups=[]
        self.base=np.full((len(z),y.shape[1]),np.nan)
        for keep,cols in fit_groups(mask):
            w=person_weights(people[keep])
            c=x[keep]*w[:,None]
            k=c@c.T
            k.flat[::len(k)+1]+=LAMBDA
            factor=cho_factor(k,check_finite=False)
            test=z@c.T
            values=y[keep][:,cols]
            mean=np.average(values,axis=0,weights=w*w)
            dy=cho_solve(factor,(values-mean)*w[:,None],check_finite=False)
            self.base[:,cols]=test@dy+mean
            self.groups.append((keep,cols,w,factor,test,dy,mean))

    def prepare(self,cal,held):
        """Cache full Schur matrices, then score each declared prefix."""
        cal,held=normalized(cal,held)
        prepared=[]
        for keep,cols,w,factor,test,dy,mean in self.groups:
            q=cal[keep]*w[:,None]
            aq=cho_solve(factor,q,check_finite=False)
            prepared.append((cols,q.T@aq,q.T@dy,held-test@aq))
        return prepared

    def predict(self,prepared,d):
        pred=self.base.copy()
        for cols,cov,rhs,residual in prepared:
            mat=cov[:d,:d].copy()
            mat.flat[::d+1]+=LAMBDA*d
            beta=cho_solve(cho_factor(mat,check_finite=False),rhs[:d],check_finite=False)
            pred[:,cols]+=residual[:,:d]@beta
        return pred

@lru_cache(maxsize=48)
def resampling_indices(n,seed,n_resamples):
    rng=np.random.default_rng(seed)
    return rng.integers(0,n,(n_resamples,n),dtype=np.int32)

@lru_cache(maxsize=48)
def sign_flips(n,seed,n_resamples):
    return 2*np.random.default_rng(seed+1).integers(0,2,(n_resamples,n),dtype=np.int8)-1

def paired(values,seed=20261125,n_resamples=10000,test=False):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    n=len(values)
    if n==0:
        return dict(people=0,effect=np.nan,lo=np.nan,hi=np.nan,p=np.nan)
    effect=float(values.mean())
    if n<8:
        return dict(people=n,effect=effect,lo=np.nan,hi=np.nan,p=np.nan)
    boot=np.quantile(np.mean(values[resampling_indices(n,seed,n_resamples)],axis=1),[.025,.975])
    p=np.nan
    if test:
        signs=sign_flips(n,seed,n_resamples)
        p=(1+np.sum(np.abs(np.mean(signs*values,axis=1))>=abs(effect)-1e-15))/(n_resamples+1)
    return dict(people=n,effect=effect,lo=float(boot[0]),hi=float(boot[1]),p=float(p))

def holm(pairs):
    valid=sorted((val,key) for key,val in pairs.items() if np.isfinite(val))
    adjusted={}
    previous=0.
    for i,(val,key) in enumerate(valid):
        previous=min(1.,max(previous,val*(len(valid)-i)))
        adjusted[key]=previous
    return adjusted
