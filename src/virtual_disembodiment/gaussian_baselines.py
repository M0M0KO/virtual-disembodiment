"""Fitted persistence, VARX and latent linear-Gaussian state-space controls.

All normalization is supplied from the development partition. Fitting uses
participant-weighted sufficient statistics. Forecasting never reads a future
neural/peripheral observation, and known drug forcing is explicit.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import erf

from .run_io import atomic_json,atomic_npz,file_hash,read_json,utcnow


def forward_fill(values):
    """Forward-only fill. Initial missing values are training-standardized zero."""
    x=np.asarray(values,float)
    index=np.maximum.accumulate(np.where(np.isfinite(x),np.arange(len(x))[:,None],-1),axis=0)
    filled=x[np.maximum(index,0),np.arange(x.shape[1])[None,:]]
    return np.where(index>=0,filled,0.)


def numpy_neural_scores(mean,scale,target,mask,horizon_weights=(.2,.3,.5)):
    at=[0,4,14]
    mean,scale,target,mask=[np.asarray(x)[:,at] for x in (mean,scale,target,mask)]
    scale=np.maximum(scale,1e-6);z=(target-mean)/scale
    crps=scale*(z*erf(z/math.sqrt(2))+math.sqrt(2/math.pi)*np.exp(-z*z/2)-1/math.sqrt(math.pi))
    counts=mask.sum(axis=-1)
    if (counts==0).any():raise ValueError('A primary horizon has no valid neural features')
    scores=(crps*mask).sum(axis=-1)/counts
    coverage=(((target-mean).__abs__()<=1.6448536269514722*scale)*mask).sum(axis=-1)/counts
    if not np.isfinite(scores).all():raise FloatingPointError('Unstable/nonfinite baseline forecast')
    weights=np.asarray(horizon_weights)
    return scores@weights,coverage@weights


def standard_record(dataset,record):
    a=record['arrays'];name='neural_'+dataset.pipeline
    y,mask=dataset.standardize(a[name],name)
    return y.astype(float),mask,a['forcing_mg'].astype(float)/10.


@dataclass
class Persistence:
    scale:np.ndarray

    def predict(self,rows):
        means=[]
        for row in rows:
            x=np.where(row['history_neural_mask'],row['history_neural'],np.nan)
            means.append(np.broadcast_to(forward_fill(x)[-1],(15,x.shape[1])))
        mean=np.asarray(means)
        return mean,np.broadcast_to(self.scale,mean.shape)

    @classmethod
    def fit(cls,dataset):
        moments=[]
        for record in dataset.records:
            if not len(record['origins']):continue
            y,mask,_=standard_record(dataset,record)
            filled=forward_fill(np.where(mask,y,np.nan))
            origins=record['origins'];target=y[origins[:,None]+np.arange(15)[None,:]]
            residual=target-filled[origins-1,None,:]
            moments.append(np.mean(residual*residual,axis=0))
        if not moments:raise ValueError('No development examples for persistence calibration')
        return cls(np.sqrt(np.maximum(np.mean(moments,axis=0),1e-8)))

    def save(self,folder):
        folder=Path(folder);atomic_npz(folder/'model.npz',scale=self.scale)
        atomic_json(folder/'model.json',{'family':'persistence','scale_fit_partition':'development',
                    'participant_weighted':True,'model_sha256':file_hash(folder/'model.npz')})


@dataclass
class VARX:
    coefficient:np.ndarray
    residual_covariance:np.ndarray
    order:int
    neural_dim:int
    ridge:float

    @classmethod
    def fit(cls,dataset,order=5,ridge=.01):
        d=16;k=d*order+3
        xx=np.zeros((k,k));xy=np.zeros((k,d));yy=np.zeros((d,d));people=0;observations=0
        for record in dataset.records:
            origins=record['origins'];origins=origins[origins>=order]
            if not len(origins):continue
            y,mask,u=standard_record(dataset,record)
            x=forward_fill(np.where(mask,y,np.nan))
            design=np.column_stack([*[x[origins-lag] for lag in range(1,order+1)],u[origins],np.ones(len(origins))])
            target=y[origins]
            weight=1/len(origins)
            xx+=weight*(design.T@design);xy+=weight*(design.T@target);yy+=weight*(target.T@target)
            people+=1;observations+=len(origins)
        if not people:raise ValueError('No eligible development states for VAR')
        xx/=people;xy/=people;yy/=people
        penalty=np.eye(k)*ridge;penalty[-1,-1]=ridge*.001
        coefficient=np.linalg.solve(xx+penalty,xy)
        covariance=yy-xy.T@coefficient-coefficient.T@xy+coefficient.T@xx@coefficient
        covariance=positive_covariance(covariance,1e-6)
        model=cls(coefficient,covariance,order,d,ridge)
        model.fit_summary={'participants':people,'one_step_observations':observations,'source_partition':'development'}
        return model

    def forecast_covariances(self,steps=15):
        # Impulse responses avoid building a dense order*d companion covariance.
        d=self.neural_dim
        a=[self.coefficient[i*d:(i+1)*d].T for i in range(self.order)]
        impulse=[np.eye(d)];cov=[];running=np.zeros((d,d))
        for step in range(steps):
            if step:
                impulse.append(sum((a[lag-1]@impulse[step-lag] for lag in range(1,min(step,self.order)+1)),np.zeros((d,d))))
            running=running+impulse[step]@self.residual_covariance@impulse[step].T
            cov.append(running.copy())
        return np.asarray(cov)

    def predict(self,rows):
        histories=np.asarray([forward_fill(np.where(r['history_neural_mask'],r['history_neural'],np.nan))[-self.order:] for r in rows])
        if histories.shape[1]!=self.order:raise ValueError('VAR order exceeds supplied neural history')
        means=[]
        for step in range(15):
            lagged=histories[:,-np.arange(1,self.order+1),:].reshape(len(rows),-1)
            forcing=np.asarray([r['future_forcing'][step] for r in rows])
            design=np.column_stack([lagged,forcing,np.ones(len(rows))])
            mean=design@self.coefficient
            means.append(mean);histories=np.concatenate([histories[:,1:],mean[:,None,:]],axis=1)
        mean=np.stack(means,axis=1)
        scale=np.sqrt(np.maximum(np.diagonal(self.forecast_covariances(),axis1=1,axis2=2),1e-8))
        return mean,np.broadcast_to(scale,mean.shape)

    def save(self,folder):
        folder=Path(folder);atomic_npz(folder/'model.npz',coefficient=self.coefficient,residual_covariance=self.residual_covariance)
        atomic_json(folder/'model.json',{'family':'VARX','order':self.order,'ridge':self.ridge,
                    'neural_dim':self.neural_dim,'forcing':'explicit supplied drug schedule; no future observations',
                    'fit':getattr(self,'fit_summary',None),'model_sha256':file_hash(folder/'model.npz')})


def positive_covariance(covariance,floor=1e-6):
    covariance=(covariance+covariance.T)/2
    values,vectors=np.linalg.eigh(covariance)
    return (vectors*np.maximum(values,floor))@vectors.T


def kalman_filter(y,mask,u,A,B,C,Q,R,keep=False):
    """Causal filtering with per-feature missing observations and Joseph update."""
    y=np.asarray(y,float);mask=np.asarray(mask,bool);u=np.asarray(u,float)
    latent=A.shape[0];m=np.zeros(latent);P=np.eye(latent);ll=0.
    filtered=[];covariances=[];predicted=[];predicted_cov=[]
    eye=np.eye(latent)
    for t in range(len(y)):
        pred=A@m+B@u[t];pp=A@P@A.T+Q
        use=np.flatnonzero(mask[t])
        if use.size:
            c=C[use];r=R[use]
            innovation=y[t,use]-c@pred
            s=c@pp@c.T+np.diag(r)
            factor=cho_factor(s,lower=True,check_finite=False)
            gain=cho_solve(factor,c@pp,check_finite=False).T
            m=pred+gain@innovation
            residual=eye-gain@c
            P=residual@pp@residual.T+(gain*r)@gain.T
            ll-=.5*(len(use)*math.log(2*math.pi)+2*np.log(np.diag(factor[0])).sum()+innovation@cho_solve(factor,innovation,check_finite=False))
        else:m,P=pred,pp
        P=(P+P.T)/2
        if keep:
            filtered.append(m.copy());covariances.append(P.copy());predicted.append(pred);predicted_cov.append(pp)
    if not np.isfinite(m).all() or not np.isfinite(P).all():raise FloatingPointError('Nonfinite Kalman state')
    if keep:return np.asarray(filtered),np.asarray(covariances),np.asarray(predicted),np.asarray(predicted_cov),ll
    return m,P,ll


def lds_expectation(task):
    y,mask,u,parameters=task
    A,B,C,Q,R=parameters
    mf,pf,mp,pp,ll=kalman_filter(y,mask,u,A,B,C,Q,R,keep=True)
    smooth=mf.copy();sp=pf.copy();n=len(y);latent=A.shape[0]
    gains=np.zeros((max(n-1,0),latent,latent))
    for t in range(n-2,-1,-1):
        g=np.linalg.solve(pp[t+1],A@pf[t]).T;gains[t]=g
        smooth[t]=mf[t]+g@(smooth[t+1]-mp[t+1])
        sp[t]=pf[t]+g@(sp[t+1]-pp[t+1])@g.T
        sp[t]=(sp[t]+sp[t].T)/2
    zz=sp+smooth[:,:,None]*smooth[:,None,:]
    d=y.shape[1];inputs=u.shape[1]
    xx=np.zeros((latent+inputs,latent+inputs));xy=np.zeros((latent+inputs,latent));yy=np.zeros((latent,latent))
    for t in range(1,n):
        reg=np.r_[smooth[t-1],u[t]]
        moment=reg[:,None]*reg[None,:];moment[:latent,:latent]+=sp[t-1]
        cross=np.outer(reg,smooth[t]);cross[:latent]+=gains[t-1]@sp[t]
        xx+=moment;xy+=cross;yy+=zz[t]
    denom=max(n-1,1);xx/=denom;xy/=denom;yy/=denom
    obs_xx=np.zeros((d,latent,latent));obs_xy=np.zeros((d,latent));obs_yy=np.zeros(d);obs_count=np.zeros(d)
    for j in range(d):
        use=mask[:,j];count=int(use.sum())
        if count:
            obs_xx[j]=zz[use].sum(axis=0)/count
            obs_xy[j]=y[use,j]@smooth[use]/count
            obs_yy[j]=np.mean(y[use,j]**2);obs_count[j]=1
    return {'xx':xx,'xy':xy,'yy':yy,'obs_xx':obs_xx,'obs_xy':obs_xy,'obs_yy':obs_yy,'obs_count':obs_count,
            'loglik_per_observation':ll/max(int(mask.sum()),1),'observations':int(mask.sum())}


@dataclass
class LinearGaussianStateSpace:
    A:np.ndarray
    B:np.ndarray
    C:np.ndarray
    Q:np.ndarray
    R:np.ndarray
    ridge:float

    @classmethod
    def fit(cls,dataset,latent_dim=8,ridge=.001,initial_radius=.98,max_iter=30,tolerance=1e-4,workers=4,progress=None,
            resume_state=None,checkpoint=None):
        sequences=[];covariance=np.zeros((16,16))
        for record in dataset.records:
            if not len(record['origins']):continue
            y,mask,u=standard_record(dataset,record)
            mask&=record['arrays']['target_mask_'+dataset.pipeline]
            filled=np.where(mask,y,0.)
            covariance+=(filled.T@filled)/max(len(y),1)
            sequences.append((y,mask,u))
        if not sequences:raise ValueError('No development sequences for latent state-space fitting')
        covariance/=len(sequences);values,vectors=np.linalg.eigh(covariance)
        C=vectors[:,-latent_dim:]*np.sqrt(np.maximum(values[-latent_dim:],.05))[None,:]
        A=np.eye(latent_dim)*initial_radius;B=np.zeros((latent_dim,2));Q=np.eye(latent_dim)*.05
        R=np.maximum(np.diag(covariance-C@C.T),.05)
        history=[]
        if resume_state is not None:
            A,B,C,Q,R=[np.array(resume_state[k],copy=True) for k in ('A','B','C','Q','R')]
            if A.shape!=(latent_dim,latent_dim) or C.shape!=(16,latent_dim):
                raise ValueError('LDS checkpoint dimensions differ from the requested model')
            history=list(resume_state['history'])
        converged=(len(history)>=5 and abs(history[-1]['development_loglik_per_observation']-
                         history[-2]['development_loglik_per_observation'])<tolerance)
        with (ProcessPoolExecutor(max_workers=workers) if workers>1 else nullcontext()) as pool:
            for iteration in range(len(history),max_iter if not converged else len(history)):
                tasks=[(y,m,u,(A,B,C,Q,R)) for y,m,u in sequences]
                results=list(pool.map(lds_expectation,tasks)) if pool else list(map(lds_expectation,tasks))
                xx=np.mean([s['xx'] for s in results],axis=0);xy=np.mean([s['xy'] for s in results],axis=0)
                yy=np.mean([s['yy'] for s in results],axis=0)
                W=np.linalg.solve(xx+ridge*np.eye(xx.shape[0]),xy).T
                A,B=W[:,:latent_dim],W[:,latent_dim:]
                radius=float(np.max(np.abs(np.linalg.eigvals(A))))
                # A registered stable model family, not a post-test repair.
                if radius>.995:A=A*(.995/radius)
                W=np.column_stack([A,B]);Q=positive_covariance(yy-W@xy-xy.T@W.T+W@xx@W.T,1e-5)
                count=np.sum([s['obs_count'] for s in results],axis=0)
                for j in range(16):
                    if not count[j]:raise ValueError('No observed training values for an EEG feature')
                    sxx=sum(s['obs_xx'][j] for s in results)/count[j]
                    sxy=sum(s['obs_xy'][j] for s in results)/count[j]
                    syy=sum(s['obs_yy'][j] for s in results)/count[j]
                    C[j]=np.linalg.solve(sxx+ridge*np.eye(latent_dim),sxy)
                    R[j]=max(float(syy-2*C[j]@sxy+C[j]@sxx@C[j]),1e-5)
                ll=float(np.mean([s['loglik_per_observation'] for s in results]))
                history.append({'iteration':iteration+1,'development_loglik_per_observation':ll,'unconstrained_transition_radius':radius})
                if checkpoint:checkpoint({'A':A,'B':B,'C':C,'Q':Q,'R':R,'history':history})
                if progress:progress(history[-1])
                if iteration>=4 and abs(history[-1]['development_loglik_per_observation']-history[-2]['development_loglik_per_observation'])<tolerance:break
        result=cls(A,B,C,Q,R,ridge);result.fit_summary={'em_history':history,'participants':len(sequences),
                        'observations':sum(int(m.sum()) for _,m,_ in sequences),'latent_dim':latent_dim,
                        'initial_radius':initial_radius,'training_partition':'development',
                        'stable_transition_radius_bound':.995}
        return result

    def predict(self,rows):
        means=[];scales=[]
        for row in rows:
            y=row['history_neural'];mask=row['history_neural_mask'];u=row['history_forcing']
            mean,cov,_=kalman_filter(y,mask,u,self.A,self.B,self.C,self.Q,self.R)
            mm=[];ss=[]
            for forced in row['future_forcing']:
                mean=self.A@mean+self.B@forced;cov=self.A@cov@self.A.T+self.Q
                mm.append(self.C@mean);ss.append(np.sqrt(np.maximum(np.diag(self.C@cov@self.C.T)+self.R,1e-8)))
            means.append(mm);scales.append(ss)
        return np.asarray(means),np.asarray(scales)

    def save(self,folder):
        folder=Path(folder);atomic_npz(folder/'model.npz',A=self.A,B=self.B,C=self.C,Q=self.Q,R=self.R)
        atomic_json(folder/'model.json',{'family':'latent_linear_Gaussian_state_space','ridge':self.ridge,
                    'fit':getattr(self,'fit_summary',None),'model_sha256':file_hash(folder/'model.npz')})


def load_baseline(folder):
    folder=Path(folder);meta=read_json(folder/'model.json')
    if file_hash(folder/'model.npz')!=meta['model_sha256']:raise ValueError('Baseline parameter checksum conflict')
    with np.load(folder/'model.npz',allow_pickle=False) as z:
        if meta['family']=='persistence':return Persistence(z['scale'])
        if meta['family']=='VARX':return VARX(z['coefficient'],z['residual_covariance'],meta['order'],meta['neural_dim'],meta['ridge'])
        if meta['family']=='latent_linear_Gaussian_state_space':
            return LinearGaussianStateSpace(z['A'],z['B'],z['C'],z['Q'],z['R'],meta['ridge'])
    raise ValueError('Unrecognized baseline family')


def evaluate_baseline(model,dataset,output,batch_size=128,horizon_weights=(.2,.3,.5)):
    output=Path(output);by_person=defaultdict(list);coverage=defaultdict(list)
    pred=[];sd=[];ys=[];masks=[];people=[];recordings=[];origins=[];forcing=[]
    rows=[]
    def consume(rows):
        mean,scale=model.predict(rows)
        y=np.asarray([r['target_neural'] for r in rows]);mask=np.asarray([r['target_neural_mask'] for r in rows])
        scores,cov=numpy_neural_scores(mean,scale,y,mask,horizon_weights)
        for r,s,c in zip(rows,scores,cov):
            by_person[r['participant_id']].append(float(s));coverage[r['participant_id']].append(float(c))
        pred.append(mean.astype(np.float32));sd.append(scale.astype(np.float32));ys.append(y);masks.append(mask)
        people.extend(r['participant_id'] for r in rows);recordings.extend(r['recording_id'] for r in rows)
        origins.extend(r['forecast_origin_sec'] for r in rows);forcing.extend(r['future_forcing'] for r in rows)
    for row in dataset.evaluation_windows():
        rows.append(row)
        if len(rows)==batch_size:consume(rows);rows=[]
    if rows:consume(rows)
    if not by_person:raise ValueError('No evaluation participants')
    atomic_npz(output/'predictions.npz',neural_mean=np.concatenate(pred),neural_scale=np.concatenate(sd),
               observed_neural=np.concatenate(ys),neural_target_mask=np.concatenate(masks),
               supplied_forcing_mg=np.asarray(forcing)*10.,participant_id=np.asarray(people),
               recording_id=np.asarray(recordings),forecast_origin_sec=np.asarray(origins),observation_cutoff_sec=np.asarray(origins))
    scores={p:float(np.mean(v)) for p,v in by_person.items()}
    cov={p:float(np.mean(v)) for p,v in coverage.items()}
    result={'neural_crps':float(np.mean(list(scores.values()))),'coverage90':float(np.mean(list(cov.values()))),
            'participant_scores':scores,'participant_coverage':cov,'participants':len(scores),
            'windows':len(origins),'primary_target':'future_neural_only','task':dataset.task,
            'created_utc':utcnow(),'prediction_file_sha256':file_hash(output/'predictions.npz')}
    atomic_json(output/'scores.json',result)
    return result
