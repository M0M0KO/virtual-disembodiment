"""Paired counterfactual forecasts through the actual fitted dynamics modules.

No fitting occurs here. Latent and emission innovations are keyed by the frozen
model, seed, participant, recording and origin, not by condition or batch size.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
import torch
from .suite_models import PriorDynamics,FeedForwardAlternative
from .suite_training import encode_batch
from .training import repeat_encoded


@dataclass(frozen=True)
class Condition:
    name:str
    ascending_gain:float=1.
    descending_gain:float=1.
    remove_body_measurements:bool=False
    message_delay:int=0
    restore:bool=False
    history_surrogate:str|None=None
    surrogate_seed:int=0
    washout_history_sec:int=0

    def validate(self):
        for value in (self.ascending_gain,self.descending_gain):
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Gate gains must be finite and within [0,1]')
        for name,value in [('message_delay',self.message_delay),('washout_history_sec',self.washout_history_sec)]:
            if isinstance(value,bool) or not isinstance(value,int) or value<0:raise ValueError(name+' must be a nonnegative integer')
        if self.history_surrogate not in (None,'replay','permuted','reversed'):
            raise ValueError('Only already observed history may supply the frozen surrogate')
        if self.restore and (self.remove_body_measurements or self.message_delay or self.history_surrogate or
                             self.washout_history_sec or self.ascending_gain!=1 or self.descending_gain!=1):
            raise ValueError('Restoration means original state and interfaces, not a mixed lesion condition')


def basic_conditions(gains=(1.,.75,.5,.25,0.),delays=(1,5,15)):
    if len(gains)!=len(set(gains)) or 1. not in gains:raise ValueError('One unique intact gain is required')
    output=[Condition('intact'),Condition('measurement-removal',remove_body_measurements=True)]
    for gain in gains:
        if gain==1:continue
        output.extend([Condition('ascending-'+str(gain),ascending_gain=gain),
                       Condition('descending-'+str(gain),descending_gain=gain),
                       Condition('both-'+str(gain),ascending_gain=gain,descending_gain=gain)])
    output.extend(Condition('causal-delay-'+str(d),message_delay=d) for d in delays)
    output.append(Condition('restore-original',restore=True))
    for condition in output:condition.validate()
    return output


def _seed(model_hash,seed,row,stream):
    identity=[model_hash,int(seed),str(row['participant_id']),str(row['recording_id']),int(row['forecast_origin_sec']),stream]
    return int.from_bytes(hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).digest()[:16],'little')


def noise_bank(model,rows,model_hash,seed,samples=16,steps=15):
    if not isinstance(samples,int) or samples<2 or not isinstance(steps,int) or steps<1:
        raise ValueError('Paired trajectory scoring requires at least two samples and a positive horizon')
    if not model_hash:raise ValueError('A frozen model identity is required for innovation pairing')
    emission=np.stack([np.random.default_rng(_seed(model_hash,seed,row,'emission')).standard_normal(
                        (samples,steps,model.neural_dim),dtype=np.float32) for row in rows])
    latent=None
    if isinstance(model,PriorDynamics):
        components=3 if model.has_peripheral else 2
        latent=np.stack([np.random.default_rng(_seed(model_hash,seed,row,'latent')).standard_normal(
                          (samples,steps,components,model.dimension),dtype=np.float32) for row in rows])
    elif not isinstance(model,FeedForwardAlternative):
        raise TypeError('Use the verified fitted dynamics or nonrecursive comparator adapter')
    return {'latent':latent,'emission':emission,'model_hash':model_hash,'seed':seed,'samples':samples,'steps':steps,
            'window_keys':[(str(r['participant_id']),str(r['recording_id']),int(r['forecast_origin_sec'])) for r in rows]}


def masked_history(batch,last_seconds):
    if last_seconds<0 or last_seconds>batch['history_body'].shape[1]:raise ValueError('Washout mask must stay inside observed history')
    if not last_seconds:return batch
    output=dict(batch)
    for key in ('history_body','history_body_mask','history_prior_body','history_prior_body_mask'):
        if key in output:
            output[key]=output[key].clone();output[key][:,-last_seconds:]=0
    return output


class HistoryReplay:
    """An open-loop past-history replay, not a fitted autonomous rescue model."""
    def __init__(self,sequence,samples,steps,mode,seed):
        if sequence.shape[1]<steps:raise ValueError('There is not enough completed history for this replay span')
        values=sequence[:,-steps:].detach().clone()
        self.original=values.clone();self.mode=mode
        if mode=='reversed':values=values.flip(1)
        elif mode=='permuted':
            order=np.random.default_rng(seed).permutation(steps)
            if np.array_equal(order,np.arange(steps)):raise ValueError('Identity permutation is not a timing control')
            values=values[:,torch.as_tensor(order,device=values.device)]
        elif mode!='replay':raise ValueError('Unknown history replay')
        self.values=values.repeat_interleave(samples,dim=0)
        self.future_observations_used=False

    def __call__(self,neural,peripheral,shared,forcing,step):
        # Deliberately ignores all evolving states. Only completed history
        # enters this control; it cannot adapt to current neural changes.
        return self.values[:,step]


def _encode(model,batch,condition):
    sequence=[];handle=None
    if condition.history_surrogate:
        if not isinstance(model,PriorDynamics) or not model.has_peripheral:
            raise ValueError('History message replay needs a real peripheral/internal-memory module')
        handle=model.peripheral.register_forward_hook(lambda module,inputs,outputs:sequence.append(outputs[0].detach().clone()))
    try:
        encoded=encode_batch(model,masked_history(batch,condition.washout_history_sec),
                             remove_body_measurements=condition.remove_body_measurements)
    finally:
        if handle is not None:handle.remove()
    return encoded,None if not sequence else sequence[-1]


@torch.no_grad()
def paired_prediction(model,batch,condition,bank):
    condition.validate()
    if model.training:raise ValueError('Counterfactual inference must use eval mode')
    device=batch['history_neural'].device;samples=bank['samples'];steps=bank['steps'];size=len(batch['history_neural'])
    if batch['future_forcing'].shape!=(size,steps,2) or bank['emission'].shape!=(size,samples,steps,model.neural_dim):
        raise ValueError('Paired bank and forecast batch have different shapes')
    encoded,history=_encode(model,batch,condition)
    metadata={'condition':condition.__dict__,'observation_cutoff':'unchanged observed history',
              'future_body_observations_used':False,'initial_embodied_state_retained':
                     not condition.remove_body_measurements and not condition.washout_history_sec,
              'rng_model_sha256':bank['model_hash'],'rng_seed':bank['seed'],
              'history_mask_is_not_free_rollout_washout':bool(condition.washout_history_sec)}
    if isinstance(model,FeedForwardAlternative):
        if condition.ascending_gain!=1 or condition.descending_gain!=1 or condition.message_delay or condition.history_surrogate or condition.restore:
            raise ValueError('Nonrecursive comparator has no message gates or dynamics restoration API')
        prediction=model.simulate(encoded,batch['future_forcing'])
        means=prediction['neural_mean'].float()[:,None].expand(-1,samples,-1,-1)
        scales=prediction['neural_scale'].float()[:,None].expand_as(means)
        metadata['latent_rollouts']=1
    else:
        repeated=repeat_encoded(encoded,samples)
        latent=bank['latent']
        if latent is None or latent.shape!=(size,samples,steps,3 if model.has_peripheral else 2,model.dimension):
            raise ValueError('Paired latent innovations do not match the fitted model')
        innovations=torch.as_tensor(latent.reshape(size*samples,steps,latent.shape[-2],model.dimension),device=device,dtype=encoded.brain.dtype)
        forcing=batch['future_forcing'].repeat_interleave(samples,0)
        if condition.restore:prediction=model.restore(repeated,forcing,innovations)
        else:
            options={'ascending_gain':condition.ascending_gain,'descending_gain':condition.descending_gain,
                     'message_delay':condition.message_delay,'innovations':innovations}
            if condition.history_surrogate:
                options['controller']=HistoryReplay(history,samples,steps,condition.history_surrogate,condition.surrogate_seed)
                options['controller_source']='autonomous_peripheral'
                metadata['message_source']='completed history replay; not a fitted autonomous generator'
                metadata['timing_control_reference']='history-replay'
            prediction=model.intervene(repeated,forcing,**options)
        means=prediction['neural_mean'].float().reshape(size,samples,steps,model.neural_dim)
        scales=prediction['neural_scale'].float().reshape_as(means)
        metadata['latent_rollouts']=samples
    mean=means.mean(1);variance=(scales.square()+means.square()).mean(1)-mean.square()
    scale=variance.clamp_min(1e-8).sqrt()
    epsilon=torch.as_tensor(bank['emission'],device=device,dtype=means.dtype)
    joint=means+scales*epsilon
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not torch.isfinite(joint).all():
        raise FloatingPointError('Nonfinite counterfactual predictive distribution')
    return {'neural_mean':mean,'neural_scale':scale,'joint_samples':joint,'metadata':metadata}


def joint_energy_score(samples,target,mask):
    """Unbiased finite-ensemble energy score for each masked joint trajectory."""
    x=np.asarray(samples,float);y=np.asarray(target,float);valid=np.asarray(mask,bool)
    if x.ndim!=4 or y.shape!=x.shape[:1]+x.shape[2:] or valid.shape!=y.shape or x.shape[1]<2:
        raise ValueError('Joint samples must be window x sample x time x feature')
    if not np.isfinite(x).all() or not np.isfinite(y[valid]).all():raise ValueError('Nonfinite trajectory values')
    result=[]
    for i in range(len(x)):
        use=valid[i].ravel()
        if not use.any():raise ValueError('Joint energy score needs observed coordinates')
        xx=x[i].reshape(x.shape[1],-1)[:,use];yy=y[i].ravel()[use]
        distance=np.linalg.norm(xx[:,None]-xx[None,:],axis=-1)
        result.append(np.linalg.norm(xx-yy,axis=-1).mean()-distance.sum()/(2*len(xx)*(len(xx)-1)))
    return np.asarray(result)
