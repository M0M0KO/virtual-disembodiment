"""Real-data probabilistic fitting, checkpoint/resume, and neural-only scoring."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .forecast_data import ForecastRecords
from .neural_dynamics import CoupledDynamics,gaussian_crps,probabilistic_training_loss
from .run_io import atomic_json,atomic_npz,file_hash,object_hash,read_json,require_same_or_new,utcnow


TENSOR_KEYS=('history_neural','history_neural_mask','history_body','history_body_mask',
             'history_forcing','future_forcing','target_neural','target_neural_mask',
             'loss_target_body','loss_target_body_mask')


def batches(iterator,size):
    iterator=iter(iterator)
    while True:
        rows=list(islice(iterator,size))
        if not rows:return
        yield rows


def device_batch(rows,device):
    return {key:torch.as_tensor(np.stack([r[key] for r in rows]),device=device) for key in TENSOR_KEYS}


def encode_batch(model,batch,dropout=0.):
    return model.encode_history(batch['history_neural'],batch['history_neural_mask'],
                                batch['history_body'],batch['history_body_mask'],
                                batch['history_forcing'],observation_dropout=dropout)


def repeat_cache(cache,samples):
    if cache is None:return None
    if isinstance(cache,torch.Tensor):return cache.repeat_interleave(samples,dim=1)  # GRU layers,batch,dimension
    return [(k.repeat_interleave(samples,dim=0),v.repeat_interleave(samples,dim=0)) for k,v in cache]


def repeat_encoded(e,samples):
    return replace(e,brain=e.brain.repeat_interleave(samples,dim=0),
                   peripheral=None if e.peripheral is None else e.peripheral.repeat_interleave(samples,dim=0),
                   shared=e.shared.repeat_interleave(samples,dim=0),
                   brain_cache=repeat_cache(e.brain_cache,samples),
                   peripheral_cache=repeat_cache(e.peripheral_cache,samples),
                   shared_cache=repeat_cache(e.shared_cache,samples),
                   reconstruction=e.reconstruction.repeat_interleave(samples,dim=0),
                   history_reconstruction_target=e.history_reconstruction_target.repeat_interleave(samples,dim=0),
                   history_reconstruction_mask=e.history_reconstruction_mask.repeat_interleave(samples,dim=0))


def sample_distribution(model,batch,samples=16,**intervention):
    e=repeat_encoded(encode_batch(model,batch),samples)
    forcing=batch['future_forcing'].repeat_interleave(samples,dim=0)
    prediction=model.simulate(e,forcing,**intervention)
    b=batch['history_neural'].shape[0]
    mean=prediction['neural_mean'].float().reshape(b,samples,-1,model.neural_dim)
    scale=prediction['neural_scale'].float().reshape_as(mean)
    mu=mean.mean(dim=1)
    variance=(scale.square()+mean.square()).mean(dim=1)-mu.square()
    return mu,variance.clamp_min(1e-8).sqrt()


def neural_window_scores(mean,scale,target,mask,horizon_weights=(.2,.3,.5)):
    """Future peripheral values do not occur anywhere in the primary score."""
    indices=[0,4,14]
    mean,scale,target,mask=[x[:,indices] for x in (mean,scale,target,mask)]
    mask=mask.float();weights=mean.new_tensor(horizon_weights,dtype=torch.float32)
    count=mask.sum(dim=-1)
    if (count==0).any():raise ValueError('A primary horizon has no eligible neural features')
    crps=(gaussian_crps(mean,scale,target)*mask).sum(dim=-1)/count
    coverage=(((target-mean).abs()<=1.6448536269514722*scale).float()*mask).sum(dim=-1)/count
    return (crps*weights).sum(dim=-1),(coverage*weights).sum(dim=-1)


def validation_groups(feature_manifest,output,seed):
    ids=sorted(r['participant_id'] for r in feature_manifest['records']
               if r['split']=='validation' and r.get('state')!='EXCLUDED_SOURCE_CLOCK')
    order=np.asarray(sorted(set(ids)),dtype=str);np.random.default_rng(seed+137).shuffle(order)
    count=len(order)//2
    if count<3:raise ValueError('Not enough validation participants for selection and fidelity audit')
    value={'selection':sorted(order[:count].tolist()),'fidelity_gate':sorted(order[count:].tolist()),
           'seed':seed+137,'source_feature_manifest_sha256':object_hash(feature_manifest),
           'outcomes_used_for_allocation':False}
    require_same_or_new(Path(output)/'validation-groups.json',value)
    return value


def evaluate(model,dataset,device,batch_size=8,samples=16,seed=4001,output=None,model_hash=None,
             horizon_weights=(.2,.3,.5),scale_factor=1.0):
    model.eval();by_person=defaultdict(list);cover=defaultdict(list)
    means=[];scales=[];targets=[];masks=[];origins=[];people=[];recordings=[];forcing=[]
    devices=[torch.cuda.current_device()] if device.type=='cuda' else []
    start=time.monotonic()
    with torch.no_grad(),torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:torch.cuda.manual_seed_all(seed)
        for rows in batches(dataset.evaluation_windows(),batch_size):
            batch=device_batch(rows,device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                mean,scale=sample_distribution(model,batch,samples)
            scale=scale*scale_factor
            score,cov=neural_window_scores(mean,scale,batch['target_neural'],batch['target_neural_mask'],horizon_weights)
            score,cov=score.cpu().numpy(),cov.cpu().numpy()
            for r,s,c in zip(rows,score,cov):
                by_person[r['participant_id']].append(float(s));cover[r['participant_id']].append(float(c))
            if output:
                means.append(mean.cpu().numpy());scales.append(scale.cpu().numpy())
                targets.append(batch['target_neural'].cpu().numpy());masks.append(batch['target_neural_mask'].cpu().numpy())
                forcing.append(batch['future_forcing'].cpu().numpy())
                origins.extend(r['forecast_origin_sec'] for r in rows)
                people.extend(r['participant_id'] for r in rows);recordings.extend(r['recording_id'] for r in rows)
    result={'neural_crps':float(np.mean([np.mean(v) for v in by_person.values()])),
            'coverage90':float(np.mean([np.mean(v) for v in cover.values()])),
            'participant_scores':{p:float(np.mean(v)) for p,v in by_person.items()},
            'participant_coverage':{p:float(np.mean(v)) for p,v in cover.items()},
            'participants':len(by_person),'windows':sum(len(v) for v in by_person.values()),
            'elapsed_sec':time.monotonic()-start,'generation_samples':samples,
            'distribution':'Gaussian moment match of learned emission variance and stochastic latent rollouts',
            'primary_target':'future_neural_only','task':dataset.task,
            'created_utc':utcnow(),'model_hash':model_hash}
    if not math.isfinite(result['neural_crps']):raise FloatingPointError('Nonfinite validation score')
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        atomic_npz(output/'predictions.npz',neural_mean=np.concatenate(means),neural_scale=np.concatenate(scales),
                   observed_neural=np.concatenate(targets),neural_target_mask=np.concatenate(masks),
                   supplied_forcing_mg=np.concatenate(forcing)*10.,participant_id=np.asarray(people),
                   recording_id=np.asarray(recordings),forecast_origin_sec=np.asarray(origins),
                   observation_cutoff_sec=np.asarray(origins))
        result['prediction_file_sha256']=file_hash(output/'predictions.npz')
        atomic_json(output/'scores.json',result)
    return result


def atomic_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.writing')
    with temp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def learning_rate_multiplier(step,total,warmup_fraction):
    warm=max(1,int(total*warmup_fraction))
    if step<warm:return (step+1)/warm
    return .5*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm))))


def fit_model(feature_root,config,spec,output,device='cuda'):
    """Complete fixed-budget real-data fit, not a shortened smoke benchmark."""
    feature_root,output=Path(feature_root),Path(output);output.mkdir(parents=True,exist_ok=True)
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested GPU is unavailable')
    if spec.get('initialization','scratch')!='scratch':
        raise ValueError('Pretrained initialization needs its verified encoder adapter; never substitute scratch silently')
    seed=int(spec['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    manifest=read_json(feature_root/'feature-manifest.json')
    groups=validation_groups(manifest,feature_root.parent,config['split']['seed'])
    data_args={'context':spec.get('context_sec',60),'pipeline':spec.get('pipeline','conservative'),
               'task':spec.get('task','forced_simulation')}
    training=ForecastRecords(feature_root/'feature-manifest.json',feature_root/'feature-scaling.json',
                             'development',stride=1,**data_args)
    validation=ForecastRecords(feature_root/'feature-manifest.json',feature_root/'feature-scaling.json',
                               'validation',stride=config['forecast']['evaluation_window_stride_sec'],
                               participant_subset=groups['selection'],**data_args)
    model=CoupledDynamics(dimension=spec.get('dimension',256),layers=spec.get('layers',8),
                          backbone=spec.get('backbone','causal_transformer'),kind=spec.get('kind','coupled')).to(device)
    inventory=model.parameter_inventory()
    low,high=config['models']['pilot_parameter_range'] if spec.get('scale','pilot')=='pilot' else config['models']['main_parameter_range']
    if not low<=inventory['trainable']<=high:
        raise ValueError('Actual trainable parameter count is outside the registered scale range')
    frozen={'model_spec':spec,'training':config['training'],'forecast':config['forecast'],
            'model_inventory':inventory,'feature_manifest_sha256':object_hash(manifest),
            'validation_groups_sha256':object_hash(groups),'training_cohort':training.cohort_summary(),
            'selection_cohort':validation.cohort_summary(),
            'implementation_sha256':{Path(__file__).name:file_hash(__file__),
                   'neural_dynamics.py':file_hash(Path(__file__).with_name('neural_dynamics.py')),
                   'forecast_data.py':file_hash(Path(__file__).with_name('forecast_data.py'))}}
    require_same_or_new(output/'fit-spec.json',frozen);spec_hash=object_hash(frozen)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    settings=config['training'];logical=settings['participant_first_batch_size']
    micro=min(logical,spec.get('micro_batch_size',8 if spec.get('scale','pilot')=='pilot' else 2))
    epoch=0;skip_batches=0;step=0;best=float('inf');patience=0
    max_epochs=settings['max_epochs'];per_person=settings['windows_per_participant_per_epoch']
    batches_per_epoch=math.ceil(len(training.participants)*per_person/logical)
    total_steps=batches_per_epoch*max_epochs
    weights=tuple(spec.get('loss_weights',settings['loss_weights'][0]))
    history=[];last=output/'last.pt'
    if last.exists():
        # Only self-produced, fingerprint-matched checkpoints are loaded here.
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['spec_sha256']!=spec_hash:raise ValueError('Checkpoint configuration/source mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        epoch=saved['next_epoch'];skip_batches=saved['skip_batches'];step=saved['global_step']
        best=saved['best_crps'];patience=saved['patience'];micro=saved['micro_batch_size'];history=saved['history']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def checkpoint(next_epoch,skip):
        atomic_checkpoint(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'next_epoch':next_epoch,'skip_batches':skip,'global_step':step,'best_crps':best,'patience':patience,
            'micro_batch_size':micro,'history':history,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'spec_sha256':spec_hash})
    if (output/'completion.json').exists():
        result=read_json(output/'completion.json')
        if result['spec_sha256']!=spec_hash or file_hash(output/'best.pt')!=result['best_checkpoint_sha256']:
            raise ValueError('Completed model snapshot differs from its declared checkpoint')
        return result
    begin=time.monotonic()
    for e in range(epoch,max_epochs):
        model.train();epoch_start=time.monotonic();loss_sum=0.;seen=0
        for batch_index,rows in enumerate(batches(training.training_epoch(seed+e*1009,per_person),logical)):
            if e==epoch and batch_index<skip_batches:continue
            for group in optimizer.param_groups:
                group['lr']=spec['learning_rate']*learning_rate_multiplier(step,total_steps,settings['warmup_fraction'])
            retry=True
            while retry:
                retry=False;optimizer.zero_grad(set_to_none=True);batch_loss=0.
                saved_rng=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []
                try:
                    for a in range(0,len(rows),micro):
                        small=rows[a:a+micro];batch=device_batch(small,device)
                        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                            encoded=encode_batch(model,batch,dropout=.1)
                            prediction=model.simulate(encoded,batch['future_forcing'])
                            loss,_=probabilistic_training_loss(prediction,batch,weights)
                        (loss*(len(small)/len(rows))).backward();batch_loss+=float(loss.detach())*len(small)/len(rows)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'])
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if micro==1:raise
                    batch=None;encoded=None;prediction=None;loss=None
                    micro=max(1,micro//2);torch.cuda.empty_cache()
                    torch.set_rng_state(saved_rng);torch.cuda.set_rng_state_all(saved_cuda);retry=True
            # An optimizer OOM is fatal, never retried after a partial update.
            optimizer.step()
            step+=1;loss_sum+=batch_loss;seen+=1
            if step%25==0:
                atomic_json(output/'status.json',{'state':'TRAINING','pid':os.getpid(),'updated_utc':utcnow(),
                            'epoch':e+1,'max_epochs':max_epochs,'batch':batch_index+1,'batches_per_epoch':batches_per_epoch,
                            'global_step':step,'best_validation_neural_crps':None if not math.isfinite(best) else best,
                            'micro_batch_size':micro,'effective_batch_size':logical,
                            'seconds_this_epoch':time.monotonic()-epoch_start,
                            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None})
            if step%100==0:checkpoint(e,batch_index+1)
        metric=evaluate(model,validation,device,seed=seed+90000,horizon_weights=config['forecast']['horizon_weights'])
        row={'epoch':e+1,'neural_crps':metric['neural_crps'],'coverage90':metric['coverage90'],
             'training_loss':loss_sum/max(1,seen),'elapsed_sec':time.monotonic()-epoch_start,'global_step':step}
        history.append(row)
        if metric['neural_crps']<best-1e-6:
            best=metric['neural_crps'];patience=0
            atomic_checkpoint(output/'best.pt',{'model':model.state_dict(),'model_spec':spec,'spec_sha256':spec_hash,
                                              'selected_epoch':e+1,'selection_neural_crps':best})
        else:patience+=1
        checkpoint(e+1,0);skip_batches=0
        atomic_json(output/'learning-curve.json',history)
        atomic_json(output/'status.json',{'state':'EPOCH_COMPLETE','pid':os.getpid(),'updated_utc':utcnow(),
                    **row,'best_selection_neural_crps':best,'patience_used':patience,'micro_batch_size':micro})
        print(json.dumps({'event':'epoch_complete',**row}),flush=True)
        if patience>=settings['patience']:break
    if not (output/'best.pt').exists():raise RuntimeError('No finite eligible checkpoint was produced')
    best_model=torch.load(output/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(best_model['model']);model_hash=file_hash(output/'best.pt')
    scores=evaluate(model,validation,device,seed=seed+90000,output=output/'selection-predictions',
                    model_hash=model_hash,horizon_weights=config['forecast']['horizon_weights'])
    result={'state':'COMPLETE','best_checkpoint_sha256':model_hash,'best_checkpoint':str(output/'best.pt'),
            'selected_epoch':best_model['selected_epoch'],'epochs_completed':len(history),'global_steps':step,
            'model_inventory':inventory,'selection_scores':scores,'spec_sha256':spec_hash,
            'protected_test_opened':False,'fidelity_gate_participants_used_for_selection':False,
            'finished_utc':utcnow(),'elapsed_sec_this_invocation':time.monotonic()-begin}
    atomic_json(output/'completion.json',result);atomic_json(output/'status.json',result)
    return result
