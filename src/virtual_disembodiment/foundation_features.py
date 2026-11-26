"""Causal availability of frozen observation features from completed raw blocks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from .real_signals import causal_filter,flatline_causal_mask,eeg_ecg_regressors
from .run_io import atomic_json,atomic_npz,file_hash,read_json,utcnow


def observation_streams(a,config,beta):
    fs=config['eeg_grid_hz'];ef=config['ecg_grid_hz']
    eeg=a['raw_eeg_uv'];valid=a['raw_eeg_available'];ecg=a['raw_ecg_mv'];ev=a['raw_ecg_available']
    conservative=[];masks=[]
    for ch in range(2):
        values,mask=causal_filter(eeg[:,ch],valid[:,ch],fs,config['eeg_bandpass_hz'],config['filter_order'],config['gap_filter_warmup_sec'])
        mask&=(np.abs(eeg[:,ch])<config['eeg_saturation_uv'])
        mask&=flatline_causal_mask(eeg[:,ch],fs,config['causal_flat_variance_uv2'],config['causal_flat_detection_sec'])
        conservative.append(values);masks.append(mask)
    conservative=np.column_stack(conservative);neural_mask=np.column_stack(masks)
    reg,reg_valid=eeg_ecg_regressors(ecg,ev,len(eeg),fs,ef,config['artifact_regression_lags_sec'])
    contribution=reg@beta;robust=conservative.copy()
    for ch in range(2):
        filtered,_=causal_filter(contribution[:,ch],reg_valid,fs,config['eeg_bandpass_hz'],config['filter_order'],config['gap_filter_warmup_sec'])
        robust[:,ch]-=filtered
    body,body_mask=causal_filter(ecg,ev,ef,[.5,45.],config['filter_order'],config['gap_filter_warmup_sec'])
    return {'conservative':(conservative,neural_mask),
            'cardiac_robust':(robust,neural_mask&np.isfinite(robust)),
            'ecg':(body[:,None],body_mask[:,None])}


def completed_blocks(signal,valid,fs,target_fs,standardize=False):
    """Resampling and normalization see only each already completed five seconds."""
    signal=np.asarray(signal,float);valid=np.asarray(valid,bool)&np.isfinite(signal)
    n=len(signal)//(5*fs)
    x=signal[:n*5*fs].reshape(n,5*fs,-1).transpose(0,2,1)
    mask=valid[:n*5*fs].reshape(n,5*fs,-1).transpose(0,2,1)
    quality=mask.mean(axis=-1)>=.8
    count=mask.sum(axis=-1,keepdims=True).clip(min=1)
    mean=np.where(mask,x,0).sum(axis=-1,keepdims=True)/count
    variance=np.where(mask,(x-mean)**2,0).sum(axis=-1,keepdims=True)/count
    index=np.maximum.accumulate(np.where(mask,np.arange(x.shape[-1])[None,None,:],-1),axis=-1)
    filled=np.take_along_axis(x,np.maximum(index,0),axis=-1)
    filled=np.where(index>=0,filled,0.)
    if standardize:filled=(filled-mean)/np.sqrt(variance).clip(min=1e-6)
    filled=np.where(quality[:,:,None],filled,0.)
    g=np.gcd(fs,target_fs)
    out=resample_poly(filled,target_fs//g,fs//g,axis=-1)
    if out.shape[-1]!=5*target_fs or not np.isfinite(out).all():raise ValueError('Malformed completed-block resampling')
    return out.astype(np.float32),quality


def available_timeline(block_features,block_valid,nseconds):
    d=block_features.shape[-1];value=np.zeros((nseconds,d),np.float32);valid=np.zeros(nseconds,bool)
    starts=np.full(nseconds,-1,np.int64);ends=starts.copy()
    for b,(feature,good) in enumerate(zip(block_features,block_valid)):
        # token index4 ends at second5, the first point where the block is available.
        lo=(b+1)*5-1;hi=min(nseconds,(b+2)*5-1)
        value[lo:hi]=feature;valid[lo:hi]=good;starts[lo:hi]=b*5;ends[lo:hi]=(b+1)*5
    return value,valid,starts,ends


def history_embedding(cache,key,begin,end):
    if not 0<=begin<end<=len(cache['token_end_sec']):raise ValueError('Invalid foundation history interval')
    values=cache[key][begin:end].copy();valid=cache[key+'_valid'][begin:end].copy()
    source_begin=cache['source_start_sec'][begin:end];source_end=cache['source_end_sec'][begin:end]
    valid&=(source_begin>=begin)&(source_end<=cache['token_end_sec'][begin:end])&(source_end<=end)
    values[~valid]=0.
    return values,valid


def encode_record_arrays(a,proc,beta,encoders,montage,batch_size=32):
    streams=observation_streams(a,proc,beta);nsec=len(a['token_end_sec'])
    output={'token_end_sec':a['token_end_sec']};mapping=None
    def save(name,features,valid):
        value,good,start,end=available_timeline(features,valid,nsec)
        output[name]=value;output[name+'_valid']=good
        if 'source_start_sec' not in output:output.update(source_start_sec=start,source_end_sec=end)
        elif not np.array_equal(start,output['source_start_sec']) or not np.array_equal(end,output['source_end_sec']):
            raise ValueError('Foundation branches have different availability clocks')
    ecg,ecg_mask=streams['ecg']
    blocks,quality=completed_blocks(ecg,ecg_mask,proc['ecg_grid_hz'],128,standardize=True)
    features=[];valid=[]
    for i in range(0,len(blocks),batch_size):
        f,v=encoders.sleep_features(blocks[i:i+batch_size],quality[i:i+batch_size]);features.append(f);valid.append(v)
    save('body_sleepfm',np.concatenate(features),np.concatenate(valid))
    for pipeline in ('conservative','cardiac_robust'):
        eeg,mask=streams[pipeline]
        sb,sq=completed_blocks(eeg,mask,proc['eeg_grid_hz'],128,standardize=True)
        lb,lq=completed_blocks(eeg,mask,proc['eeg_grid_hz'],200)
        # The publisher's EEG-only encoder takes microvolt values divided by100.
        lb=(lb/100.).reshape(len(lb),2,5,200)
        sf=[];sv=[];lf=[];lv=[]
        for i in range(0,len(sb),batch_size):
            f,v=encoders.sleep_features(sb[i:i+batch_size],sq[i:i+batch_size]);sf.append(f);sv.append(v)
            f,v,mapping=encoders.labram_features(lb[i:i+batch_size],lq[i:i+batch_size],montage);lf.append(f);lv.append(v)
        save('neural_sleepfm_'+pipeline,np.concatenate(sf),np.concatenate(sv))
        save('neural_labram_'+pipeline,np.concatenate(lf),np.concatenate(lv))
    return output,{'labram_position_mapping':mapping,'block_sec':5,
                   'sleepfm_normalization':'within completed block only; not the publisher demo full-record z-score',
                   'source_eeg_band_hz':proc['eeg_bandpass_hz'],'ecg_band_hz':[.5,45.],
                   'upsampling_adds_no_unobserved_frequency_content':True}


_ENCODERS=None
def initialize_worker(project):
    global _ENCODERS
    import torch
    from .foundation_encoders import FrozenObservationEncoders
    torch.set_num_threads(1);_ENCODERS=FrozenObservationEncoders(project,device='cpu')


def process_foundation_record(task):
    record,raw_record,config,beta,output,spec_hash=task;output=Path(output)
    receipt=output/(record['recording_id']+'.json');path=output/(record['recording_id']+'.npz')
    if receipt.exists():
        old=read_json(receipt)
        if old['spec_sha256']!=spec_hash or old['source_features_sha256']!=record['output_sha256']:
            raise ValueError('Foundation feature inputs or implementation changed')
        if file_hash(path)!=old['output_sha256']:raise ValueError('Foundation feature checkpoint is corrupt')
        return old
    if file_hash(record['feature_file'])!=record['output_sha256']:raise ValueError('Base feature snapshot changed')
    keys=('raw_eeg_uv','raw_eeg_available','raw_ecg_mv','raw_ecg_available','token_end_sec')
    with np.load(record['feature_file'],allow_pickle=False) as z:a={k:z[k] for k in keys}
    if len(a['token_end_sec'])<5:raise ValueError('A source record has no complete observation block')
    features,notes=encode_record_arrays(a,config['preprocessing'],np.asarray(beta),_ENCODERS,raw_record['eeg_montage'])
    atomic_npz(path,**features)
    value={'state':'COMPLETE','recording_id':record['recording_id'],'participant_id':record['participant_id'],
           'split':record['split'],'feature_file':str(path),'source_features_sha256':record['output_sha256'],
           'output_sha256':file_hash(path),'spec_sha256':spec_hash,'notes':notes,
           'valid_tokens':{k:int(v.sum()) for k,v in features.items() if k.endswith('_valid')},
           'clinical_labels_loaded':False,'protected_test_outcomes_evaluated':False,'created_utc':utcnow()}
    atomic_json(receipt,value);return value
