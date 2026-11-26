"""Participant-grouped histories and targets, with protected-test access checks."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np

from .run_io import object_hash, read_json


def eligible_origins(neural_mask, ecg_fraction, target_mask, context, horizon=15, stride=1,
                     min_fraction=0.8, common_context=None):
    neural_mask=np.asarray(neural_mask,bool);target_mask=np.asarray(target_mask,bool)
    ecg_fraction=np.asarray(ecg_fraction,float)
    if neural_mask.shape!=target_mask.shape or neural_mask.ndim!=2 or neural_mask.shape[1]!=16:
        raise ValueError('Primary neural mask must contain 2 channels x 8 features')
    if context<1 or horizon<1 or stride<1 or len(ecg_fraction)!=len(neural_mask):
        raise ValueError('Invalid forecast window dimensions')
    common=max(context,common_context or context)
    per_channel=np.column_stack([neural_mask[:,:8].all(axis=1),neural_mask[:,8:].all(axis=1)])
    sums=np.vstack([np.zeros((1,2)),np.cumsum(per_channel,axis=0)])
    ecg_sum=np.r_[0.,np.cumsum(ecg_fraction)]
    future_bad=np.r_[0,np.cumsum(~target_mask.all(axis=1))]
    origins=np.arange(common,len(neural_mask)-horizon+1,stride,dtype=np.int64)
    good=((sums[origins]-sums[origins-context])/context>=min_fraction).all(axis=1)
    good&=(ecg_sum[origins]-ecg_sum[origins-context])/context>=min_fraction
    good&=(future_bad[origins+horizon]-future_bad[origins])==0
    return origins[good]


class ForecastRecords:
    """One in-memory cache of small feature arrays, not raw waveform payloads.

    This object has no clinical label fields. Future physiology is returned
    under a loss-target key, never concatenated with a history or forcing input.
    """
    def __init__(self, manifest_path, scaling_path, split, context=60,
                 pipeline='conservative', task='forced_simulation', stride=1,
                 common_context=None, protected_authorization=None, participant_subset=None):
        if split not in ('development','validation','test'):
            raise ValueError('Unknown DOSE-I partition')
        if pipeline not in ('conservative','cardiac_robust') or task not in ('forced_simulation','autonomous_forecast'):
            raise ValueError('Unknown feature pipeline/task')
        self.manifest=read_json(manifest_path);self.scaling=read_json(scaling_path)
        if split=='test':
            if protected_authorization is None:
                raise PermissionError('Protected test stays closed until validation freeze')
            auth=read_json(protected_authorization)
            if not auth.get('validation_fidelity_passed') or auth.get('feature_manifest_sha256')!=object_hash(self.manifest):
                raise PermissionError('Protected authorization is invalid or refers to different features')
        self.records=[];self.by_participant=defaultdict(list)
        self.context=context;self.pipeline=pipeline;self.task=task
        self.common_context=max(context,common_context or context)
        keys=['neural_'+pipeline,'neural_mask_'+pipeline,'target_mask_'+pipeline,'body','body_mask',
              'forcing_mg','token_end_sec','ecg_observed_fraction']
        for r in self.manifest['records']:
            if r['split']!=split:
                continue
            if r.get('state')=='EXCLUDED_SOURCE_CLOCK':
                continue
            if participant_subset is not None and r['participant_id'] not in participant_subset:
                continue
            with np.load(r['feature_file'],allow_pickle=False) as z:
                a={k:z[k] for k in keys}
            origins=eligible_origins(a['neural_mask_'+pipeline],a['ecg_observed_fraction'],
                                    a['target_mask_'+pipeline],context,15,stride,
                                    common_context=common_context)
            idx=len(self.records)
            self.records.append({'metadata':r,'arrays':a,'origins':origins})
            if origins.size:self.by_participant[r['participant_id']].append(idx)
        self.participants=sorted(self.by_participant)
        self.window_count=sum(len(r['origins']) for r in self.records)
        if not self.window_count:
            raise ValueError('No eligible forecast windows in requested cohort/context')

    def standardize(self,x,name):
        stats=self.scaling[name]
        mean=np.asarray(stats['mean'],np.float32);scale=np.asarray(stats['scale'],np.float32)
        active=np.asarray(stats['active'],bool)
        mask=np.isfinite(x)&active
        y=np.where(mask,(x-mean)/scale,0).astype(np.float32)
        return y,mask

    def window(self,record_index,origin):
        r=self.records[record_index];a=r['arrays'];i=int(origin);c=self.context
        position=np.searchsorted(r['origins'],i)
        if position>=len(r['origins']) or r['origins'][position]!=i:
            raise ValueError('Origin was not declared eligible')
        neural,neural_mask=self.standardize(a['neural_'+self.pipeline][i-c:i],'neural_'+self.pipeline)
        body,body_mask=self.standardize(a['body'][i-c:i],'body')
        yn,ym=self.standardize(a['neural_'+self.pipeline][i:i+15],'neural_'+self.pipeline)
        yp,pm=self.standardize(a['body'][i:i+15],'body')
        future_forcing=a['forcing_mg'][i:i+15]/10.0
        if self.task=='autonomous_forecast':future_forcing=np.zeros_like(future_forcing)
        return {'history_neural':neural,'history_neural_mask':neural_mask,
                'history_body':body,'history_body_mask':body_mask,
                'history_forcing':a['forcing_mg'][i-c:i]/10.0,
                'future_forcing':future_forcing.astype(np.float32),
                'target_neural':yn,'target_neural_mask':ym&a['target_mask_'+self.pipeline][i:i+15],
                'loss_target_body':yp,'loss_target_body_mask':pm,
                'participant_id':r['metadata']['participant_id'],
                'recording_id':r['metadata']['recording_id'],
                'forecast_origin_sec':i,'observation_cutoff_sec':i,
                'task':self.task,'context_sec':c}

    def training_epoch(self,seed,windows_per_participant=64):
        """Uniform participants, then uniform eligible windows of that person."""
        rng=np.random.default_rng(seed)
        people=np.repeat(np.arange(len(self.participants)),windows_per_participant)
        rng.shuffle(people)
        for person in people:
            pid=self.participants[int(person)]
            ri=int(rng.choice(self.by_participant[pid]));r=self.records[ri]
            yield self.window(ri,int(rng.choice(r['origins'])))

    def evaluation_windows(self):
        for ri,r in enumerate(self.records):
            for origin in r['origins']:
                yield self.window(ri,int(origin))

    def cohort_summary(self):
        return {'participants':len(self.participants),'recordings':sum(bool(len(r['origins'])) for r in self.records),
                'windows':self.window_count,'context_sec':self.context,'task':self.task,
                'excluded_recordings':[r['metadata']['recording_id'] for r in self.records if not len(r['origins'])],
                'common_context_sec':self.common_context}
