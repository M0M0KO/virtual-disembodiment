"""Optional frozen observation priors and strictly baseline-only metadata inputs."""
from __future__ import annotations
import csv
import io
from pathlib import Path
import zipfile
import numpy as np
from .forecast_data import ForecastRecords,eligible_origins
from .foundation_features import history_embedding
from .run_io import file_hash,object_hash,read_json,require_same_or_new

NUMERIC=('age','height','weight','bmi')
FLAGS=('drugs_bblocker','drugs_opioids','drugs_neuroleptics','drugs_benzodiazepines','drugs_antiepileptics',
       'conditions_OHS','conditions_SAS','conditions_oHE','conditions_CHF')


def common_history_origins(arrays,pipeline,context,common_context,stride):
    """Use the exact intersection of history-quality eligibility, not just a later clock start."""
    contexts=sorted(set([context,common_context]+[c for c in (60,180,600) if c<=common_context]))
    origins=None
    for c in contexts:
        current=eligible_origins(arrays['neural_mask_'+pipeline],arrays['ecg_observed_fraction'],
                                 arrays['target_mask_'+pipeline],c,15,stride,common_context=common_context)
        origins=current if origins is None else np.intersect1d(origins,current,assume_unique=True)
    return origins


def metadata_encoding(project,feature_manifest,output):
    archive=Path(project)/'data/raw/dose_i/18483292/static.zip'
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('DOSE-I_static_data.csv'))
        rows=list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
    source={}
    for r in rows:
        pid='DOSE-I-'+str(int(r['ID']))
        if pid in source:raise ValueError('Duplicate static metadata study ID')
        source[pid]=r
    people=sorted(set(r['participant_id'] for r in feature_manifest['records'] if r['state']!='EXCLUDED_SOURCE_CLOCK'))
    if not set(people)<=set(source):raise ValueError('Static metadata is missing study IDs')
    values=np.full((len(people),len(NUMERIC)),np.nan)
    for i,pid in enumerate(people):
        for j,name in enumerate(NUMERIC):
            try:values[i,j]=float(source[pid][name])
            except (ValueError,KeyError):pass
    train={r['participant_id'] for r in feature_manifest['records'] if r['split']=='development' and r['state']!='EXCLUDED_SOURCE_CLOCK'}
    train_values=values[[p in train for p in people]]
    mean=np.nanmean(train_values,axis=0);scale=np.nanstd(train_values,axis=0).clip(min=1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():raise ValueError('No development statistics for a static field')
    encoded={}
    for i,pid in enumerate(people):
        r=source[pid];numeric=values[i];present=np.isfinite(numeric)
        vector=list(np.where(present,(numeric-mean)/scale,0.))+list(present.astype(float))
        sex=r['sex'].strip().lower();vector.extend([float(sex=='female'),float(sex=='male'),float(sex not in ('female','male'))])
        try:asa=int(float(r['ASA']))
        except ValueError:asa=0
        vector.extend([float(asa==n) for n in range(1,6)]+[float(asa not in range(1,6))])
        for field in FLAGS:
            value=r[field].strip().lower();vector.extend([float(value=='true'),float(value in ('true','false'))])
        encoded[pid]=vector
    frozen={'numeric_columns':list(NUMERIC),'flag_columns':list(FLAGS),'mean':mean.tolist(),'scale':scale.tolist(),
            'fit_partition':'development','fit_participants':sorted(train),'archive_sha256':file_hash(archive),
            'feature_manifest_sha256':object_hash(feature_manifest),'dimension':len(next(iter(encoded.values()))),
            'participant_id_is_model_input':False,'clinical_state_labels_are_model_inputs':False}
    require_same_or_new(Path(output)/'metadata-encoding.json',frozen)
    return encoded,frozen


class SuiteRecords(ForecastRecords):
    def __init__(self,*args,observation_prior='scratch',foundation_manifest=None,metadata=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.common_eligibility=kwargs.get('common_context') is not None
        if self.common_eligibility:
            self.by_participant.clear()
            for i,record in enumerate(self.records):
                record['origins']=common_history_origins(record['arrays'],self.pipeline,self.context,
                                                        self.common_context,kwargs.get('stride',1))
                if record['origins'].size:self.by_participant[record['metadata']['participant_id']].append(i)
            self.participants=sorted(self.by_participant)
            self.window_count=sum(len(record['origins']) for record in self.records)
            if not self.window_count:raise ValueError('No windows in the exact common history-eligibility intersection')
        if observation_prior not in ('scratch','sleepfm_frozen','labram_frozen'):raise ValueError('Unknown observation prior')
        self.observation_prior=observation_prior;self.metadata=metadata;self.prior_cache={}
        if observation_prior!='scratch':
            manifest=read_json(foundation_manifest)
            if manifest['state']!='COMPLETE':raise ValueError('Frozen observation extraction is unfinished')
            candidates={r['recording_id']:r for r in manifest['records']}
            for i,record in enumerate(self.records):
                source=record['metadata'];r=candidates[source['recording_id']]
                if r['source_features_sha256']!=source['output_sha256'] or file_hash(r['feature_file'])!=r['output_sha256']:
                    raise ValueError('Observation prior and base features have different provenance')
                name='neural_'+('sleepfm' if observation_prior=='sleepfm_frozen' else 'labram')+'_'+self.pipeline
                keys=[name,name+'_valid','token_end_sec','source_start_sec','source_end_sec']
                if observation_prior=='sleepfm_frozen':keys+=['body_sleepfm','body_sleepfm_valid']
                with np.load(r['feature_file'],allow_pickle=False) as z:self.prior_cache[i]={k:z[k] for k in keys}
                if not np.array_equal(self.prior_cache[i]['token_end_sec'],record['arrays']['token_end_sec']):
                    raise ValueError('Observation-prior and physical-feature clocks differ')

    def window(self,record_index,origin):
        row=super().window(record_index,origin);begin=int(origin)-self.context;end=int(origin)
        row['history_time_sec']=np.arange(begin+1,end+1,dtype=np.float32)[:,None]
        if self.metadata is not None:row['static_metadata']=np.asarray(self.metadata[row['participant_id']],np.float32)
        if self.observation_prior!='scratch':
            cache=self.prior_cache[record_index]
            name='neural_'+('sleepfm' if self.observation_prior=='sleepfm_frozen' else 'labram')+'_'+self.pipeline
            value,valid=history_embedding(cache,name,begin,end)
            row.update(history_prior_neural=value,history_prior_neural_mask=valid[:,None])
            if self.observation_prior=='sleepfm_frozen':
                value,valid=history_embedding(cache,'body_sleepfm',begin,end)
                row.update(history_prior_body=value,history_prior_body_mask=valid[:,None])
        return row

    def cohort_summary(self):
        return {**super().cohort_summary(),'common_eligibility_policy':
                'intersection of all registered histories up to common_context' if self.common_eligibility else 'own history'}
