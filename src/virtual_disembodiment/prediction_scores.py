"""Independent scoring of frozen distributions; no model loading or fitting."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.special import erf
from .run_io import atomic_json,atomic_npz,file_hash,read_json,utcnow

REQUIRED=('neural_mean','neural_scale','observed_neural','neural_target_mask',
          'participant_id','recording_id','forecast_origin_sec','observation_cutoff_sec','supplied_forcing_mg')


def checked_arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        missing=set(REQUIRED)-set(archive.files)
        if missing:raise ValueError('Prediction file lacks '+','.join(sorted(missing)))
        a={k:archive[k].copy() for k in REQUIRED}
    shape=a['neural_mean'].shape
    if len(shape)!=3 or shape[1:]!=(15,16) or not shape[0]:raise ValueError('Unexpected primary prediction shape')
    for key in ('neural_scale','observed_neural','neural_target_mask'):
        if a[key].shape!=shape:raise ValueError('Prediction arrays do not align')
    if a['neural_target_mask'].dtype!=bool:raise ValueError('Target quality mask must be boolean')
    for key in ('participant_id','recording_id','forecast_origin_sec','observation_cutoff_sec'):
        if a[key].shape!=(shape[0],):raise ValueError('Prediction identifiers do not align')
    for key in ('forecast_origin_sec','observation_cutoff_sec'):
        if a[key].dtype.kind not in 'iu' or (a[key]<0).any():raise ValueError('Forecast times must be nonnegative integer seconds')
    if (a['observation_cutoff_sec']>a['forecast_origin_sec']).any():raise ValueError('Observation cutoff is after forecast origin')
    if a['supplied_forcing_mg'].shape!=(shape[0],15,2) or not np.isfinite(a['supplied_forcing_mg']).all():
        raise ValueError('Forcing scenario has wrong shape or nonfinite values')
    if not np.isfinite(a['neural_mean']).all() or not np.isfinite(a['neural_scale']).all() or (a['neural_scale']<=0).any():
        raise FloatingPointError('Nonfinite/invalid predictive distribution')
    mask=a['neural_target_mask']
    if not np.isfinite(a['observed_neural'][mask]).all():raise ValueError('Observed neural targets include unmasked nonfinite values')
    for name in ('participant_id','recording_id'):
        if a[name].dtype.kind not in 'US' or any(not str(x) for x in a[name]):raise ValueError('Empty or non-text source identity')
    keys=window_keys(a)
    if len(set(keys))!=len(keys):raise ValueError('Repeated forecast key would double count a window')
    return a


def window_keys(arrays):
    return list(zip(arrays['participant_id'].tolist(),arrays['recording_id'].tolist(),arrays['forecast_origin_sec'].tolist()))


def matched_arrays(left,right):
    """Require the full common eligible sample, never silently intersect outcomes."""
    lk,rk=window_keys(left),window_keys(right)
    if set(lk)!=set(rk):raise ValueError('Models use different windows; declare a common eligible cohort before comparison')
    index={k:i for i,k in enumerate(rk)};order=np.asarray([index[k] for k in lk])
    aligned={k:v[order] for k,v in right.items()}
    for name in ('neural_target_mask','observation_cutoff_sec','supplied_forcing_mg'):
        if not np.array_equal(left[name],aligned[name]):raise ValueError('Model input/target lineage differs: '+name)
    mask=left['neural_target_mask']
    if not np.array_equal(left['observed_neural'][mask],aligned['observed_neural'][mask]):
        raise ValueError('Models are not evaluated against the same standardized targets')
    return aligned


def score_arrays(a,horizon_weights=(.2,.3,.5)):
    mean=a['neural_mean'].astype(float);scale=a['neural_scale'].astype(float)
    mask=a['neural_target_mask'];target=np.where(mask,a['observed_neural'],mean)
    z=(target-mean)/scale
    crps=scale*(z*erf(z/math.sqrt(2))+math.sqrt(2/math.pi)*np.exp(-z*z/2)-1/math.sqrt(math.pi))
    count=mask.sum(axis=2)
    if (count[:,[0,4,14]]==0).any():raise ValueError('No valid neural feature at a primary horizon')
    per_horizon=np.where(mask,crps,0.).sum(axis=2)/np.maximum(count,1)
    cover=np.where(mask,np.abs(target-mean)<=1.6448536269514722*scale,False).sum(axis=2)/np.maximum(count,1)
    at=[0,4,14];weights=np.asarray(horizon_weights,float)
    if weights.shape!=(3,) or (weights<0).any() or not np.isclose(weights.sum(),1):raise ValueError('Invalid primary horizon weights')
    window_crps=per_horizon[:,at]@weights;window_coverage=cover[:,at]@weights
    by_person={};by_horizon={};coverage={}
    for person in sorted(set(a['participant_id'].tolist())):
        use=a['participant_id']==person
        by_person[person]=float(window_crps[use].mean())
        by_horizon[person]=per_horizon[use][:,at].mean(axis=0).tolist()
        coverage[person]=float(window_coverage[use].mean())
    return {'neural_crps':float(np.mean(list(by_person.values()))),
            'coverage90':float(np.mean(list(coverage.values()))),'participant_scores':by_person,
            'participant_coverage':coverage,'participant_horizon_crps':by_horizon,
            'horizon_crps':np.mean(list(by_horizon.values()),axis=0).tolist(),
            'participants':len(by_person),'windows':len(window_crps),'horizon_sec':[1,5,15],
            'horizon_weights':list(horizon_weights),'primary_target':'future_neural_only'}


def verify_and_score(folder,allowed_participants=None,horizon_weights=(.2,.3,.5)):
    folder=Path(folder);receipt=read_json(folder/'scores.json');path=folder/'predictions.npz'
    if receipt['prediction_file_sha256']!=file_hash(path):raise ValueError('Prediction receipt checksum mismatch')
    a=checked_arrays(path)
    if allowed_participants is not None and not set(a['participant_id'].tolist())<=set(allowed_participants):
        raise PermissionError('Predictions contain participants outside the declared evaluation partition')
    scores=score_arrays(a,horizon_weights)
    for key in ('neural_crps','coverage90'):
        if not np.isclose(scores[key],receipt[key],rtol=2e-6,atol=1e-7):
            raise ValueError('Independent score differs from saved model receipt: '+key)
    return scores,a


def freeze_prediction(source_folder,destination,provenance,allowed_participants=None,horizon_weights=(.2,.3,.5)):
    """An additive export; preserves original prediction files and numeric arrays."""
    required={'model_sha256','feature_manifest_sha256','configuration_sha256','code_commit','evaluation_partition'}
    if not required<=set(provenance) or any(not provenance[k] for k in required):raise ValueError('Frozen prediction provenance is incomplete')
    scores,arrays=verify_and_score(source_folder,allowed_participants,horizon_weights)
    source_receipt=read_json(Path(source_folder)/'scores.json')
    if not source_receipt.get('created_utc'):raise ValueError('Source prediction generation time is missing')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Choose a new frozen prediction directory')
    destination.mkdir(parents=True)
    stamp=source_receipt['created_utc'];exported=utcnow()
    atomic_npz(destination/'predictions.npz',**arrays,model_sha256=np.asarray(provenance['model_sha256']),
               generated_utc=np.asarray(stamp),evaluation_partition=np.asarray(provenance['evaluation_partition']),
               feature_manifest_sha256=np.asarray(provenance['feature_manifest_sha256']))
    record={**scores,**provenance,'created_utc':stamp,'exported_utc':exported,'source_predictions_sha256':file_hash(Path(source_folder)/'predictions.npz'),
            'source_scores_sha256':file_hash(Path(source_folder)/'scores.json'),
            'prediction_file_sha256':file_hash(destination/'predictions.npz')}
    atomic_json(destination/'scores.json',record)
    return record
