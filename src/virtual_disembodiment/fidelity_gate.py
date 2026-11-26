"""Validation-only fidelity decisions and protected-test readiness checks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .prediction_scores import matched_arrays,verify_and_score
from .run_io import file_hash,object_hash,read_json,utcnow

DEFAULT_POLICY={'coverage_range':[.85,.95],'minimum_relative_improvement':.05,
                'equivalence_tolerance':.01,'catastrophic_horizon15_ratio':2.,
                'drift_rule':'Every seed must have finite distributions and 15s neural CRPS at most twice the tuned LDS value.',
                'rule_status':'proposed_engineering_cutoff_pending_author_confirmation',
                'review':{'state':'pending_author_confirmation','confirmation_reference':None}}

REQUIRED_READINESS=('source_quality_and_grouping','statistical_baselines','neural_and_long_history_comparators',
                    'feedforward_multimodal_comparator','shortcut_comparators','matched_memory_and_shared_driver',
                    'pretrained_conditions_or_documented_resource_exclusion','preprocessing_sensitivity',
                    'autonomous_forecast','scale_decision','natural_and_disconnection_training',
                    'development_damage_controls','development_rescue_controllers','statistical_analysis_plan')


def aggregate_fixed_seeds(folders,expected_seeds,allowed_participants,reference=None):
    """Average scores over seeds within each participant, not over windows/people."""
    if set(folders)!=set(map(str,expected_seeds)):raise ValueError('The full prespecified seed set is required')
    scores=[];hashes={};first=reference
    for seed in expected_seeds:
        folder=Path(folders[str(seed)]);score,a=verify_and_score(folder,allowed_participants)
        if first is None:first=a
        else:matched_arrays(first,a)
        scores.append(score);hashes[str(seed)]={'predictions_sha256':file_hash(folder/'predictions.npz'),
                                               'scores_sha256':file_hash(folder/'scores.json')}
    people=sorted(scores[0]['participant_scores'])
    result={'participant_scores':{p:float(np.mean([s['participant_scores'][p] for s in scores])) for p in people},
            'participant_coverage':{p:float(np.mean([s['participant_coverage'][p] for s in scores])) for p in people},
            'horizon_crps':np.mean([s['horizon_crps'] for s in scores],axis=0).tolist(),
            'worst_seed_horizon15_crps':max(s['horizon_crps'][2] for s in scores),
            'participants':len(people),'windows':scores[0]['windows'],'seeds':list(expected_seeds),
            'prediction_receipts':hashes}
    result['neural_crps']=float(np.mean(list(result['participant_scores'].values())))
    result['coverage90']=float(np.mean(list(result['participant_coverage'].values())))
    return result,first


def decide(coupled,persistence,lds,coupled_neural_common,neural_history,policy=None):
    policy=dict(DEFAULT_POLICY if policy is None else policy)
    values=[r['neural_crps'] for r in (coupled,persistence,lds,coupled_neural_common,neural_history)]
    if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Positive finite CRPS values are required for relative comparisons')
    low,high=policy['coverage_range']
    policy_values=[low,high,policy['minimum_relative_improvement'],policy['equivalence_tolerance'],policy['catastrophic_horizon15_ratio']]
    if not np.isfinite(policy_values).all() or not (0<=low<high<=1) or not (0<=policy['minimum_relative_improvement']<1) or policy['equivalence_tolerance']<0 or policy['catastrophic_horizon15_ratio']<1:
        raise ValueError('Invalid fidelity operating policy')
    drift_values=[coupled['worst_seed_horizon15_crps'],lds['horizon_crps'][2],coupled['coverage90']]
    if not np.isfinite(drift_values).all() or min(drift_values[:2])<=0 or not 0<=coupled['coverage90']<=1:
        raise ValueError('Invalid drift or coverage score')
    relative=1-coupled_neural_common['neural_crps']/neural_history['neural_crps']
    drift_ratio=coupled['worst_seed_horizon15_crps']/lds['horizon_crps'][2]
    checks={'better_than_persistence':coupled['neural_crps']<persistence['neural_crps'],
            'better_than_tuned_LDS':coupled['neural_crps']<lds['neural_crps'],
            'coverage90_in_registered_range':low<=coupled['coverage90']<=high,
            'minimum_gain_over_preselected_neural_history':relative>=policy['minimum_relative_improvement']-1e-12,
            'all_seeds_stable_at_15sec':drift_ratio<=policy['catastrophic_horizon15_ratio']}
    review=policy.get('review',{})
    confirmed=review.get('state')=='confirmed' and bool(review.get('confirmation_reference'))
    return {'state':('VALIDATION_FIDELITY_PASS' if all(checks.values()) else 'VALIDATION_FIDELITY_FAIL')
                    if confirmed else 'FIDELITY_CHECKS_PENDING_POLICY_REVIEW',
            'validation_fidelity_passed':all(checks.values()) and confirmed,
            'provisional_checks_passed':all(checks.values()),'checks':checks,'policy':policy,
            'relative_improvement_over_neural_history':float(relative),'worst_seed_horizon15_to_LDS_ratio':float(drift_ratio),
            'coupled_primary':coupled,'persistence':persistence,'LDS':lds,
            'coupled_on_neural_common_cohort':coupled_neural_common,'preselected_neural_history':neural_history,
            'policy_review':review,'policy_confirmed':confirmed,
            'protected_test_opened':False,'created_utc':utcnow()}


def evaluate_gate(coupled_folders,persistence_folder,lds_folder,coupled_common_folders,neural_folders,
                  seeds,allowed_participants,policy=None):
    coupled,ref=aggregate_fixed_seeds(coupled_folders,seeds,allowed_participants)
    persistence,pa=verify_and_score(persistence_folder,allowed_participants);matched_arrays(ref,pa)
    lds,la=verify_and_score(lds_folder,allowed_participants);matched_arrays(ref,la)
    common,common_ref=aggregate_fixed_seeds(coupled_common_folders,seeds,allowed_participants)
    neural,_=aggregate_fixed_seeds(neural_folders,seeds,allowed_participants,common_ref)
    return decide(coupled,persistence,lds,common,neural,policy)


def eligible_equivalence_set(validation_scores,policy=None):
    """Retain equally predictive alternatives, including those without body routes."""
    policy=DEFAULT_POLICY if policy is None else policy;low,high=policy['coverage_range']
    if not validation_scores:raise ValueError('Empty final-model collection')
    if not all(np.isfinite(v['neural_crps']) and v['neural_crps']>0 for v in validation_scores.values()):
        raise ValueError('Invalid final-model score')
    calibrated={k:v for k,v in validation_scores.items() if low<=v['coverage90']<=high}
    if not calibrated:return {'eligible':[],'best_calibrated_crps':None,'tolerance':policy['equivalence_tolerance']}
    best=min(v['neural_crps'] for v in calibrated.values())
    return {'eligible':sorted(k for k,v in calibrated.items() if v['neural_crps']<=best*(1+policy['equivalence_tolerance'])),
            'best_calibrated_crps':best,'tolerance':policy['equivalence_tolerance'],
            'intervention_results_used_for_membership':False}


def protected_authorization(feature_manifest,gate,readiness,root):
    """No test access merely because a small pilot or one model has passed."""
    review=gate.get('policy_review',{})
    if review.get('state')!='confirmed' or not review.get('confirmation_reference'):
        raise PermissionError('Custom fidelity cutoffs need an actual author confirmation before protected access')
    if not gate.get('validation_fidelity_passed'):raise PermissionError('Validation fidelity failed; protected analysis remains closed')
    missing=set(REQUIRED_READINESS)-set(readiness)
    if missing:raise PermissionError('Full retrospective preparation is incomplete: '+','.join(sorted(missing)))
    root=Path(root).resolve();checked={}
    for name in REQUIRED_READINESS:
        item=readiness[name];path=(root/item['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Readiness evidence must stay within the run directory')
        if file_hash(path)!=item['sha256']:raise ValueError('Readiness receipt checksum mismatch: '+name)
        record=read_json(path)
        if record.get('state') not in ('COMPLETE','COMPLETE_STATISTICAL_BASELINES','VERIFIED_RESOURCE_EXCLUSION'):
            raise PermissionError('Required stage is unfinished: '+name)
        if record.get('state')=='VERIFIED_RESOURCE_EXCLUSION' and name!='pretrained_conditions_or_documented_resource_exclusion':
            raise PermissionError('Only explicitly resource-conditional pretraining may use this exclusion status')
        if record.get('protected_test_opened') is not False:raise PermissionError('Readiness receipt does not establish a closed protected test')
        checked[name]={'path':str(path.relative_to(root)),'sha256':item['sha256']}
    return {'validation_fidelity_passed':True,'feature_manifest_sha256':object_hash(feature_manifest),
            'validation_gate_sha256':object_hash(gate),'preparation_receipts':checked,
            'created_utc':utcnow(),'protected_test_opened':False}
