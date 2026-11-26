"""Resumable full-budget fitting of observation-prior and comparator families.

Kept separate from the original engine so its active checkpoints and source
fingerprints remain unchanged. Future physiology enters loss targets only.
"""
from __future__ import annotations
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .neural_dynamics import probabilistic_training_loss
from .run_io import atomic_json, atomic_npz, file_hash, object_hash, read_json, require_same_or_new, utcnow
from .suite_data import SuiteRecords, metadata_encoding
from .suite_models import PriorDynamics, FeedForwardAlternative
from .training import (TENSOR_KEYS, batches, repeat_encoded, neural_window_scores,
                       validation_groups, atomic_checkpoint, learning_rate_multiplier)

EXTRA_KEYS = ('history_prior_neural', 'history_prior_neural_mask', 'history_prior_body',
              'history_prior_body_mask', 'history_time_sec', 'static_metadata')
ENCODE_KEYS = {'history_prior_neural': 'prior_neural', 'history_prior_neural_mask': 'prior_neural_mask',
               'history_prior_body': 'prior_body', 'history_prior_body_mask': 'prior_body_mask',
               'history_time_sec': 'history_time_sec', 'static_metadata': 'static_metadata'}


def device_batch(rows, device):
    keys = TENSOR_KEYS + tuple(k for k in EXTRA_KEYS if k in rows[0])
    if any(set(k for k in EXTRA_KEYS if k in row) != set(k for k in EXTRA_KEYS if k in rows[0]) for row in rows):
        raise ValueError('Mixed optional input schemas in one batch')
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), device=device) for k in keys}


def encode_batch(model, batch, dropout=0., remove_body_measurements=False):
    # Explicit allowlist: no loss target, clinical label, or future forcing is
    # forwarded to a history encoder, even if the batch holds those fields.
    extra = {target: batch[source] for source, target in ENCODE_KEYS.items() if source in batch}
    return model.encode_history(batch['history_neural'], batch['history_neural_mask'],
                                batch['history_body'], batch['history_body_mask'], batch['history_forcing'],
                                observation_dropout=dropout, remove_body_measurements=remove_body_measurements, **extra)


def build_model(spec, offline_parameters=0, metadata_dim=35):
    prior = spec.get('observation_prior', 'scratch')
    if prior not in ('scratch', 'sleepfm_frozen', 'labram_frozen'):
        raise ValueError('Unknown observation prior')
    if spec.get('initialization', 'scratch') != 'scratch':
        raise ValueError('The dynamics core is scratch; frozen observation inputs are declared separately')
    common = {'observation_prior': prior, 'offline_encoder_parameters': offline_parameters}
    if spec['kind'] in FeedForwardAlternative.KINDS:
        if spec['backbone'] != 'nonrecursive_feedforward':
            raise ValueError('Nonrecursive comparator requires its declared backbone')
        return FeedForwardAlternative(spec['kind'], context=spec.get('context_sec', 60),
                                      width=spec.get('dimension', 1920), depth=spec.get('layers', 3),
                                      metadata_dim=metadata_dim, **common)
    if spec['kind'] not in ('coupled', 'brain_only', 'shared_driver', 'memory_replacement'):
        raise ValueError('Unknown dynamics kind')
    return PriorDynamics(dimension=spec.get('dimension', 256), layers=spec.get('layers', 8),
                         backbone=spec['backbone'], kind=spec['kind'], **common)


def training_condition(model, spec, config):
    regime = spec.get('training_regime', 'natural')
    if regime == 'natural':
        return False, {}
    if regime != 'disconnection_augmented':
        raise ValueError('Unknown training regime')
    if isinstance(model, FeedForwardAlternative):
        raise ValueError('Nonrecursive controls have no message gates; do not label them disconnection-trained')
    # One perturbation per logical batch, drawn before microbatch partitioning.
    gains = config['interventions']['gains']
    gain = float(gains[int(torch.randint(len(gains), ()).item())])
    mode = int(torch.randint(5, ()).item())
    return mode == 4, ({'ascending_gain': gain} if mode == 1 else
                       {'descending_gain': gain} if mode == 2 else
                       {'ascending_gain': gain, 'descending_gain': gain} if mode == 3 else {})


def sample_distribution(model, batch, samples=16, **intervention):
    effective = 1 if isinstance(model, FeedForwardAlternative) else samples
    if effective < 1:
        raise ValueError('At least one latent rollout is required')
    encoded = encode_batch(model, batch)
    if effective != 1:
        encoded = repeat_encoded(encoded, effective)
    prediction = model.simulate(encoded, batch['future_forcing'].repeat_interleave(effective, dim=0), **intervention)
    size = len(batch['history_neural'])
    mean = prediction['neural_mean'].float().reshape(size, effective, -1, model.neural_dim)
    scale = prediction['neural_scale'].float().reshape_as(mean)
    mu = mean.mean(1)
    variance = (scale.square() + mean.square()).mean(1) - mu.square()
    return mu, variance.clamp_min(1e-8).sqrt()


def evaluate_suite(model, dataset, device, batch_size=8, samples=16, seed=4001, output=None,
                   model_hash=None, horizon_weights=(.2, .3, .5), scale_factor=1.):
    if not math.isfinite(scale_factor) or scale_factor<=0:
        raise ValueError('Predictive scale calibration must be finite and positive')
    device = torch.device(device)
    model.eval()
    by_person = defaultdict(list)
    cover = defaultdict(list)
    arrays = defaultdict(list)
    origins, people, recordings = [], [], []
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    start = time.monotonic()
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        for rows in batches(dataset.evaluation_windows(), batch_size):
            batch = device_batch(rows, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                mean, scale = sample_distribution(model, batch, samples)
            scale = scale * scale_factor
            score, cov = neural_window_scores(mean, scale, batch['target_neural'], batch['target_neural_mask'], horizon_weights)
            for row, s, c in zip(rows, score.cpu().tolist(), cov.cpu().tolist()):
                by_person[row['participant_id']].append(s)
                cover[row['participant_id']].append(c)
            if output:
                for name, value in (('neural_mean', mean), ('neural_scale', scale),
                                    ('observed_neural', batch['target_neural']), ('neural_target_mask', batch['target_neural_mask']),
                                    ('supplied_forcing_mg', batch['future_forcing'] * 10.)):
                    arrays[name].append(value.cpu().numpy())
                origins.extend(r['forecast_origin_sec'] for r in rows)
                people.extend(r['participant_id'] for r in rows)
                recordings.extend(r['recording_id'] for r in rows)
    effective = 1 if isinstance(model, FeedForwardAlternative) else samples
    result = {'neural_crps': float(np.mean([np.mean(v) for v in by_person.values()])),
              'coverage90': float(np.mean([np.mean(v) for v in cover.values()])),
              'participant_scores': {p: float(np.mean(v)) for p, v in by_person.items()},
              'participant_coverage': {p: float(np.mean(v)) for p, v in cover.items()},
              'participants': len(by_person), 'windows': sum(map(len, by_person.values())),
              'elapsed_sec': time.monotonic() - start, 'generation_samples': effective,
              'distribution': ('analytical Gaussian emission from parallel nonrecursive readouts' if effective == 1 else
                               'Gaussian moment match of learned emission variance and stochastic latent rollouts'),
              'primary_target': 'future_neural_only', 'task': dataset.task, 'created_utc': utcnow(), 'model_hash': model_hash}
    if not math.isfinite(result['neural_crps']):
        raise FloatingPointError('Nonfinite validation score')
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_npz(output / 'predictions.npz', **{k: np.concatenate(v) for k, v in arrays.items()},
                   participant_id=np.asarray(people), recording_id=np.asarray(recordings),
                   forecast_origin_sec=np.asarray(origins), observation_cutoff_sec=np.asarray(origins))
        result['prediction_file_sha256'] = file_hash(output / 'predictions.npz')
        atomic_json(output / 'scores.json', result)
    return result


def implementation_hashes():
    root = Path(__file__).parent
    names = ('suite_training.py', 'suite_data.py', 'suite_models.py', 'training.py', 'neural_dynamics.py',
             'forecast_data.py', 'foundation_features.py', 'run_io.py')
    return {name: file_hash(root / name) for name in names}


def fit_suite_model(run_root, config, spec, output, device='cuda'):
    run_root, output = Path(run_root), Path(output)
    feature_root = run_root / 'preprocessing'
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU is unavailable')
    seed = int(spec['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = read_json(feature_root / 'feature-manifest.json')
    groups = validation_groups(manifest, run_root, config['split']['seed'])
    prior = spec.get('observation_prior', 'scratch')
    foundation_manifest = run_root / 'foundation/feature-manifest.json'
    offline_parameters = 0
    encoder_receipt = None
    if prior != 'scratch':
        encoder_receipt = read_json(run_root / 'foundation/encoder-inventory.json')
        if encoder_receipt['feature_manifest_sha256'] != file_hash(foundation_manifest):
            raise ValueError('Encoder inventory belongs to different frozen observations')
        key = {'sleepfm_frozen': 'sleepfm_frozen_parameters', 'labram_frozen': 'labram_frozen_encoder_parameters'}[prior]
        offline_parameters = encoder_receipt['inventory'][key]
        if encoder_receipt['inventory']['trainable_parameters'] != 0:
            raise ValueError('Observation cache must come from frozen encoders')
    metadata, metadata_spec = (None, None)
    if spec['kind'] == 'metadata_only':
        metadata, metadata_spec = metadata_encoding(run_root.parents[1], manifest, run_root)
    data_args = {'context': spec.get('context_sec', 60), 'common_context': spec.get('common_context_sec'),
                 'pipeline': spec.get('pipeline', 'conservative'), 'task': spec.get('task', 'forced_simulation'),
                 'observation_prior': prior, 'foundation_manifest': foundation_manifest, 'metadata': metadata}
    training = SuiteRecords(feature_root / 'feature-manifest.json', feature_root / 'feature-scaling.json',
                            'development', stride=1, **data_args)
    validation = SuiteRecords(feature_root / 'feature-manifest.json', feature_root / 'feature-scaling.json',
                              'validation', stride=config['forecast']['evaluation_window_stride_sec'],
                              participant_subset=groups['selection'], **data_args)
    model = build_model(spec, offline_parameters, 35 if metadata_spec is None else metadata_spec['dimension']).to(device)
    inventory = model.parameter_inventory()
    low, high = config['models']['pilot_parameter_range'] if spec.get('scale', 'pilot') == 'pilot' else config['models']['main_parameter_range']
    if not low <= inventory['trainable'] <= high:
        raise ValueError('Actual trainable parameter count is outside the registered scale range')
    # Validate regime even when a completed fit is being inspected.
    if spec.get('training_regime', 'natural') not in ('natural', 'disconnection_augmented'):
        raise ValueError('Unknown training regime')
    if isinstance(model, FeedForwardAlternative) and spec.get('training_regime', 'natural') != 'natural':
        raise ValueError('No message gates exist in a nonrecursive comparator')
    frozen = {'engine': 'suite-v1', 'model_spec': spec, 'training': config['training'], 'forecast': config['forecast'],
              'augmentation_gains': config['interventions']['gains'], 'model_inventory': inventory,
              'feature_manifest_sha256': object_hash(manifest), 'validation_groups_sha256': object_hash(groups),
              'training_cohort': training.cohort_summary(), 'selection_cohort': validation.cohort_summary(),
              'implementation_sha256': implementation_hashes(), 'encoder_receipt': encoder_receipt,
              'metadata_spec': metadata_spec,
              'core_initialization': 'scratch', 'observation_prior': prior,
              'selection_and_early_stop_on_natural_validation': True}
    require_same_or_new(output / 'fit-spec.json', frozen)
    spec_hash = object_hash(frozen)
    if (output / 'completion.json').exists():
        result = read_json(output / 'completion.json')
        if result['spec_sha256'] != spec_hash or file_hash(output / 'best.pt') != result['best_checkpoint_sha256']:
            raise ValueError('Completed model fingerprint mismatch')
        scores = read_json(output / 'selection-predictions/scores.json')
        if scores != result['selection_scores'] or file_hash(output / 'selection-predictions/predictions.npz') != scores['prediction_file_sha256']:
            raise ValueError('Completed model prediction receipt mismatch')
        return result
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    settings = config['training']
    logical = settings['participant_first_batch_size']
    micro = min(logical, spec.get('micro_batch_size', 32 if spec.get('scale', 'pilot') == 'pilot' else 2))
    max_epochs = settings['max_epochs']
    per_person = settings['windows_per_participant_per_epoch']
    per_epoch = math.ceil(len(training.participants) * per_person / logical)
    weights = tuple(spec.get('loss_weights', settings['loss_weights'][0]))
    if micro < 1 or len(weights) != 3 or any(w < 0 or not math.isfinite(w) for w in weights):
        raise ValueError('Invalid microbatch size or loss coefficients')
    epoch, skip_batches, step, best, patience, history = 0, 0, 0, float('inf'), 0, []
    last = output / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved['spec_sha256'] != spec_hash:
            raise ValueError('Checkpoint configuration/source mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, skip_batches, step = saved['next_epoch'], saved['skip_batches'], saved['global_step']
        best, patience, history, micro = saved['best_crps'], saved['patience'], saved['history'], saved['micro_batch_size']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])

    def checkpoint(next_epoch, skip):
        atomic_checkpoint(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'next_epoch': next_epoch, 'skip_batches': skip, 'global_step': step,
                                'best_crps': best, 'patience': patience, 'history': history, 'micro_batch_size': micro,
                                'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                                'numpy_rng': np.random.get_state(), 'spec_sha256': spec_hash,
                                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})

    begin = time.monotonic()
    for e in range(epoch, max_epochs):
        if patience >= settings['patience']:
            break
        model.train()
        epoch_start, loss_sum, seen = time.monotonic(), 0., 0
        for batch_index, rows in enumerate(batches(training.training_epoch(seed + e * 1009, per_person), logical)):
            if e == epoch and batch_index < skip_batches:
                continue
            for group in optimizer.param_groups:
                group['lr'] = spec['learning_rate'] * learning_rate_multiplier(step, per_epoch * max_epochs, settings['warmup_fraction'])
            remove, intervention = training_condition(model, spec, config)
            while True:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.
                saved_rng = torch.get_rng_state()
                saved_cuda = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
                try:
                    for start in range(0, len(rows), micro):
                        small = rows[start:start + micro]
                        batch = device_batch(small, device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                            encoded = encode_batch(model, batch, dropout=.1, remove_body_measurements=remove)
                            prediction = model.simulate(encoded, batch['future_forcing'], **intervention)
                            loss, _ = probabilistic_training_loss(prediction, batch, weights)
                        factor = len(small) / len(rows)
                        (loss * factor).backward()
                        batch_loss += float(loss.detach()) * factor
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'])
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite gradient norm')
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if micro == 1:
                        raise
                    batch = encoded = prediction = loss = None
                    micro = max(1, micro // 2)
                    torch.cuda.empty_cache()
                    torch.set_rng_state(saved_rng)
                    torch.cuda.set_rng_state_all(saved_cuda)
            optimizer.step()  # optimizer OOM is fatal, never retry a partly applied update
            step += 1
            loss_sum += batch_loss
            seen += 1
            if step % 25 == 0:
                atomic_json(output / 'status.json', {'state': 'TRAINING', 'pid': os.getpid(), 'updated_utc': utcnow(),
                            'epoch': e + 1, 'batch': batch_index + 1, 'batches_per_epoch': per_epoch,
                            'max_epochs': max_epochs, 'global_step': step, 'micro_batch_size': micro,
                            'effective_batch_size': logical, 'observation_prior': prior,
                            'best_selection_neural_crps': None if not math.isfinite(best) else best})
            if step % 100 == 0:
                checkpoint(e, batch_index + 1)
        metric = evaluate_suite(model, validation, device, seed=seed + 90000,
                                horizon_weights=config['forecast']['horizon_weights'])
        row = {'epoch': e + 1, 'neural_crps': metric['neural_crps'], 'coverage90': metric['coverage90'],
               'training_loss': loss_sum / max(1, seen), 'elapsed_sec': time.monotonic() - epoch_start, 'global_step': step}
        history.append(row)
        if metric['neural_crps'] < best - 1e-6:
            best, patience = metric['neural_crps'], 0
            atomic_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'model_spec': spec, 'spec_sha256': spec_hash,
                                                  'selected_epoch': e + 1, 'selection_neural_crps': best})
        else:
            patience += 1
        checkpoint(e + 1, 0)
        skip_batches = 0
        atomic_json(output / 'learning-curve.json', history)
        atomic_json(output / 'status.json', {'state': 'EPOCH_COMPLETE', 'pid': os.getpid(), 'updated_utc': utcnow(),
                    **row, 'best_selection_neural_crps': best, 'patience_used': patience, 'micro_batch_size': micro})
        print(__import__('json').dumps({'event': 'epoch_complete', **row}), flush=True)
    if not (output / 'best.pt').exists():
        raise RuntimeError('No finite eligible checkpoint was produced')
    selected = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model'])
    model_hash = file_hash(output / 'best.pt')
    scores = evaluate_suite(model, validation, device, seed=seed + 90000, output=output / 'selection-predictions',
                            model_hash=model_hash, horizon_weights=config['forecast']['horizon_weights'])
    result = {'state': 'COMPLETE', 'best_checkpoint_sha256': model_hash, 'best_checkpoint': str(output / 'best.pt'),
              'selected_epoch': selected['selected_epoch'], 'epochs_completed': len(history), 'global_steps': step,
              'model_inventory': inventory, 'selection_scores': scores, 'spec_sha256': spec_hash,
              'protected_test_opened': False, 'fidelity_gate_participants_used_for_selection': False,
              'finished_utc': utcnow(), 'elapsed_sec_this_invocation': time.monotonic() - begin}
    atomic_json(output / 'completion.json', result)
    atomic_json(output / 'status.json', result)
    return result
