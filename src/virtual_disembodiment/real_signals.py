"""Causal waveform processing for the published DOSE-I/HMC data streams.

All features at time t use raw samples strictly before t. No centred filter,
whole-record adaptive peak threshold, or backward imputation is used.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


EEG_FEATURES = ('log_delta', 'log_theta', 'log_alpha', 'log_beta', 'log_gamma',
                'log_rms', 'mean_abs', 'line_length')
BODY_FEATURES = ('ecg_log_rms', 'ecg_mean_abs', 'ecg_slope_rms', 'rr_last',
                 'heart_rate_last', 'confirmed_beats', 'rr_std_30sec',
                 'pleth_mean', 'pleth_log_rms', 'resp_derived_mean',
                 'resp_derived_log_rms', 'systolic_pressure', 'diastolic_pressure',
                 'mean_pressure')


def causal_grid(times, values, fs, end_sec, max_hold_sec):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError('strictly increasing timestamps and aligned 1-D values required')
    grid = np.arange(int(np.floor(end_sec * fs)), dtype=np.float64)/fs
    good = np.isfinite(values) & np.isfinite(times)
    if not good.any():
        return grid, np.zeros(grid.size), np.zeros(grid.size, dtype=bool)
    source_t, source_v = times[good], values[good]
    index = np.searchsorted(source_t, grid, side='right')-1
    safe = np.maximum(index, 0)
    valid = (index >= 0) & (grid-source_t[safe] <= max_hold_sec+1e-10)
    output = source_v[safe].copy(); output[~valid] = 0
    return grid, output, valid


def causal_filter(values, valid, fs, band, order, warmup_sec):
    """State is reset over missing runs; each restart carries an explicit mask."""
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    if not (0 < band[0] < band[1] < fs/2):
        raise ValueError('band must lie strictly inside Nyquist')
    sos = signal.butter(order, band, btype='bandpass', fs=fs, output='sos')
    output = np.zeros_like(values); usable = valid.copy()
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges==1), np.flatnonzero(edges==-1)
    warmup = int(np.ceil(warmup_sec*fs))
    for a, b in zip(starts, ends):
        output[a:b] = signal.sosfilt(sos, values[a:b])
        usable[a:min(b, a+warmup)] = False
    return output, usable


def flatline_causal_mask(eeg, fs, threshold, duration_sec):
    """A completed two-second block affects the NEXT block, never its past."""
    block = int(2*fs)
    usable = np.ones(len(eeg), dtype=bool)
    run = 0
    for a in range(0, len(eeg)-block+1, block):
        b = a+block
        run = run+1 if np.var(eeg[a:b]) < threshold else 0
        if run*2 >= duration_sec:
            usable[b:min(len(eeg), b+block)] = False
    return usable


def eeg_second_features(filtered, valid, fs, config):
    n = len(filtered)//fs
    x = np.asarray(filtered[:n*fs]).reshape(n, fs)
    mask = np.asarray(valid[:n*fs]).reshape(n, fs)
    window = np.hanning(fs)
    # Fixed one-second, completed windows, not publisher pEEG (which includes labels).
    spectrum = np.abs(np.fft.rfft(x*window[None, :], axis=1))**2/(fs*np.square(window).sum())
    frequency = np.fft.rfftfreq(fs, d=1/fs)
    vals = []
    for low, high in config['bands_hz'].values():
        vals.append(np.log(np.maximum(spectrum[:,(frequency>=low)&(frequency<high)].sum(axis=1), 1e-12)))
    vals += [np.log(np.maximum(np.sqrt(np.mean(x*x, axis=1)), 1e-12)),
             np.mean(np.abs(x), axis=1), np.mean(np.abs(np.diff(x, axis=1)), axis=1)]
    values = np.column_stack(vals).astype(np.float32)
    usable = mask.mean(axis=1) >= config['min_observed_fraction']
    values[~usable] = np.nan
    return values, np.repeat(usable[:,None], len(vals), axis=1), mask.mean(axis=1)


def causal_r_peaks(ecg, valid, fs, config):
    """Peak times plus availability times; one-sample confirmation is explicit.

    This transparent QRS detector is an engineering feature extractor, not an
    adjudicated medical detector. Implausible intervals remain missing.
    """
    qrs, good = causal_filter(ecg, valid, fs, config['bandpass_hz'], 2,
                              config['threshold_history_sec'])
    diff = np.diff(qrs, prepend=0)
    k = max(1, round(config['integration_sec']*fs))
    envelope = signal.lfilter(np.ones(k)/k, [1.0], diff*diff)
    width = max(3, round(config['threshold_history_sec']*fs))
    sums = np.cumsum(np.r_[0., envelope]); squares = np.cumsum(np.r_[0., envelope**2])
    at = np.arange(len(envelope)); left = np.maximum(at-width,0)
    count = np.maximum(at-left, 1)
    # Each threshold excludes the candidate and later samples.
    mean = (sums[at]-sums[left])/count
    var = np.maximum((squares[at]-squares[left])/count - mean**2, 0)
    threshold = mean+config['threshold_sd']*np.sqrt(var)
    local = np.flatnonzero((envelope[1:-1] > envelope[:-2]) &
                           (envelope[1:-1] >= envelope[2:]) &
                           (envelope[1:-1] > threshold[1:-1]))+1
    peaks = []
    for i in local:
        # A larger future peak never replaces an already emitted peak.
        if good[i-1:i+2].all() and (not peaks or i-peaks[-1]>=config['refractory_sec']*fs):
            peaks.append(int(i))
    indices = np.asarray(peaks, dtype=np.int64)
    return indices/fs, (indices+1)/fs


def body_second_features(ecg, valid, fs, times, extra, nseconds, config):
    x = ecg[:nseconds*fs].reshape(nseconds, fs)
    good = valid[:nseconds*fs].reshape(nseconds, fs)
    values = np.full((nseconds, len(BODY_FEATURES)), np.nan, dtype=np.float32)
    values[:,0] = np.log(np.maximum(np.sqrt(np.mean(x*x, axis=1)),1e-12))
    values[:,1] = np.mean(np.abs(x),axis=1)
    values[:,2] = np.sqrt(np.mean(np.diff(x,axis=1)**2,axis=1))
    wave_ok = good.mean(axis=1)>=config['min_observed_fraction']
    values[~wave_ok,:3] = np.nan
    peak, available = causal_r_peaks(ecg, valid, fs, config['r_peak'])
    rr = np.diff(peak)
    lo, hi = config['r_peak']['rr_range_sec']
    accepted_rr = (rr>=lo)&(rr<=hi)
    for second in range(1, nseconds+1):
        seen = np.flatnonzero(available < second)
        values[second-1,5] = np.sum((available>=second-1)&(available<second))
        if seen.size >= 2:
            end = seen[-1]-1
            if accepted_rr[end] and second-available[end+1]<=5:
                values[second-1,3:5] = [rr[end],60/rr[end]]
            use = accepted_rr & (available[1:]<second) & (available[1:]>=second-30)
            if use.sum()>=2:
                values[second-1,6] = np.std(rr[use])
    for key, first_col in [('pleth',7),('resp_derived',9)]:
        if key not in extra:
            continue
        t, v = extra[key]
        for second in range(1,nseconds+1):
            part = v[(t>=second-1)&(t<second)&np.isfinite(v)]
            if part.size >= 10:
                values[second-1,first_col] = np.mean(part)
                values[second-1,first_col+1] = np.log(max(np.sqrt(np.mean(part*part)),1e-12))
    for key, col in [('systolic_pressure',11),('diastolic_pressure',12),('mean_pressure',13)]:
        if key not in extra:
            continue
        t, v = extra[key]; good = np.isfinite(v); t, v = t[good],v[good]
        for second in range(1,nseconds+1):
            i = np.searchsorted(t,second,side='left')-1
            if i>=0 and second-t[i]<=300:
                values[second-1,col] = v[i]
    # No peak-derived value is available where the underlying ECG is unusable.
    values[~wave_ok,3:7] = np.nan
    return values, np.isfinite(values), {'confirmed_peak_count':int(peak.size),
                'plausible_rr_fraction':float(accepted_rr.mean()) if rr.size else None,
                'detector_validated_by_human':False,
                'peak_confirmation_delay_sec':1/fs}


def eeg_ecg_regressors(ecg, ecg_valid, length, eeg_fs, ecg_fs, lags):
    t = np.arange(length)/eeg_fs
    columns, masks = [], []
    for lag in lags:
        at = np.floor((t-lag)*ecg_fs+1e-8).astype(np.int64)
        safe = np.clip(at,0,len(ecg)-1)
        columns.append(ecg[safe]); masks.append((at>=0)&ecg_valid[safe])
    return np.column_stack(columns), np.logical_and.reduce(masks)


def cardiac_regression_sufficient_stats(eeg, eeg_valid, ecg, ecg_valid, config):
    reg, good = eeg_ecg_regressors(ecg,ecg_valid,len(eeg),config['eeg_grid_hz'],
                                  config['ecg_grid_hz'],config['artifact_regression_lags_sec'])
    good &= eeg_valid.all(axis=1)
    return reg[good].T@reg[good], reg[good].T@eeg[good], int(good.sum())


def waveform_to_features(eeg, eeg_valid, ecg, ecg_valid, config, extra=None, cardiac_beta=None):
    fs = config['eeg_grid_hz']; ecg_fs = config['ecg_grid_hz']
    if np.asarray(eeg).ndim != 2 or eeg.shape[1] != 2:
        raise ValueError('Primary neural stream needs exactly two explicitly mapped EEG channels')
    nsec = min(len(eeg)//fs, len(ecg)//ecg_fs)
    raw_filtered, raw_mask, qualities = [], [], []
    for ch in range(2):
        valid = eeg_valid[:,ch] & (np.abs(eeg[:,ch])<config['eeg_saturation_uv'])
        valid &= flatline_causal_mask(eeg[:,ch],fs,config['causal_flat_variance_uv2'],
                                     config['causal_flat_detection_sec'])
        # Filtering availability and quality are separate: a saturation does not
        # cause a 10-second reset for every adjacent sample, but remains masked.
        filtered, mask = causal_filter(eeg[:,ch],eeg_valid[:,ch],fs,
                    config['eeg_bandpass_hz'],config['filter_order'],config['gap_filter_warmup_sec'])
        mask &= valid
        raw_filtered.append(filtered); raw_mask.append(mask)
    raw_filtered, raw_mask = np.column_stack(raw_filtered), np.column_stack(raw_mask)
    streams = {'conservative':raw_filtered}
    if cardiac_beta is not None:
        reg, reg_valid = eeg_ecg_regressors(ecg,ecg_valid,len(eeg),fs,ecg_fs,
                                           config['artifact_regression_lags_sec'])
        # Regression was fit on DEVELOPMENT participants only. Filter the same
        # causal ECG contribution, using the same reset rules, before subtraction.
        contribution = reg@cardiac_beta
        robust = raw_filtered.copy()
        for ch in range(2):
            filtered, mask = causal_filter(contribution[:,ch], reg_valid, fs,
                        config['eeg_bandpass_hz'],config['filter_order'],config['gap_filter_warmup_sec'])
            robust[:,ch] -= filtered
        streams['cardiac_robust'] = robust
    outputs = {}
    for name, stream in streams.items():
        feats, masks = [], []
        for ch in range(2):
            f,m,q = eeg_second_features(stream[:,ch],raw_mask[:,ch],fs,config)
            feats.append(f[:nsec]); masks.append(m[:nsec]); qualities.append(q[:nsec])
        outputs['neural_'+name] = np.column_stack(feats)
        outputs['neural_mask_'+name] = np.column_stack(masks)
    bf, bm, qrs = body_second_features(ecg,ecg_valid,ecg_fs,None,extra or {},nsec,config)
    outputs.update(body=bf,body_mask=bm,token_end_sec=np.arange(1,nsec+1,dtype=np.int64),
                   neural_observed_fraction=np.column_stack(qualities[:2]),
                   ecg_observed_fraction=ecg_valid[:nsec*ecg_fs].reshape(nsec,ecg_fs).mean(axis=1))
    return outputs, qrs
