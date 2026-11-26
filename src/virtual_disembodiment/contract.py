"""Explicit synchronized EEG+ECG contracts and channel audits.

The contract is deliberately stricter than a generic array loader.  It keeps
participant/recording identity, units, sampling clocks, provenance, and the
distinction between measured respiration and ECG-derived respiration visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .errors import ContractError, FutureLeakageError


@dataclass(frozen=True)
class ChannelSignal:
    name: str
    modality: str
    data: np.ndarray
    sample_rate_hz: float
    unit: str
    start_time_sec: float = 0.0
    independent_evidence: bool = True
    source_label: str | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim not in (1, 2):
            raise ContractError(f"{self.name}: signal must be 1-D or 2-D")
        if array.shape[0] < 2:
            raise ContractError(f"{self.name}: signal needs at least two samples")
        if not np.isfinite(float(self.sample_rate_hz)) or self.sample_rate_hz <= 0:
            raise ContractError(f"{self.name}: sample_rate_hz must be finite and positive")
        if not str(self.unit).strip():
            raise ContractError(f"{self.name}: physical unit is required")
        if not np.isfinite(float(self.start_time_sec)):
            raise ContractError(f"{self.name}: start_time_sec must be finite")

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.data).shape[0])

    @property
    def end_time_sec(self) -> float:
        return float(self.start_time_sec + (self.n_samples - 1) / self.sample_rate_hz)

    @property
    def n_channels(self) -> int:
        data = np.asarray(self.data)
        return 1 if data.ndim == 1 else int(data.shape[1])


@dataclass(frozen=True)
class Recording:
    participant_id: str
    recording_id: str
    dataset: str
    channels: Mapping[str, ChannelSignal]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.participant_id).strip():
            raise ContractError("participant_id is required; recording-level splitting is not allowed")
        if not str(self.recording_id).strip():
            raise ContractError("recording_id is required")
        if not str(self.dataset).strip():
            raise ContractError("dataset is required")
        if not self.channels:
            raise ContractError("at least one channel is required")

    def by_modality(self, modality: str) -> tuple[ChannelSignal, ...]:
        return tuple(channel for channel in self.channels.values() if channel.modality == modality)


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    message: str
    proposal_paragraph: str


@dataclass(frozen=True)
class ChannelAuditReport:
    participant_id: str
    recording_id: str
    dataset: str
    issues: tuple[AuditIssue, ...]

    @property
    def errors(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[AuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "recording_id": self.recording_id,
            "dataset": self.dataset,
            "ok": self.ok,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


@dataclass(frozen=True)
class SynchronizedSignals:
    time_sec: np.ndarray
    eeg: np.ndarray
    ecg: np.ndarray
    eeg_mask: np.ndarray
    ecg_mask: np.ndarray
    target_sample_rate_hz: float
    interpolation_mode: str = "causal_last_observation_hold"


def _issue(code: str, severity: str, message: str, paragraph: str) -> AuditIssue:
    return AuditIssue(code, severity, message, paragraph)


def audit_recording(recording: Recording, required_eeg_channels: int | None = None) -> ChannelAuditReport:
    issues: list[AuditIssue] = []
    eeg = recording.by_modality("eeg")
    ecg = recording.by_modality("ecg")
    derived_respiration = recording.by_modality("respiration_derived")

    if not eeg:
        issues.append(_issue("MISSING_EEG", "error", "no verified EEG channel", "P0073/P0081"))
    if not ecg:
        issues.append(_issue("MISSING_ECG", "error", "no concurrent ECG channel", "P0043/P0073/P0082"))
    eeg_channel_count = sum(channel.n_channels for channel in eeg)
    if required_eeg_channels is not None and eeg_channel_count < required_eeg_channels:
        issues.append(
            _issue(
                "INSUFFICIENT_PRIMARY_EEG",
                "error",
                f"{eeg_channel_count} EEG channels found; {required_eeg_channels} required for this primary resource",
                "P0049/P0073",
            )
        )

    verified_ecg = recording.metadata.get("verified_ecg")
    if ecg and verified_ecg is not True:
        issues.append(_issue("ECG_NOT_VERIFIED", "error", "ECG provenance is not explicitly verified", "P0069/P0073"))
    if recording.dataset.lower().replace("-", "") in {"ds005620", "openneurods005620"} and verified_ecg is not True:
        issues.append(
            _issue(
                "DS005620_ECG_UNVERIFIED",
                "error",
                "ds005620 is not assumed to contain a usable ECG/peripheral stream",
                "P0057-P0059/P0129",
            )
        )

    for channel in derived_respiration:
        if channel.independent_evidence:
            issues.append(
                _issue(
                    "DERIVED_RESPIRATION_MARKED_INDEPENDENT",
                    "error",
                    f"{channel.name} is ECG-derived but marked as independent respiration",
                    "P0043/P0082",
                )
            )
        else:
            issues.append(
                _issue(
                    "DERIVED_RESPIRATION_NOT_INDEPENDENT",
                    "warning",
                    f"{channel.name} remains a derived ECG variable, not independent respiratory evidence",
                    "P0043/P0082",
                )
            )

    for modality, channels in (("eeg", eeg), ("ecg", ecg)):
        units = {channel.unit for channel in channels}
        if len(units) > 1:
            issues.append(
                _issue(
                    "INCONSISTENT_UNITS",
                    "error",
                    f"{modality} channels have inconsistent physical units: {sorted(units)}",
                    "P0074/P0081-P0082/P0224",
                )
            )
        for channel in channels:
            if not np.isfinite(np.asarray(channel.data, dtype=float)).all():
                issues.append(
                    _issue(
                        "NONFINITE_SIGNAL",
                        "warning",
                        f"{channel.name} contains non-finite samples; masks must be retained",
                        "P0074",
                    )
                )

    if recording.metadata.get("clock_alignment_verified") is not True:
        issues.append(
            _issue(
                "CLOCK_ALIGNMENT_UNVERIFIED",
                "error",
                "recoverable EEG/ECG clock alignment is not explicitly verified",
                "P0073/P0069",
            )
        )

    if eeg and ecg:
        overlap_start = max(max(channel.start_time_sec for channel in eeg), max(channel.start_time_sec for channel in ecg))
        overlap_end = min(min(channel.end_time_sec for channel in eeg), min(channel.end_time_sec for channel in ecg))
        if overlap_end <= overlap_start:
            issues.append(
                _issue("NO_CLOCK_OVERLAP", "error", "EEG and ECG time ranges do not overlap", "P0073/P0080"))

    return ChannelAuditReport(recording.participant_id, recording.recording_id, recording.dataset, tuple(issues))


def audit_batch(recordings: list[Recording] | tuple[Recording, ...], required_eeg_channels: int | None = None) -> dict[str, Any]:
    """Audit recordings and detect cross-recording unit conflicts."""

    reports = [audit_recording(recording, required_eeg_channels=required_eeg_channels) for recording in recordings]
    unit_by_channel: dict[tuple[str, str], set[str]] = {}
    for recording in recordings:
        for channel in recording.channels.values():
            unit_by_channel.setdefault((channel.modality, channel.name), set()).add(channel.unit)
    cross_recording_issues = []
    for key, units in unit_by_channel.items():
        if len(units) > 1:
            cross_recording_issues.append(
                {
                    "code": "CROSS_RECORDING_UNIT_CONFLICT",
                    "severity": "error",
                    "channel": key,
                    "units": sorted(units),
                    "proposal_paragraph": "P0074/P0081-P0082/P0224",
                }
            )
    return {
        "ok": all(report.ok for report in reports) and not cross_recording_issues,
        "recordings": [report.to_dict() for report in reports],
        "cross_recording_issues": cross_recording_issues,
    }


def _as_2d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    return array[:, None] if array.ndim == 1 else array


def _interpolate_with_gap_mask(
    source_time: np.ndarray,
    source_data: np.ndarray,
    target_time: np.ndarray,
    max_gap_sec: float | None,
    interpolation_mode: str = "causal_last_observation_hold",
    observation_cutoff_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if interpolation_mode not in {"causal_last_observation_hold", "offline_linear"}:
        raise ContractError(f"unknown interpolation_mode: {interpolation_mode}")
    if interpolation_mode == "offline_linear" and observation_cutoff_sec is not None:
        raise FutureLeakageError(
            "offline_linear interpolation is non-causal and cannot be used with a forecast observation cutoff"
        )
    source_time = np.asarray(source_time, dtype=float)
    source_data = _as_2d(source_data)
    finite_rows = np.isfinite(source_data).all(axis=1) & np.isfinite(source_time)
    if finite_rows.sum() < 2:
        return np.full((target_time.size, source_data.shape[1]), np.nan), np.zeros(target_time.size, dtype=bool)
    time = source_time[finite_rows]
    values = source_data[finite_rows]
    if interpolation_mode == "offline_linear":
        output = np.column_stack([np.interp(target_time, time, values[:, column]) for column in range(values.shape[1])])
        positions = np.searchsorted(time, target_time, side="left")
        left = np.clip(positions - 1, 0, time.size - 1)
        right = np.clip(positions, 0, time.size - 1)
        mask = (target_time >= time[0]) & (target_time <= time[-1])
        gaps = time[right] - time[left]
    else:
        # Causal last-observation hold: every target receives the most recent
        # finite source sample at or before it.  No right-side sample is read.
        positions = np.searchsorted(time, target_time, side="right") - 1
        valid_positions = positions >= 0
        safe_positions = np.clip(positions, 0, time.size - 1)
        output = values[safe_positions].copy()
        mask = valid_positions & (target_time <= time[-1])
        gaps = np.full(target_time.shape, np.inf, dtype=float)
        gaps[valid_positions] = target_time[valid_positions] - time[safe_positions[valid_positions]]
    if max_gap_sec is not None:
        if max_gap_sec <= 0:
            raise ContractError("max_gap_sec must be positive when supplied")
        mask &= gaps <= max_gap_sec + 1e-12
    output[~mask] = np.nan
    return output, mask


def synchronize_recording(
    recording: Recording,
    target_sample_rate_hz: float,
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
    observation_cutoff_sec: float | None = None,
    max_interpolation_gap_sec: float | None = None,
    interpolation_mode: str = "causal_last_observation_hold",
) -> SynchronizedSignals:
    """Put EEG and ECG on a common time grid without inventing coverage.

    The caller must provide a target rate and, for formal use, a frozen
    interpolation-gap rule.  The default causal last-observation hold never
    reads a source sample to the right of a target time.  ``offline_linear`` is
    available only without an observation cutoff and is therefore blocked for
    forecast-history construction.  No anti-aliasing claim is made here.
    """

    report = audit_recording(recording)
    if not report.ok:
        raise ContractError("Cannot synchronize an ineligible recording: " + "; ".join(issue.code for issue in report.errors))
    if not np.isfinite(target_sample_rate_hz) or target_sample_rate_hz <= 0:
        raise ContractError("target_sample_rate_hz must be finite and positive")
    eeg = recording.by_modality("eeg")
    ecg = recording.by_modality("ecg")
    overlap_start = max(max(channel.start_time_sec for channel in eeg), max(channel.start_time_sec for channel in ecg))
    overlap_end = min(min(channel.end_time_sec for channel in eeg), min(channel.end_time_sec for channel in ecg))
    start = overlap_start if start_time_sec is None else max(overlap_start, float(start_time_sec))
    end = overlap_end if end_time_sec is None else min(overlap_end, float(end_time_sec))
    if observation_cutoff_sec is not None:
        end = min(end, float(observation_cutoff_sec))
    if end <= start:
        raise ContractError("requested synchronization interval is empty")
    step = 1.0 / float(target_sample_rate_hz)
    time = np.arange(start, end + step * 0.25, step, dtype=float)
    eeg_arrays = []
    eeg_masks = []
    for channel in eeg:
        source_time = channel.start_time_sec + np.arange(channel.n_samples, dtype=float) / channel.sample_rate_hz
        values, mask = _interpolate_with_gap_mask(
            source_time,
            channel.data,
            time,
            max_interpolation_gap_sec,
            interpolation_mode=interpolation_mode,
            observation_cutoff_sec=observation_cutoff_sec,
        )
        eeg_arrays.append(values[:, 0] if values.shape[1] == 1 else values)
        eeg_masks.append(mask)
    ecg_channel = ecg[0]
    ecg_time = ecg_channel.start_time_sec + np.arange(ecg_channel.n_samples, dtype=float) / ecg_channel.sample_rate_hz
    ecg_values, ecg_mask = _interpolate_with_gap_mask(
        ecg_time,
        ecg_channel.data,
        time,
        max_interpolation_gap_sec,
        interpolation_mode=interpolation_mode,
        observation_cutoff_sec=observation_cutoff_sec,
    )
    eeg = np.column_stack([_as_2d(value) for value in eeg_arrays])
    # column_stack above can create (n, channels) or (n, channels, 1) for odd
    # inputs; normalize explicitly for a stable downstream contract.
    eeg = np.asarray(eeg, dtype=float).reshape(time.size, -1)
    combined_eeg_mask = np.logical_and.reduce(np.asarray(eeg_masks, dtype=bool), axis=0)
    return SynchronizedSignals(
        time,
        eeg,
        ecg_values[:, 0],
        combined_eeg_mask,
        ecg_mask,
        float(target_sample_rate_hz),
        interpolation_mode,
    )

