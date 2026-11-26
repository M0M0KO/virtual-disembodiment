"""Virtual disembodiment, damage controls, and rescue message contracts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .errors import ContractError, FutureLeakageError


def validate_gate_gain(value: object, name: str = "gate gain") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{name} must be a finite numeric value")
    converted = float(value)
    if not np.isfinite(converted):
        raise ContractError(f"{name} must be a finite numeric value")
    return converted


@dataclass(frozen=True)
class VirtualIntervention:
    name: str
    ascending_gain: float = 1.0
    descending_gain: float = 1.0
    phase_shift_steps: int = 0
    remove_peripheral_measurements: bool = False
    proposal_paragraph: str = "P0112-P0114"

    def __post_init__(self) -> None:
        validate_gate_gain(self.ascending_gain, "ascending_gain")
        validate_gate_gain(self.descending_gain, "descending_gain")


def proposal_gate_sweep() -> tuple[float, ...]:
    """The prespecified gain sequence from P0114."""

    return (1.0, 0.75, 0.5, 0.25, 0.0)


def apply_message_intervention(messages: np.ndarray, intervention: VirtualIntervention) -> np.ndarray:
    values = np.asarray(messages, dtype=float)
    if values.ndim != 2:
        raise ContractError("peripheral messages must be shape (time, features)")
    output = values.copy()
    if intervention.remove_peripheral_measurements:
        output.fill(0.0)
    if not isinstance(intervention.phase_shift_steps, (int, np.integer)) or isinstance(intervention.phase_shift_steps, bool):
        raise ContractError("phase_shift_steps must be an integer")
    if intervention.phase_shift_steps < 0:
        raise ContractError(
            "negative phase shifts are non-causal; use a separately labelled offline_noncausal_control"
        )
    if intervention.phase_shift_steps:
        shift = int(intervention.phase_shift_steps)
        delayed = np.zeros_like(output)
        if shift < output.shape[0]:
            delayed[shift:] = output[:-shift]
        output = delayed
    return output


def require_safe_rescue_source(source: str, allow_oracle: bool = False) -> None:
    """Reject actual future participant physiology unless explicitly labelled oracle."""

    allowed = {"synthetic", "development_replay", "participant_swapped", "generated_controller"}
    if source == "actual_future_observation" and not allow_oracle:
        raise FutureLeakageError(
            "actual future peripheral observations are forbidden in autonomous rescue; "
            "use an explicitly labelled oracle-assimilation analysis"
        )
    if source not in allowed and not (source == "actual_future_observation" and allow_oracle):
        raise ContractError(f"unrecognized rescue source: {source}")


def validate_future_messages(message_times_sec: np.ndarray, observation_cutoff_sec: float) -> None:
    times = np.asarray(message_times_sec, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ContractError("message_times_sec must be a finite 1-D array")
    if np.any(times <= observation_cutoff_sec):
        raise FutureLeakageError("rescue forcing messages must begin after the observation cutoff")


def random_damage_masks(
    dimension: int,
    *,
    n_masks: int = 100,
    affected_units: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Generate prespecified random masks for a damage-control family (P0118).

    The function does not tune masks against a protected result.  Matching a
    target activation/forecast decrement remains an explicit development-stage
    operation for the caller.
    """

    if dimension <= 0 or n_masks <= 0 or not 0 < affected_units <= dimension:
        raise ContractError("damage-mask dimensions are invalid")
    rng = np.random.default_rng(seed)
    masks = np.zeros((n_masks, dimension), dtype=bool)
    for row in range(n_masks):
        masks[row, rng.choice(dimension, size=affected_units, replace=False)] = True
    return masks

