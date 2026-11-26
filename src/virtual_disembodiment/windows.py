"""Time-based forecast windows with explicit observation cutoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .errors import ContractError, FutureLeakageError


@dataclass(frozen=True)
class ForecastWindow:
    participant_id: str
    recording_id: str
    origin_time_sec: float
    observation_cutoff_sec: float
    history_indices: np.ndarray
    target_indices: Mapping[float, int]
    target_times_sec: Mapping[float, float]

    def assert_no_future_leakage(self, time_sec: np.ndarray) -> None:
        times = np.asarray(time_sec, dtype=float)
        history_times = times[self.history_indices]
        if history_times.size and not np.all(history_times < self.observation_cutoff_sec):
            raise FutureLeakageError("history contains a sample at or after the observation cutoff")
        for horizon, index in self.target_indices.items():
            if times[index] < self.observation_cutoff_sec:
                raise FutureLeakageError(f"target for {horizon}s is before the observation cutoff")


def build_forecast_windows(
    time_sec: np.ndarray,
    *,
    participant_id: str = "synthetic",
    recording_id: str = "synthetic-recording",
    context_length_sec: float = 60.0,
    forecast_horizons_sec: tuple[float, ...] = (1.0, 5.0, 15.0),
    origin_stride_sec: float | None = None,
    min_history_fraction: float = 1.0,
) -> tuple[ForecastWindow, ...]:
    """Create windows from timestamps rather than sample indices.

    The 60-second history and 1/5/15-second horizons are proposal defaults
    (P0105/P0224).  A target is always selected strictly after the origin, and
    the returned history contains only timestamps strictly before it.
    """

    times = np.asarray(time_sec, dtype=float)
    if times.ndim != 1 or times.size < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ContractError("time_sec must be a finite, strictly increasing 1-D array")
    try:
        context = float(context_length_sec)
        horizons = tuple(float(horizon) for horizon in forecast_horizons_sec)
    except (TypeError, ValueError) as exc:
        raise ContractError("context_length_sec and forecast_horizons_sec must be numeric") from exc
    if not np.isfinite(context) or context <= 0:
        raise ContractError("context and horizons must be positive")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ContractError("forecast_horizons_sec must be non-empty and contain no duplicates")
    if any(not np.isfinite(horizon) or horizon <= 0 for horizon in horizons):
        raise ContractError("forecast horizons must be finite and positive")
    if not 0 < min_history_fraction <= 1:
        raise ContractError("min_history_fraction must be in (0, 1]")
    if origin_stride_sec is None:
        stride = float(times[1] - times[0])
    else:
        try:
            stride = float(origin_stride_sec)
        except (TypeError, ValueError) as exc:
            raise ContractError("origin_stride_sec must be numeric") from exc
    if not np.isfinite(stride) or stride <= 0:
        raise ContractError("origin_stride_sec must be positive")
    max_horizon = max(horizons)

    windows: list[ForecastWindow] = []
    next_origin = times[0] + context
    while next_origin + max_horizon <= times[-1] + 1e-12:
        cutoff = float(next_origin)
        history = np.flatnonzero((times < cutoff) & (times >= cutoff - context))
        if history.size:
            observed_duration = times[history[-1]] - times[history[0]]
        else:
            observed_duration = 0.0
        if observed_duration + (times[1] - times[0]) >= context * min_history_fraction:
            targets: dict[float, int] = {}
            target_times: dict[float, float] = {}
            for horizon in horizons:
                requested = cutoff + horizon
                index = int(np.searchsorted(times, requested, side="left"))
                if index >= times.size:
                    break
                targets[float(horizon)] = index
                target_times[float(horizon)] = float(times[index])
            if len(targets) == len(horizons):
                window = ForecastWindow(
                    participant_id=participant_id,
                    recording_id=recording_id,
                    origin_time_sec=cutoff,
                    observation_cutoff_sec=cutoff,
                    history_indices=history,
                    target_indices=targets,
                    target_times_sec=target_times,
                )
                window.assert_no_future_leakage(times)
                windows.append(window)
        next_origin += stride
    return tuple(windows)


def extract_window(
    window: ForecastWindow,
    time_sec: np.ndarray,
    neural: np.ndarray,
    peripheral: np.ndarray | None = None,
) -> dict[str, np.ndarray | dict[float, np.ndarray] | float]:
    """Extract history and future neural targets; peripheral targets are never returned."""

    times = np.asarray(time_sec, dtype=float)
    neural_array = np.asarray(neural)
    if neural_array.shape[0] != times.size:
        raise ContractError("neural and time arrays have different lengths")
    window.assert_no_future_leakage(times)
    history_neural = neural_array[window.history_indices]
    result: dict[str, np.ndarray | dict[float, np.ndarray] | float] = {
        "history_time_sec": times[window.history_indices],
        "history_neural": history_neural,
        "observation_cutoff_sec": window.observation_cutoff_sec,
        "target_neural": {horizon: neural_array[index] for horizon, index in window.target_indices.items()},
    }
    if peripheral is not None:
        peripheral_array = np.asarray(peripheral)
        if peripheral_array.shape[0] != times.size:
            raise ContractError("peripheral and time arrays have different lengths")
        # Only the as-of history is exposed to the caller.  This makes it hard
        # to accidentally feed actual future body observations into a rescue.
        result["history_peripheral"] = peripheral_array[window.history_indices]
    return result

