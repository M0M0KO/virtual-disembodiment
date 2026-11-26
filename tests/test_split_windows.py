from __future__ import annotations

import unittest

import numpy as np

from virtual_disembodiment.errors import ContractError, FutureLeakageError
from virtual_disembodiment.splits import grouped_participant_split
from virtual_disembodiment.windows import build_forecast_windows, extract_window


class Session:
    def __init__(self, participant_id: str, recording_id: str) -> None:
        self.participant_id = participant_id
        self.recording_id = recording_id


class SplitWindowTests(unittest.TestCase):
    def test_sessions_are_grouped_by_participant(self) -> None:
        sessions = [Session(f"P{i}", f"P{i}_S{j}") for i in range(6) for j in range(2)]
        split = grouped_participant_split(sessions, seed=7)
        for participant in {item.participant_id for item in sessions}:
            self.assertEqual(len({split.assignments[participant]}), 1)
        split.assert_disjoint()

    def test_unseeded_split_is_blocked(self) -> None:
        with self.assertRaises(ContractError):
            grouped_participant_split(["P1", "P2", "P3"], seed=None)

    def test_history_is_strictly_before_cutoff(self) -> None:
        time = np.arange(0.0, 100.0, 1.0)
        windows = build_forecast_windows(time, context_length_sec=60.0, origin_stride_sec=5.0)
        self.assertTrue(windows)
        first = windows[0]
        self.assertTrue(np.all(time[first.history_indices] < first.observation_cutoff_sec))
        extracted = extract_window(first, time, np.column_stack([time, time + 1]), time[:, None])
        for target_time in first.target_times_sec.values():
            self.assertGreaterEqual(target_time, first.observation_cutoff_sec)
        self.assertEqual(extracted["history_peripheral"].shape[0], first.history_indices.size)

    def test_future_history_is_rejected(self) -> None:
        time = np.arange(0.0, 100.0, 1.0)
        window = build_forecast_windows(time, context_length_sec=60.0, origin_stride_sec=5.0)[0]
        bad_indices = np.concatenate([window.history_indices, np.asarray([int(window.target_indices[1.0])])])
        bad = type(window)(window.participant_id, window.recording_id, window.origin_time_sec, window.observation_cutoff_sec, bad_indices, window.target_indices, window.target_times_sec)
        with self.assertRaises(FutureLeakageError):
            bad.assert_no_future_leakage(time)

    def test_window_parameters_require_finite_positive_values(self) -> None:
        time = np.arange(0.0, 100.0, 1.0)
        with self.assertRaises(ContractError):
            build_forecast_windows(time, context_length_sec=0.0)
        with self.assertRaises(ContractError):
            build_forecast_windows(time, context_length_sec=np.inf)
        with self.assertRaises(ContractError):
            build_forecast_windows(time, origin_stride_sec=0.0)
        with self.assertRaises(ContractError):
            build_forecast_windows(time, origin_stride_sec=np.nan)
        with self.assertRaises(ContractError):
            build_forecast_windows(time, forecast_horizons_sec=())
        with self.assertRaises(ContractError):
            build_forecast_windows(time, forecast_horizons_sec=(1.0, 1.0))
        with self.assertRaises(ContractError):
            build_forecast_windows(time, forecast_horizons_sec=(1.0, np.inf))

