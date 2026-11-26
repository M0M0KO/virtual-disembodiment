"""CPU-first research primitives for the Virtual Disembodiment proposal.

The package deliberately contains no data downloader, no model training entry
point, and no import-time dependency on MNE or PyTorch.  It is a reviewable
Stage-A implementation of contracts, leakage-safe windows, deterministic
linear dynamics, interventions, and proposal-linked metrics.
"""

from .errors import ContractError, FutureLeakageError, FormalRunBlocked
from .config import load_config, preflight_config
from .contract import (
    ChannelAuditReport,
    ChannelSignal,
    Recording,
    SynchronizedSignals,
    audit_batch,
    audit_recording,
    synchronize_recording,
)
from .splits import ParticipantSplit, grouped_participant_split
from .windows import ForecastWindow, build_forecast_windows, extract_window
from .model import CoupledLinearStateSpace, State
from .interventions import random_damage_masks
from .outputs import create_new_synthetic_output, resolve_new_synthetic_output

__all__ = [
    "ChannelAuditReport",
    "ChannelSignal",
    "ContractError",
    "CoupledLinearStateSpace",
    "ForecastWindow",
    "FormalRunBlocked",
    "FutureLeakageError",
    "ParticipantSplit",
    "Recording",
    "State",
    "SynchronizedSignals",
    "audit_batch",
    "audit_recording",
    "build_forecast_windows",
    "create_new_synthetic_output",
    "extract_window",
    "grouped_participant_split",
    "load_config",
    "preflight_config",
    "random_damage_masks",
    "resolve_new_synthetic_output",
    "synchronize_recording",
]

__version__ = "0.1.0"

