"""Configuration loading and conservative preflight gates.

The JSON configuration keeps proposal defaults and unresolved launch inputs
explicit.  A synthetic run may use an explicit caller-owned seed and synthetic
signals; a formal run is blocked while any path in ``formal_required_paths`` is
unknown or while the config still declares the Stage-A synthetic-only mode.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ContractError, FormalRunBlocked


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config without resolving data paths or touching data."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError("Configuration root must be a JSON object")
    return value


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreflightReport:
    mode: str
    ok: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "config_sha256": self.digest,
        }

    def require_ok(self) -> None:
        if not self.ok:
            raise FormalRunBlocked("Formal run blocked: " + "; ".join(self.blockers))


def _as_tuple(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(values)


def preflight_config(config: dict[str, Any], mode: str = "plan") -> PreflightReport:
    """Validate config structure and report unresolved formal inputs.

    ``plan`` and ``synthetic`` are intentionally non-blocking for unknown
    acquisition/model values.  ``formal`` requires every path listed by the
    config and refuses the proposal's Stage-A ``synthetic_only`` mode.
    """

    if mode not in {"plan", "synthetic", "formal"}:
        raise ContractError(f"Unsupported preflight mode: {mode}")

    blockers: list[str] = []
    warnings: list[str] = []
    if config.get("schema_version") != 1:
        blockers.append("schema_version must equal 1")
    if config.get("project") != "virtual-disembodiment":
        blockers.append("project must be virtual-disembodiment")

    if config.get("analysis_mode") != "synthetic_only":
        warnings.append("analysis_mode is not synthetic_only; review the scope before use")

    for path in config.get("formal_required_paths", []):
        value = _get_path(config, path)
        if value is None or value == "" or value == []:
            blockers.append(f"unknown required value: {path}")

    # These are structural safety checks even for a synthetic plan.
    split = config.get("participant_split", {})
    ratios = split.get("ratios") if isinstance(split, dict) else None
    if ratios is not None:
        try:
            if len(ratios) != 3 or abs(sum(float(x) for x in ratios) - 1.0) > 1e-8:
                blockers.append("participant_split.ratios must contain three values summing to 1")
        except (TypeError, ValueError):
            blockers.append("participant_split.ratios must be numeric")

    horizon = config.get("windowing", {})
    horizons = horizon.get("forecast_horizons_sec") if isinstance(horizon, dict) else None
    if horizons != [1, 5, 15]:
        warnings.append("forecast_horizons_sec differs from the proposal's P0105/P0224 1,5,15 s task")

    # The proposal explicitly forbids treating ECG-derived respiration as an
    # independent respiratory measurement (P0043/P0082).
    derived = config.get("channels", {}).get("respiration_derived", {})
    if isinstance(derived, dict) and derived.get("independent_evidence") is not False:
        blockers.append("channels.respiration_derived.independent_evidence must be false")

    if mode == "formal" and config.get("analysis_mode") == "synthetic_only":
        blockers.append("formal mode is disabled until acquisition/provenance and model freeze are completed")
    if mode == "formal" and config.get("formal_run_enabled") is not True:
        blockers.append("formal_run_enabled must be explicitly true after the freeze review")

    if mode == "synthetic":
        warnings.append("synthetic mode never reads raw data and cannot establish proposal evidence")
    if mode == "plan":
        warnings.append("plan mode is read-only and does not validate a real manifest")

    # In plan/synthetic mode, structural blockers still indicate a bad config;
    # unresolved formal inputs are reported but do not block a smoke test.
    unresolved_prefix = "unknown required value: "
    structural = [item for item in blockers if not item.startswith(unresolved_prefix)]
    if mode != "formal":
        warnings.extend(item for item in blockers if item.startswith(unresolved_prefix))
        blockers = structural

    return PreflightReport(
        mode=mode,
        ok=not blockers,
        blockers=_as_tuple(blockers),
        warnings=_as_tuple(warnings),
        digest=config_digest(config),
    )

