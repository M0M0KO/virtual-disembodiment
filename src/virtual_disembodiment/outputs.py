"""Safe destination checks for caller-requested synthetic smoke output."""

from __future__ import annotations

from pathlib import Path

from .errors import ContractError


_RESERVED_PROJECT_DIRS = ("data", "results", "operations", "docs")


def _has_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def resolve_new_synthetic_output(output: str | Path, project_root: str | Path) -> Path:
    """Resolve and validate a new output directory without creating it.

    The final path must not exist, be a dangling symlink, or contain a symlink
    component.  The resolved path may be outside the project, but it may never
    be the project root or a resolved descendant of ``data``, ``results``,
    ``operations`` or ``docs``.  These checks apply even when the final path or
    reserved directory has not been created yet.
    """

    root = Path(project_root).expanduser().resolve(strict=True)
    candidate = Path(output).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists() or candidate.is_symlink():
        raise ContractError(f"synthetic output must be a new directory: {candidate}")
    if _has_symlink_component(candidate):
        raise ContractError(f"synthetic output may not use a symlink component: {candidate}")
    try:
        resolved = candidate.resolve(strict=False)
    except RuntimeError as exc:
        raise ContractError(f"synthetic output path cannot be resolved: {candidate}") from exc

    if resolved == root:
        raise ContractError(
            "synthetic output cannot be the project root or a data/results/operations/docs subtree: "
            f"{resolved}"
        )
    reserved = tuple(root / name for name in _RESERVED_PROJECT_DIRS)
    if any(resolved == directory or directory in resolved.parents for directory in reserved):
        raise ContractError(
            "synthetic output cannot be the project root or a data/results/operations/docs subtree: "
            f"{resolved}"
        )
    return resolved


def create_new_synthetic_output(output: str | Path, project_root: str | Path) -> Path:
    """Atomically create a validated new output directory."""

    resolved = resolve_new_synthetic_output(output, project_root)
    try:
        resolved.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ContractError(f"synthetic output appeared before creation: {resolved}") from exc
    return resolved

