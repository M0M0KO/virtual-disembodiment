"""Participant-grouped split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .errors import ContractError


@dataclass(frozen=True)
class ParticipantSplit:
    assignments: Mapping[str, str]
    ratios: tuple[float, float, float]
    seed: int

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted(self.assignments))

    def members(self, split_name: str) -> tuple[str, ...]:
        return tuple(sorted(pid for pid, name in self.assignments.items() if name == split_name))

    def assert_disjoint(self) -> None:
        groups = [set(self.members(name)) for name in ("development", "validation", "test")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ContractError("participant groups overlap")


def _participant_ids(records_or_ids: Iterable[object]) -> list[str]:
    values: list[str] = []
    for item in records_or_ids:
        value = getattr(item, "participant_id", item)
        if not isinstance(value, str) or not value.strip():
            raise ContractError("every split unit needs a non-empty participant_id")
        values.append(value)
    return values


def grouped_participant_split(
    records_or_ids: Sequence[object],
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int | None = None,
) -> ParticipantSplit:
    """Assign all sessions/recordings from one participant to one split.

    The default ratios and the participant grouping are from P0070/P0040.  A
    seed is mandatory so that a formal assignment is reproducible; synthetic
    callers should pass their own explicit seed.
    """

    if seed is None:
        raise ContractError("split seed is unresolved; formal grouping cannot proceed")
    if len(ratios) != 3 or any(float(ratio) <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ContractError("ratios must be three positive values summing to 1")
    ids = _participant_ids(records_or_ids)
    unique = sorted(set(ids))
    if len(unique) < 3:
        raise ContractError("at least three participants are needed for development/validation/test")

    rng = np.random.default_rng(int(seed))
    order = list(unique)
    rng.shuffle(order)
    n = len(order)
    counts = np.floor(np.asarray(ratios, dtype=float) * n).astype(int)
    # Keep all three partitions non-empty when enough participants exist.
    for index in range(3):
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ContractError("cannot create three non-empty participant partitions")
            counts[donor] -= 1
            counts[index] = 1
    while counts.sum() < n:
        fractional = np.asarray(ratios) * n - counts
        counts[int(np.argmax(fractional))] += 1
    while counts.sum() > n:
        donor = int(np.argmax(counts - 1))
        counts[donor] -= 1

    names = ("development", "validation", "test")
    assignments: dict[str, str] = {}
    cursor = 0
    for name, count in zip(names, counts.tolist()):
        for participant_id in order[cursor : cursor + count]:
            assignments[participant_id] = name
        cursor += count
    result = ParticipantSplit(assignments=assignments, ratios=tuple(float(x) for x in ratios), seed=int(seed))
    result.assert_disjoint()
    if set(assignments) != set(unique):
        raise ContractError("split assignment omitted a participant")
    return result

