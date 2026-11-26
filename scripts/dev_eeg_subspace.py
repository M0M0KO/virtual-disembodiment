"""Calibrate a linear EEG reader and construct matched state interventions."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Reader:
    origin: np.ndarray
    offset: np.ndarray
    weight: np.ndarray
    target: np.ndarray
    control: np.ndarray

    def predict(self, state):
        return (state - self.origin) @ self.weight + self.offset


def fit_reader(states, target, valid, people, rank=4, ridge=1.0, seed=3101):
    """Give each calibration participant equal total weight per output."""
    x, y = np.asarray(states, float), np.asarray(target, float)
    valid, people = np.asarray(valid, bool), np.asarray(people)
    if x.ndim != 2 or y.ndim != 2 or valid.shape != y.shape or len(x) != len(y) or people.shape != (len(x),):
        raise ValueError("Calibration state, target, mask and participant arrays must align")
    if not np.isfinite(x).all() or not np.isfinite(y[valid]).all() or rank < 1 or rank > min(y.shape[1], x.shape[1] // 2):
        raise ValueError("Invalid calibration values or subspace rank")
    _, group, count = np.unique(people, return_inverse=True, return_counts=True)
    w = 1.0 / count[group]
    origin = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x - origin) ** 2, axis=0, weights=w)).clip(1e-8)
    z = (x - origin) / sd
    weight = np.empty((x.shape[1], y.shape[1]))
    offset = np.empty(y.shape[1])
    for col in range(y.shape[1]):
        keep = valid[:, col]
        if not keep.any():
            raise ValueError("Unobserved EEG target in calibration")
        groups = group[keep]
        precision = 1.0 / np.bincount(groups, minlength=len(count))[groups]
        precision /= precision.mean()
        zz, yy = z[keep], y[keep, col]
        x_mean = np.average(zz, axis=0, weights=precision)
        y_mean = np.average(yy, weights=precision)
        zz = zz - x_mean
        regularized = zz.T @ (zz * precision[:, None]) + ridge * np.eye(zz.shape[1])
        coefficient = np.linalg.solve(regularized, zz.T @ ((yy - y_mean) * precision))
        weight[:, col] = coefficient / sd
        offset[col] = y_mean - x_mean @ coefficient
    directions, singular, _ = np.linalg.svd(weight, full_matrices=False)
    if singular[rank - 1] <= np.finfo(float).eps * singular[0] * max(weight.shape):
        raise ValueError("Calibration reader lacks the prespecified rank")
    target_basis = directions[:, :rank]
    random = np.random.default_rng(174000 + seed).standard_normal(target_basis.shape)
    random -= target_basis @ (target_basis.T @ random)
    control, _ = np.linalg.qr(random, mode="reduced")
    return Reader(origin, offset, weight, target_basis, control)


def matched_delta(candidate, reference, epsilon=1e-9):
    """Match the per-window norm without changing the candidate's direction."""
    magnitude = np.linalg.norm(candidate, axis=-1, keepdims=True)
    desired = np.linalg.norm(reference, axis=-1, keepdims=True)
    if np.any((magnitude < epsilon) & (desired > epsilon)):
        raise ValueError("A nonzero edit lacks a valid matched control direction")
    return candidate * (desired / np.maximum(magnitude, epsilon))


def edits(reader, state, amount=0.5):
    centered = state - reader.origin
    target = -amount * ((centered @ reader.target) @ reader.target.T)
    random = -((centered @ reader.control) @ reader.control.T)
    return target, matched_delta(random, target)


def rescue_edits(reader, attenuated, aligned_donor, earlier_donor):
    aligned = ((aligned_donor - attenuated) @ reader.target) @ reader.target.T
    earlier = ((earlier_donor - attenuated) @ reader.target) @ reader.target.T
    return aligned, matched_delta(earlier, aligned)
