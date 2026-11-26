"""Verify rank, orthogonality and per-state displacement on generated arrays."""
import numpy as np

from scripts.dev_eeg_subspace import edits, fit_reader


def main():
    rng = np.random.default_rng(19)
    calibration = rng.normal(size=(96, 12))
    evaluation = rng.normal(size=(48, 12))
    targets = calibration[:, :4]
    people = np.repeat(np.arange(6), 16)
    reader = fit_reader(calibration, targets, np.ones_like(targets, dtype=bool), people)

    np.testing.assert_allclose(reader.target.T @ reader.target, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(reader.control.T @ reader.control, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(reader.target.T @ reader.control, np.zeros((4, 4)), atol=1e-12)
    targeted, control = edits(reader, evaluation)
    targeted_norm = np.linalg.norm(targeted, axis=1)
    np.testing.assert_allclose(targeted_norm, np.linalg.norm(control, axis=1), atol=1e-12)
    baseline = reader.predict(evaluation)
    target_change = np.mean(np.abs(reader.predict(evaluation + targeted) - baseline))
    control_change = np.mean(np.abs(reader.predict(evaluation + control) - baseline))
    if not target_change > control_change:
        raise RuntimeError("The targeted edit did not affect the synthetic reader")
    norm_error = np.max(np.abs(targeted_norm - np.linalg.norm(control, axis=1)))
    print(f"Generated fixture max displacement error: {norm_error:.3g}")


if __name__ == "__main__":
    main()
