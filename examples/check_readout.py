"""Compare readout strategies on generated data with disjoint participant roles."""
import numpy as np

from scripts.dev_representation_metrics import diagnostic_w


def main():
    rng = np.random.default_rng(17)
    calibration = rng.normal(size=(64, 4))
    evaluation = rng.normal(size=(48, 4))
    calibration_people = np.repeat(["cal_a", "cal_b", "cal_c", "cal_d"], 16)
    evaluation_people = np.repeat(["eval_a", "eval_b", "eval_c"], 16)
    if set(calibration_people) & set(evaluation_people):
        raise ValueError("Calibration and evaluation people must be disjoint")

    # Exchange two coordinates to make a condition shift with unchanged target.
    order = [1, 0, 2, 3]
    calibration_target = calibration[:, :1]
    evaluation_target = evaluation[:, :1]
    valid = np.ones_like(calibration_target, dtype=bool)
    fixed = diagnostic_w(calibration, evaluation[:, order], calibration_target, valid, calibration_people)
    matched = diagnostic_w(calibration[:, order], evaluation[:, order], calibration_target, valid, calibration_people)
    fixed_mae = np.mean(np.abs(fixed - evaluation_target))
    matched_mae = np.mean(np.abs(matched - evaluation_target))
    if not matched_mae < fixed_mae:
        raise RuntimeError("Generated fixture did not recover the expected readout")
    print(f"Generated fixture MAE, fixed={fixed_mae:.4f}, condition-matched={matched_mae:.4f}")


if __name__ == "__main__":
    main()
