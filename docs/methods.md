# Study design

The original validation benchmark used standardized future EEG features. Given a 60-second history, the model scored 1-, 5-, and 15-second EEG targets under a common specified dosing schedule. Future EEG and physiology were withheld. The primary continuous ranked probability score (CRPS) weights were 0.2, 0.3, and 0.5. The 5% relative improvement criterion compared the coupled model with the strongest preselected neural-history comparator. That criterion was unmet.

## Representation and history pathway

The matched development comparison used EEG-only, genuine-body, acquisition-mask, and matched-shuffled-body arms. Geometry and linear readouts used the same eligible windows where possible. Geometry could have fewer eligible participants under its prespecified window requirement. Geometry, diagnostic mean absolute error (MAE), and native forecast CRPS have distinct estimands.

The history intervention attenuated the ascending body-to-brain message to 0.5 during history encoding. The random-direction control used the same location and per-message displacement norm. Inputs, target masks, and stochastic innovations were paired. No rollout message was directly edited. The fixed ridge reader was trained on intact calibration states and applied to the evaluation conditions. Condition-matched readers used the corresponding calibration states at the same capacity and ridge penalty (lambda 1). The diagnostic targets were the last EEG feature token, the masked mean of the preceding 59 EEG tokens, and the masked mean of 60 slow-body tokens. Calibration and evaluation participants were disjoint.

## Direct intervention on EEG-readable state directions

The direct-state intervention used the five frozen predictors under directed history attenuation. A participant-weighted, masked ridge reader for current EEG was fit on directed states from the original calibration participants. Centering and scaling came from calibration. The four leading left singular vectors of the reader coefficients in native activation coordinates defined the target basis. The seed-specific random basis of equal rank was orthogonal to it.

The targeted edit removed half of the centered projection onto the target basis. At each evaluation window, the control edit was rescaled to match the targeted edit's Euclidean displacement. Both edits were applied to the final encoded brain state immediately before the first native forecast step and compared with no edit. History, dosing, stochastic innovations, and target masks were held fixed across conditions. The [archived operator implementation](../scripts/dev_eeg_subspace.py) includes the conditional rescue transformation. The forecast loss interval against the matched random control did not meet the rescue condition, and rescue was not run.

## Independent units and reporting

The main development comparison used 16 participants, 1,437 matched windows, and five technical seeds. Summaries are first formed within participants. The direct-state comparisons report mean paired effects and 95% intervals from 20,000 participant bootstrap resamples. Positive differences in MAE and CRPS indicate more error for the first condition. The direct edit targets four directions at one intervention strength and one forecast horizon. A small selective CRPS contrast under these conditions does not establish formal equivalence or global independence from EEG.
