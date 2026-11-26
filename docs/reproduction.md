# Reproduction guide

## Software verification

The root README commands exercise the tested portable code paths. Unit tests cover participant grouping, causal windows, paired interventions and score alignment. The generated readout example uses disjoint calibration and evaluation participants. The generated direct-state example checks that the target and control bases are orthogonal, each edit has the same per-state norm, and the targeted change alters the fitted reader. These checks use no clinical data or model checkpoints.

## Map the manuscript to the released materials

| Manuscript analysis | This release |
| --- | --- |
| EEG forecast fidelity and timing | Native dynamics, causal signal/window utilities, and forecast scoring code. The original analysis uses additional frozen participants, checkpoints and output bindings. |
| Genuine bodily input and condition-matched reader recovery | Archived geometry and ridge utilities with a generated verification example. Participant-level activation and score archives remain in the research environment. |
| Direct EEG-readable subspace intervention | Archived `fit_reader` and `edits` source, fixture example, and verified aggregate paired contrasts corresponding to the manuscript's direct-state effect panels. |

The [direct-state source table](../source_tables/eeg_subspace_effects.csv) includes the four pairwise effect and interval rows from the completed development summary, without participant records. Its aggregate parent summary has SHA-256 `c6ed8a8e50f3c4628d54bbe6fd77ef8f1e352cc0cd38ca11cdbec1fc206aaea3`. The [archived subspace source](../scripts/dev_eeg_subspace.py) has SHA-256 `d6b506ffd877e5238941668c84ea340162e39b9d6776c0d9bca68b8bd1bf1d21`.

A complete raw-recording reproduction would additionally require the admitted source waves, the original participant roles, preprocessing and forcing bindings, the matched-window manifest, the authorized five frozen checkpoints and the evaluation arrays. The public code is not an entry point for the protected test. Participant-level figure panels are outside this snapshot.
