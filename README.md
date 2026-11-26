# Virtual Disembodiment

[中文](README.zh-CN.md) · [Code v0.1.0](https://github.com/M0M0KO/virtual-disembodiment/releases/tag/v0.1.0) · [Methods](docs/methods.md) · [Reproduction](docs/reproduction.md)

Code accompanying *Decodable physiological information and predictive dependence in a brain–body model of propofol sedation*.

## Research question

How does body input change the internal representations of an EEG forecasting model? Does the information accessible to a reader correspond to the information used by the model's native forecast?

We study EEG, peripheral physiology and a specified dosing scenario during propofol sedation. We compare internal information organization, EEG accessibility and future-EEG prediction. Readers are fit on calibration participants and scored on distinct evaluation participants.

## Main findings

| Question | Result |
| --- | --- |
| Does body input change the representation? | Genuine body input improved linear access to slow physiological features. The original future-EEG comparison did not meet the prespecified 5% improvement criterion. |
| Can EEG accessibility recover after a history-pathway change? | Refitting an equal-capacity reader reduced current-EEG diagnostic MAE under directed attenuation by a mean **12.59%** relative to the fixed reader, with a **95% interval of 10.82–14.42%**, across **16 matched evaluation participants**. |
| Does the native forecast depend on EEG-readable directions? | Attenuating four calibration-defined EEG-readable directions increased current-EEG readout MAE by **0.07635** relative to a rank- and displacement-matched random control. The corresponding 1-s forecast CRPS difference was **+0.00000349**, with a **95% interval from −0.00001698 to +0.00002374**. |

Readout MAE measures access to recent observed EEG in the model state; CRPS measures the native probabilistic forecast of future EEG. The 12.59% reduction describes readout recovery. The direct-state CRPS contrast spans zero; formal equivalence remains untested.

The [direct-state source table](source_tables/eeg_subspace_effects.csv) covers 16 participants, 1,437 matched windows and five frozen model seeds. Seeds are averaged within participants before paired participant-level effects and intervals are computed.

## Quick start

Use Python 3.10 or later. Download the source, enter the repository root, and create and activate a virtual environment before running:

```bash
python -m pip install -e ".[dynamics]"
python -m unittest discover -s tests -v
python -m examples.check_readout
python -m examples.check_subspace
```

| Example | What it checks | Output |
| --- | --- | --- |
| `examples.check_readout` | Fixed and condition-matched readers on generated data with disjoint calibration and evaluation participants | MAE for both readers |
| `examples.check_subspace` | Subspace rank, orthogonality and matched per-state displacement norms | Maximum displacement-norm difference |

Both examples verify reader and intervention operators without recordings or model checkpoints.

## Reproduction coverage

| Analysis | Included here | Additional study inputs |
| --- | --- | --- |
| Future-EEG prediction | Causal features, windows, native dynamics and probabilistic scoring | Source signals, frozen participant roles, dosing scenario, checkpoints and evaluation arrays |
| Representation geometry and reader recovery | Geometry and ridge utilities, generated-data examples | Condition-specific states, matched windows and participant-level results |
| EEG-readable subspace intervention | Subspace fitting, state edits, generated-data example and four aggregate effects | Five frozen models, evaluation arrays and per-window intervention records |

The original study scores 1-, 5- and 15-s EEG targets with CRPS weights of 0.2, 0.3 and 0.5. Development readouts and model interventions are exploratory analyses; the original validation conclusion and 5% criterion are unchanged. See the [reproduction guide](docs/reproduction.md) for methods and input requirements.

## Repository structure

| Location | Contents |
| --- | --- |
| [`docs/`](docs/) | Methods, reproduction and data availability |
| [`src/virtual_disembodiment/`](src/virtual_disembodiment/) | Features, windows, models, interventions and scoring |
| [`scripts/dev_representation_metrics.py`](scripts/dev_representation_metrics.py) | Representation geometry and diagnostic readers |
| [`scripts/dev_eeg_subspace.py`](scripts/dev_eeg_subspace.py) | EEG-readable subspaces and state edits |
| [`examples/`](examples/) · [`tests/`](tests/) | Generated-data examples and software checks |
| [`source_tables/`](source_tables/) | Aggregate result tables |

## Data

Source recordings are available from [DOSE-I v1](https://doi.org/10.5281/zenodo.18483292), under the provider's data-use terms. This repository distributes research code and aggregate source tables. Participant-level recordings, trained models and the complete study input manifests are not included in this version. See [data availability](docs/data-availability.md) for details.

## Citation

Cite the study using the manuscript title above. When citing code, include the [repository URL](https://github.com/M0M0KO/virtual-disembodiment) and the version used; the existing research-code release is `v0.1.0`.

## License

Code and software documentation are available under the [MIT License](LICENSE). Source recordings retain the provider's terms; data and software licenses apply separately.
