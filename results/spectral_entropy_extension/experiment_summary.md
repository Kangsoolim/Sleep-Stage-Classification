# Spectral entropy exploratory extension

This is a fixed feature-extension comparison on the accepted Sleep-EDF subjects 0–9, recording 1, and 9,913 30-s EEG epochs. The accepted five-fold subject-wise fold assignments and Balanced RF settings (`n_estimators=100`, `random_state=241`, `class_weight=balanced`) were reused.

## Pooled OOF results

| Feature set | Dimension | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Balanced RF / 10D band-power | 10 | 0.766367 | 0.706107 |
| Balanced RF / 12D + spectral entropy | 12 | 0.770100 | 0.708557 |

## Five-fold mean ± SD

| Feature set | Accuracy | Macro-F1 |
|---|---:|---:|
| 10D | 0.766106 ± 0.028885 | 0.699800 ± 0.025480 |
| 12D + entropy | 0.769832 ± 0.032587 | 0.702745 ± 0.018255 |

## Change (12D + entropy minus 10D)

| Metric | Delta |
|---|---:|
| accuracy | +0.003732 |
| macro f1 | +0.002450 |
| W f1 | -0.000576 |
| N1 f1 | +0.000762 |
| N2 f1 | +0.004409 |
| Stage 3+4 f1 | +0.011644 |
| REM f1 | -0.003989 |

## Per-class pooled OOF F1

| Stage | 10D | 12D + entropy | Delta |
|---|---:|---:|---:|
| W | 0.866723 | 0.866146 | -0.000576 |
| N1 | 0.356473 | 0.357234 | +0.000762 |
| N2 | 0.822190 | 0.826598 | +0.004409 |
| Stage 3+4 | 0.780442 | 0.792085 | +0.011644 |
| REM | 0.704710 | 0.700721 | -0.003989 |

## Interpretation

The extension result is descriptive for this fixed subject-independent setting. No hypothesis test or parameter search was conducted. The observed changes are reported as measured; they do not establish statistical significance or general effectiveness.

## Validation notes

- Recomputed Core 10D features exact match accepted cache: `True`; maximum absolute error 0, maximum relative error 0.
- 10D OOF predictions exact match accepted result: `True`.
- Accepted pooled baseline metric tolerance check: `True`.
- Five folds, 9,913 epochs, and one OOF prediction per epoch are validated in `validation_checks.json`.
