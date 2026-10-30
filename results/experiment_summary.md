# Grouped RF controlled comparison: acceptance summary

## Locked design

- Data: accepted 10-subject, night-1 cache; two EEG channels and the fixed 10 official band-power features.
- Splits: the saved phase-4 `GroupKFold` assignments were read unchanged. Each fold's two held-out subjects were shared by both models.
- Models: `RandomForestClassifier(n_estimators=100, random_state=241)` with only `class_weight` changed (`None` vs `"balanced"`). No tuning, subject/fold changes, feature changes, or exclusions were made.

## Overall OOF metrics

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Standard RF | 0.7689 | 0.6894 |
| Balanced RF | 0.7664 | 0.7061 |

## Per-class OOF F1

| Class | Standard RF | Balanced RF | Change (balanced - standard) |
|---|---:|---:|---:|
| Sleep stage W | 0.8512 | 0.8667 | +0.0156 |
| Sleep stage 1 | 0.2980 | 0.3565 | +0.0584 |
| Sleep stage 2 | 0.8333 | 0.8222 | -0.0111 |
| Stage 3+4 | 0.7710 | 0.7804 | +0.0095 |
| Sleep stage R | 0.6933 | 0.7047 | +0.0114 |

## Factual interpretation

- N1 F1 changed by +0.0584; REM F1 changed by +0.0114 when class weighting was applied.
- Overall Accuracy changed by -0.0025, while Macro-F1 changed by +0.0168.
- In this run, the lower Accuracy alongside higher N1 and REM F1 is an Accuracy-versus-minority-class-performance trade-off; it is reported here as an observed comparison result, not as a causal claim.
- These are held-out grouped-comparison results under one fixed data/split/model setting. They describe an association with class weighting, not a causal generalization beyond this experiment.

## Confusion matrices

Rows are true labels; columns are predicted labels. Numerical matrices are also stored in `results/confusion_matrices.csv`.
### Standard RF

| True \ Pred | W | N1 | N2 | Stage 3+4 | REM |
|---|---:|---:|---:|---:|---:|
| W | 1544 | 124 | 29 | 8 | 55 |
| N1 | 216 | 197 | 129 | 3 | 219 |
| N2 | 54 | 83 | 3635 | 300 | 240 |
| Stage 3+4 | 10 | 1 | 343 | 1126 | 3 |
| REM | 44 | 153 | 276 | 1 | 1120 |

### Balanced RF

| True \ Pred | W | N1 | N2 | Stage 3+4 | REM |
|---|---:|---:|---:|---:|---:|
| W | 1551 | 140 | 10 | 10 | 49 |
| N1 | 178 | 285 | 91 | 3 | 207 |
| N2 | 45 | 184 | 3357 | 435 | 291 |
| Stage 3+4 | 11 | 2 | 229 | 1237 | 4 |
| REM | 34 | 224 | 167 | 2 | 1167 |
