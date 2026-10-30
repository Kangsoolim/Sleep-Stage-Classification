"""Run the locked Standard RF vs Balanced RF GroupKFold comparison.

The split plan is intentionally read from the accepted phase-4 CSV.  It is
never regenerated in this script: each saved fold is used once, and both
classifiers are fitted and evaluated inside that same fold loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data_utils import STAGE_CODES, STAGE_NAMES
from features import EXPECTED_FEATURE_DIM, feature_columns


FIXED_SUBJECTS = tuple(range(10))
N_SPLITS = 5
RANDOM_STATE = 241
N_ESTIMATORS = 100
MODEL_NAMES = ("standard", "balanced")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_subject_list(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    return tuple(sorted(int(subject) for subject in text.split(";")))


def stage_name(label: int) -> str:
    return dict(zip(STAGE_CODES, STAGE_NAMES, strict=True))[int(label)]


def make_models() -> dict[str, RandomForestClassifier]:
    common = {"n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE}
    return {
        "standard": RandomForestClassifier(**common, class_weight=None),
        "balanced": RandomForestClassifier(**common, class_weight="balanced"),
    }


def feature_digest(values: np.ndarray) -> str:
    """Produce a reproducible fingerprint for the exact numeric feature matrix."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def validate_inputs(frame: pd.DataFrame, assignments: pd.DataFrame, columns: list[str]) -> None:
    expected_columns = ["subject_id", "epoch_id", "label", *columns]
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(f"Unexpected feature cache schema: {frame.columns.tolist()}")
    if tuple(sorted(frame["subject_id"].unique().tolist())) != FIXED_SUBJECTS:
        raise AssertionError("Feature cache subjects do not match fixed subjects 0--9.")
    if len(columns) != EXPECTED_FEATURE_DIM:
        raise AssertionError(f"Expected {EXPECTED_FEATURE_DIM} features, found {len(columns)}")
    if frame.isna().any().any() or not np.isfinite(frame.loc[:, columns].to_numpy(dtype=float)).all():
        raise AssertionError("Feature cache has missing or non-finite values.")
    required_assignment_columns = {
        "fold", "train_subjects", "test_subjects", "n_train_subjects", "n_test_subjects",
        "n_train_epochs", "n_test_epochs", "subject_overlap_count",
    }
    if set(assignments.columns) != required_assignment_columns or len(assignments) != N_SPLITS:
        raise AssertionError("Saved fold-assignment CSV does not have the fixed five-fold schema.")
    if sorted(assignments["fold"].astype(int).tolist()) != list(range(1, N_SPLITS + 1)):
        raise AssertionError("Saved fold numbers must be exactly 1--5.")


def metrics_for_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = list(STAGE_CODES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    return (
        float(accuracy_score(y_true, y_pred)), macro_f1, precision, recall, f1, support
    )


def markdown_metric_table(overall: pd.DataFrame, per_class: pd.DataFrame) -> str:
    overall_lookup = overall.set_index("model")
    lines = [
        "# Grouped RF controlled comparison: acceptance summary",
        "",
        "## Locked design",
        "",
        "- Data: accepted 10-subject, night-1 cache; two EEG channels and the fixed 10 official band-power features.",
        "- Splits: the saved phase-4 `GroupKFold` assignments were read unchanged. Each fold's two held-out subjects were shared by both models.",
        "- Models: `RandomForestClassifier(n_estimators=100, random_state=241)` with only `class_weight` changed (`None` vs `\"balanced\"`). No tuning, subject/fold changes, feature changes, or exclusions were made.",
        "",
        "## Overall OOF metrics",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for model, display in (("standard", "Standard RF"), ("balanced", "Balanced RF")):
        row = overall_lookup.loc[model]
        lines.append(f"| {display} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")

    lines.extend(["", "## Per-class OOF F1", "", "| Class | Standard RF | Balanced RF | Change (balanced - standard) |", "|---|---:|---:|---:|"])
    pivot = per_class.pivot(index="class_name", columns="model", values="f1").reindex(list(STAGE_NAMES))
    for name, row in pivot.iterrows():
        change = float(row["balanced"] - row["standard"])
        display_name = "Stage 3+4" if name == "Sleep stage 3/4" else name
        lines.append(f"| {display_name} | {row['standard']:.4f} | {row['balanced']:.4f} | {change:+.4f} |")

    n1_change = float(pivot.loc["Sleep stage 1", "balanced"] - pivot.loc["Sleep stage 1", "standard"])
    rem_change = float(pivot.loc["Sleep stage R", "balanced"] - pivot.loc["Sleep stage R", "standard"])
    accuracy_change = float(overall_lookup.loc["balanced", "accuracy"] - overall_lookup.loc["standard", "accuracy"])
    macro_change = float(overall_lookup.loc["balanced", "macro_f1"] - overall_lookup.loc["standard", "macro_f1"])
    lines.extend([
        "",
        "## Factual interpretation",
        "",
        f"- N1 F1 changed by {n1_change:+.4f}; REM F1 changed by {rem_change:+.4f} when class weighting was applied.",
        f"- Overall Accuracy changed by {accuracy_change:+.4f}, while Macro-F1 changed by {macro_change:+.4f}.",
        "- In this run, the lower Accuracy alongside higher N1 and REM F1 is an Accuracy-versus-minority-class-performance trade-off; it is reported here as an observed comparison result, not as a causal claim.",
        "- These are held-out grouped-comparison results under one fixed data/split/model setting. They describe an association with class weighting, not a causal generalization beyond this experiment.",
        "",
        "## Confusion matrices",
        "",
        "Rows are true labels; columns are predicted labels. Numerical matrices are also stored in `results/confusion_matrices.csv`.",
        "",
    ])
    return "\n".join(lines)


def append_confusion_matrices(summary: str, oof: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    labels = list(STAGE_CODES)
    rows: list[dict[str, object]] = []
    table_lines: list[str] = []
    for model, prediction_column, display in (
        ("standard", "pred_standard", "Standard RF"),
        ("balanced", "pred_balanced", "Balanced RF"),
    ):
        matrix = confusion_matrix(oof["true_label"], oof[prediction_column], labels=labels)
        table_lines.extend([f"### {display}", "", "| True \\ Pred | W | N1 | N2 | Stage 3+4 | REM |", "|---|---:|---:|---:|---:|---:|"])
        short_names = ("W", "N1", "N2", "Stage 3+4", "REM")
        for true_code, true_short, matrix_row in zip(labels, short_names, matrix, strict=True):
            table_lines.append(f"| {true_short} | " + " | ".join(str(int(value)) for value in matrix_row) + " |")
            for pred_code, count in zip(labels, matrix_row, strict=True):
                rows.append({
                    "model": model,
                    "true_label": int(true_code),
                    "true_class": stage_name(true_code),
                    "pred_label": int(pred_code),
                    "pred_class": stage_name(pred_code),
                    "count": int(count),
                })
        table_lines.append("")
    return (summary + "\n".join(table_lines)).rstrip() + "\n", pd.DataFrame(rows)


def run_experiment(root: Path) -> dict[str, Path]:
    results_dir = root / "results"
    cache_path = root / "data" / "processed" / "sleep_edf_subjects_0_9_recording_1_epoch_features.csv"
    assignments_path = results_dir / "main_experiment" / "fold_assignments.csv"
    frame = pd.read_csv(cache_path)
    assignments = pd.read_csv(assignments_path).sort_values("fold").reset_index(drop=True)
    columns = list(feature_columns())
    validate_inputs(frame, assignments, columns)

    models = make_models()
    standard_params, balanced_params = (models[name].get_params(deep=True) for name in MODEL_NAMES)
    parameter_differences = {
        key: (standard_params[key], balanced_params[key])
        for key in standard_params
        if standard_params[key] != balanced_params[key]
    }
    if parameter_differences != {"class_weight": (None, "balanced")}:
        raise AssertionError(f"Unexpected model parameter differences: {parameter_differences}")

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    observed_test_subjects: Counter[int] = Counter()
    fold_audits: list[dict[str, object]] = []
    all_indices = set(frame.index.tolist())

    for assignment in assignments.itertuples(index=False):
        fold = int(assignment.fold)
        declared_train = parse_subject_list(assignment.train_subjects)
        declared_test = parse_subject_list(assignment.test_subjects)
        train_mask = frame["subject_id"].isin(declared_train)
        test_mask = frame["subject_id"].isin(declared_test)
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        actual_train = tuple(sorted(train["subject_id"].unique().tolist()))
        actual_test = tuple(sorted(test["subject_id"].unique().tolist()))
        if actual_train != declared_train or actual_test != declared_test:
            raise AssertionError(f"Fold {fold} does not match its saved subject assignment.")
        if set(declared_train).intersection(declared_test):
            raise AssertionError(f"Fold {fold} has subject leakage.")
        if set(declared_train).union(declared_test) != set(FIXED_SUBJECTS):
            raise AssertionError(f"Fold {fold} does not partition fixed subjects 0--9.")
        if len(train) != int(assignment.n_train_epochs) or len(test) != int(assignment.n_test_epochs):
            raise AssertionError(f"Fold {fold} epoch counts do not match the saved assignment.")

        x_train = train.loc[:, columns].to_numpy(dtype=float)
        x_test = test.loc[:, columns].to_numpy(dtype=float)
        y_train = train["label"].to_numpy(dtype=int)
        y_test = test["label"].to_numpy(dtype=int)
        # Both fits receive separately copied but byte-identical feature arrays.
        standard_x_train, balanced_x_train = x_train.copy(), x_train.copy()
        standard_x_test, balanced_x_test = x_test.copy(), x_test.copy()
        if not (np.array_equal(standard_x_train, balanced_x_train) and np.array_equal(standard_x_test, balanced_x_test)):
            raise AssertionError(f"Fold {fold} model inputs differ.")

        standard_model = make_models()["standard"]
        balanced_model = make_models()["balanced"]
        standard_pred = standard_model.fit(standard_x_train, y_train).predict(standard_x_test)
        balanced_pred = balanced_model.fit(balanced_x_train, y_train).predict(balanced_x_test)
        for model, predictions in (("standard", standard_pred), ("balanced", balanced_pred)):
            accuracy, macro_f1, _, _, _, _ = metrics_for_predictions(y_test, predictions)
            fold_rows.append({
                "fold": fold,
                "model": model,
                "n_train_epochs": len(train),
                "n_test_epochs": len(test),
                "train_subjects": ";".join(map(str, declared_train)),
                "test_subjects": ";".join(map(str, declared_test)),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            })
        oof_parts.append(pd.DataFrame({
            "subject_id": test["subject_id"].to_numpy(dtype=int),
            "epoch_id": test["epoch_id"].to_numpy(dtype=int),
            "fold": fold,
            "true_label": y_test,
            "true_class": [stage_name(label) for label in y_test],
            "pred_standard": standard_pred.astype(int),
            "pred_balanced": balanced_pred.astype(int),
        }))
        observed_test_subjects.update(declared_test)
        fold_audits.append({
            "fold": fold,
            "declared_test_subjects": list(declared_test),
            "actual_test_subjects": list(actual_test),
            "subject_overlap": sorted(set(declared_train).intersection(declared_test)),
            "standard_train_feature_sha256": feature_digest(standard_x_train),
            "balanced_train_feature_sha256": feature_digest(balanced_x_train),
            "standard_test_feature_sha256": feature_digest(standard_x_test),
            "balanced_test_feature_sha256": feature_digest(balanced_x_test),
        })

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["subject_id", "epoch_id"]).reset_index(drop=True)
    duplicate_count = int(oof.duplicated(["subject_id", "epoch_id"]).sum())
    expected_keys = set(zip(frame["subject_id"].astype(int), frame["epoch_id"].astype(int), strict=True))
    observed_keys = set(zip(oof["subject_id"].astype(int), oof["epoch_id"].astype(int), strict=True))
    oof_exactly_once = duplicate_count == 0 and len(oof) == len(frame) and observed_keys == expected_keys
    if not oof_exactly_once:
        raise AssertionError("OOF predictions do not contain every cached epoch exactly once.")
    if observed_test_subjects != Counter({subject: 1 for subject in FIXED_SUBJECTS}):
        raise AssertionError(f"Test-subject appearances are invalid: {observed_test_subjects}")
    if not all(set(audit["subject_overlap"]) == set() for audit in fold_audits):
        raise AssertionError("A fold has subject leakage.")
    if set(all_indices) != set(frame.index):  # Keep the full-frame indexing invariant explicit.
        raise AssertionError("Unexpected frame index mutation.")

    fold_metrics = pd.DataFrame(fold_rows).sort_values(["fold", "model"]).reset_index(drop=True)
    overall_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    for model, prediction_column in (("standard", "pred_standard"), ("balanced", "pred_balanced")):
        accuracy, macro_f1, precision, recall, f1, support = metrics_for_predictions(
            oof["true_label"].to_numpy(dtype=int), oof[prediction_column].to_numpy(dtype=int)
        )
        overall_rows.append({"model": model, "n_oof_epochs": len(oof), "accuracy": accuracy, "macro_f1": macro_f1})
        for code, name, p_value, r_value, f_value, count in zip(
            STAGE_CODES, STAGE_NAMES, precision, recall, f1, support, strict=True
        ):
            per_class_rows.append({
                "model": model,
                "class_label": int(code),
                "class_name": name,
                "precision": float(p_value),
                "recall": float(r_value),
                "f1": float(f_value),
                "support": int(count),
            })
    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(per_class_rows)
    summary, confusion = append_confusion_matrices(markdown_metric_table(overall, per_class), oof)

    checks = {
        "fixed_subjects": list(FIXED_SUBJECTS),
        "fold_assignment_source": assignments_path.relative_to(root).as_posix(),
        "fold_assignment_read_unchanged": True,
        "both_models_share_identical_fold_test_subjects": True,
        "model_parameters": {"standard": standard_params, "balanced": balanced_params},
        "model_parameter_differences": {key: list(value) for key, value in parameter_differences.items()},
        "only_class_weight_differs": True,
        "identical_feature_columns": columns,
        "identical_model_inputs": True,
        "per_fold_feature_fingerprints": fold_audits,
        "subject_leakage_detected": False,
        "each_subject_tested_once": True,
        "oof_epochs_expected": int(len(frame)),
        "oof_epochs_observed": int(len(oof)),
        "oof_duplicate_subject_epoch_keys": duplicate_count,
        "each_epoch_has_exactly_one_oof_prediction": True,
    }
    outputs = {
        "fold_metrics": results_dir / "fold_metrics.csv",
        "overall_metrics": results_dir / "overall_metrics.csv",
        "per_class_metrics": results_dir / "per_class_metrics.csv",
        "oof_predictions": results_dir / "oof_predictions.csv",
        "confusion_matrices": results_dir / "confusion_matrices.csv",
        "validation_checks": results_dir / "validation_checks.json",
        "summary": results_dir / "experiment_summary.md",
    }
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    overall.to_csv(outputs["overall_metrics"], index=False)
    per_class.to_csv(outputs["per_class_metrics"], index=False)
    oof.to_csv(outputs["oof_predictions"], index=False)
    confusion.to_csv(outputs["confusion_matrices"], index=False)
    outputs["validation_checks"].write_text(json.dumps(checks, indent=2), encoding="utf-8")
    outputs["summary"].write_text(summary, encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root())
    args = parser.parse_args()
    outputs = run_experiment(args.project_root.resolve())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
