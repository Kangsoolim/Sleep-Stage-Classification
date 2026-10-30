"""Run the locked 10D vs 12D spectral-entropy feature extension.

The script reads the accepted feature cache and saved subject-wise folds. It
rebuilds PSD features from the existing local EDF files only to calculate the
two spectral-entropy columns and to validate the accepted 10D features. No
folds or model settings are regenerated or tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data_utils import (
    EEG_CHANNELS,
    STAGE_CODES,
    STAGE_NAMES,
    load_sleep_edf_epochs,
)
from features import FREQ_BANDS, PSD_RANGE_HZ, feature_columns


ROOT = Path(__file__).resolve().parents[1]
CACHE_RELATIVE_PATH = Path(
    "data/processed/sleep_edf_subjects_0_9_recording_1_epoch_features.csv"
)
FOLDS_RELATIVE_PATH = Path("results/main_experiment/fold_assignments.csv")
FETCHED_RELATIVE_PATH = Path("results/main_experiment/fetched_files.json")
ACCEPTED_OOF_RELATIVE_PATH = Path("results/oof_predictions.csv")
ACCEPTED_OVERALL_RELATIVE_PATH = Path("results/overall_metrics.csv")
ACCEPTED_CLASS_RELATIVE_PATH = Path("results/per_class_metrics.csv")
OUTPUT_RELATIVE_PATH = Path("results/spectral_entropy_extension")
EXTENDED_CACHE_FILENAME = (
    "sleep_edf_subjects_0_9_recording_1_epoch_features_with_entropy.csv"
)
TABLE_FILENAME = "table_spectral_entropy_extension.csv"
FIGURE_FILENAME = "figure_6_spectral_entropy_extension.png"

FIXED_SUBJECTS = tuple(range(10))
EXPECTED_EPOCHS = 9_913
RANDOM_STATE = 241
N_ESTIMATORS = 100
FEATURE_RTOL = 1e-8
FEATURE_ATOL = 1e-10
METRIC_ATOL = 1e-8
SHORT_STAGES = ("W", "N1", "N2", "Stage 3+4", "REM")
STAGE_LABELS = dict(zip(STAGE_CODES, SHORT_STAGES, strict=True))
PALETTE = {
    "W": "#4C78A8",
    "N1": "#F58518",
    "N2": "#54A24B",
    "Stage 3+4": "#B279A2",
    "REM": "#E45756",
    "10D": "#4C78A8",
    "12D + entropy": "#E07B39",
}
INK = "#263238"
GRID = "#D9E1E5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_subject_list(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    return tuple(sorted(int(item) for item in text.split(";") if item.strip())) if text else ()


def psd_features_and_entropy(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return Core-order 10D features, two normalized entropies, and PSD grid.

    The compute_psd call deliberately matches the accepted Core call exactly:
    no explicit method or estimator arguments are added.
    """
    spectrum = epochs.compute_psd(
        picks="eeg", fmin=PSD_RANGE_HZ[0], fmax=PSD_RANGE_HZ[1]
    )
    psds, freqs = spectrum.get_data(return_freqs=True)
    if psds.ndim != 3 or psds.shape[1] != len(EEG_CHANNELS):
        raise AssertionError(f"Unexpected PSD shape: {psds.shape}")
    if not np.isfinite(psds).all() or np.any(psds < 0):
        raise AssertionError("PSD values must be finite and non-negative.")
    if len(freqs) < 2 or not np.isclose(freqs[0], PSD_RANGE_HZ[0]) or not np.isclose(freqs[-1], PSD_RANGE_HZ[1]):
        raise AssertionError(f"PSD frequency grid does not span {PSD_RANGE_HZ}: {freqs[[0, -1]]}")

    psd_sum = np.sum(psds, axis=-1, keepdims=True)
    if not np.isfinite(psd_sum).all() or np.any(psd_sum <= 0):
        raise AssertionError("Every epoch/channel must have positive finite 0.5-30 Hz PSD power.")
    relative_psd = psds / psd_sum

    blocks: list[np.ndarray] = []
    for fmin, fmax in FREQ_BANDS.values():
        band_mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(band_mask):
            raise AssertionError(f"No PSD bins found for band {fmin}-{fmax} Hz.")
        blocks.append(relative_psd[:, :, band_mask].mean(axis=-1).reshape(len(psds), -1))
    band_power = np.concatenate(blocks, axis=1)

    # K is fixed to all valid bins in the shared PSD grid for every row/channel.
    valid_frequency_mask = np.isfinite(freqs) & (freqs >= PSD_RANGE_HZ[0]) & (freqs <= PSD_RANGE_HZ[1])
    k_bins = int(valid_frequency_mask.sum())
    if k_bins < 2:
        raise AssertionError(f"Normalized entropy requires at least two fixed PSD bins, got K={k_bins}.")
    fixed_psd = psds[:, :, valid_frequency_mask]
    fixed_sums = fixed_psd.sum(axis=-1, keepdims=True)
    probabilities = fixed_psd / fixed_sums
    positive = probabilities > 0
    log_terms = np.zeros_like(probabilities, dtype=np.float64)
    log_terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    entropy = -log_terms.sum(axis=-1) / math.log(k_bins)
    tolerance = 1e-12
    if not np.isfinite(entropy).all():
        raise AssertionError("Spectral entropy contains NaN or Inf.")
    if entropy.min() < -tolerance or entropy.max() > 1.0 + tolerance:
        raise AssertionError(f"Normalized spectral entropy is outside [0, 1]: {entropy.min()}, {entropy.max()}")
    entropy = np.clip(entropy, 0.0, 1.0)
    return band_power, entropy, freqs, str(getattr(spectrum, "method", "MNE Epochs.compute_psd default"))


def extract_extension_features(root: Path, cache: pd.DataFrame, fetched: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_keys = cache[["subject_id", "epoch_id", "label"]].copy()
    cache_keys["subject_id"] = cache_keys["subject_id"].astype(int)
    cache_keys["epoch_id"] = cache_keys["epoch_id"].astype(int)
    cache_keys["label"] = cache_keys["label"].astype(int)
    if len(cache) != EXPECTED_EPOCHS:
        raise AssertionError(f"Accepted cache must contain {EXPECTED_EPOCHS} rows, got {len(cache)}")
    if cache.duplicated(["subject_id", "epoch_id"]).any():
        raise AssertionError("Accepted feature cache has duplicate subject/epoch keys.")
    if tuple(sorted(cache["subject_id"].unique().tolist())) != FIXED_SUBJECTS:
        raise AssertionError("Accepted feature cache subjects are not 0-9.")
    columns = list(feature_columns())
    expected_cache_columns = ["subject_id", "epoch_id", "label", *columns]
    if cache.columns.tolist() != expected_cache_columns:
        raise AssertionError("Accepted 10D cache schema or feature order differs from Core.")

    mne_cache = Path(fetched["mne_cache_dir"])
    if not mne_cache.is_absolute():
        mne_cache = root / mne_cache
    edf_dir = mne_cache / "physionet-sleep-data"
    frames: list[pd.DataFrame] = []
    shared_frequencies: np.ndarray | None = None
    psd_method: str | None = None

    for record in fetched["files"]:
        subject_id = int(record["subject_id"])
        if int(record["recording"]) != 1 or subject_id not in FIXED_SUBJECTS:
            raise AssertionError(f"Unexpected EDF record in fetched_files.json: {record}")
        psg = edf_dir / str(record["psg_filename"])
        hypnogram = edf_dir / str(record["hypnogram_filename"])
        if not psg.is_file() or not hypnogram.is_file():
            raise FileNotFoundError(f"Required local EDF cache file is missing: {psg if not psg.is_file() else hypnogram}")

        raw, _, _, epochs = load_sleep_edf_epochs(psg, hypnogram)
        try:
            if not np.isclose(raw.info["sfreq"], 100.0, rtol=0.0, atol=1e-12):
                raise AssertionError(f"Subject {subject_id} sampling frequency is {raw.info['sfreq']}, expected 100 Hz.")
            power, entropy, freqs, current_method = psd_features_and_entropy(epochs)
            if shared_frequencies is None:
                shared_frequencies = freqs.copy()
                # MNE exposes the selected PSD method; parameters still follow Core defaults.
                psd_method = current_method
            elif not np.array_equal(shared_frequencies, freqs):
                raise AssertionError(f"Subject {subject_id} PSD grid differs from the fixed grid.")
            labels = epochs.events[:, 2].astype(int, copy=False)
            if len(power) != len(labels) or len(entropy) != len(labels):
                raise AssertionError(f"Subject {subject_id} feature/label row count mismatch.")
            subject_frame = pd.DataFrame(power, columns=columns)
            subject_frame.insert(0, "label", labels)
            subject_frame.insert(0, "epoch_id", np.arange(len(subject_frame), dtype=int))
            subject_frame.insert(0, "subject_id", subject_id)
            for channel_index, channel in enumerate(EEG_CHANNELS):
                subject_frame[f"spectral_entropy_{channel}"] = entropy[:, channel_index]
            frames.append(subject_frame)
        finally:
            raw.close()
            del epochs

    calculated = pd.concat(frames, ignore_index=True)
    calculated_keys = calculated[["subject_id", "epoch_id", "label"]].astype(int)
    if not calculated_keys.equals(cache_keys):
        mismatches = int((calculated_keys != cache_keys).any(axis=1).sum())
        raise AssertionError(f"EDF-derived subject/epoch/label alignment differs from accepted cache ({mismatches} rows).")

    recomputed = calculated.loc[:, columns].to_numpy(dtype=np.float64)
    accepted = cache.loc[:, columns].to_numpy(dtype=np.float64)
    difference = np.abs(recomputed - accepted)
    max_abs = float(difference.max(initial=0.0))
    relative = np.zeros_like(difference)
    nonzero_ref = np.abs(accepted) > 0
    np.divide(difference, np.abs(accepted), out=relative, where=nonzero_ref)
    relative[(~nonzero_ref) & (difference > 0)] = np.inf
    max_rel = float(relative.max(initial=0.0))
    exact = bool(np.array_equal(recomputed, accepted))
    within_tolerance = bool(np.allclose(recomputed, accepted, rtol=FEATURE_RTOL, atol=FEATURE_ATOL))
    if not within_tolerance:
        raise AssertionError(
            "Recomputed 10D band-power features differ materially from accepted cache; "
            f"max_abs_error={max_abs:.17g}, max_rel_error={max_rel:.17g}. Stopping before model fits."
        )

    # Use the immutable accepted 10D columns in the extension cache/model inputs.
    extended = cache.copy()
    for entropy_column in (f"spectral_entropy_{EEG_CHANNELS[0]}", f"spectral_entropy_{EEG_CHANNELS[1]}"):
        extended[entropy_column] = calculated[entropy_column].to_numpy(dtype=np.float64)
    entropy_values = extended.iloc[:, -2:].to_numpy(dtype=np.float64)
    if not np.isfinite(entropy_values).all() or entropy_values.min() < 0 or entropy_values.max() > 1:
        raise AssertionError("Entropy cache failed finite/range validation.")

    diagnostics = {
        "recomputed_10d_exact_match": exact,
        "recomputed_10d_within_tolerance": within_tolerance,
        "comparison_tolerance": {"rtol": FEATURE_RTOL, "atol": FEATURE_ATOL},
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "entropy_min": float(entropy_values.min()),
        "entropy_max": float(entropy_values.max()),
        "entropy_finite": True,
        "psd_frequency_bin_count_K": int(len(shared_frequencies) if shared_frequencies is not None else 0),
        "psd_frequency_first_hz": float(shared_frequencies[0]) if shared_frequencies is not None else None,
        "psd_frequency_last_hz": float(shared_frequencies[-1]) if shared_frequencies is not None else None,
        "psd_method": psd_method,
    }
    return extended, diagnostics


def validate_assignments(assignments: pd.DataFrame, frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {
        "fold", "train_subjects", "test_subjects", "n_train_subjects", "n_test_subjects",
        "n_train_epochs", "n_test_epochs", "subject_overlap_count",
    }
    if set(assignments.columns) != required or len(assignments) != 5:
        raise AssertionError("Saved fold assignment schema/count differs from accepted 5-fold file.")
    assignments = assignments.sort_values("fold").reset_index(drop=True)
    if assignments["fold"].astype(int).tolist() != [1, 2, 3, 4, 5]:
        raise AssertionError("Saved folds must be numbered 1-5.")
    expected = set(FIXED_SUBJECTS)
    audits: list[dict[str, Any]] = []
    test_appearances: list[int] = []
    for row in assignments.itertuples(index=False):
        train_subjects = read_subject_list(row.train_subjects)
        test_subjects = read_subject_list(row.test_subjects)
        if set(train_subjects) & set(test_subjects):
            raise AssertionError(f"Saved fold {row.fold} leaks subjects.")
        if set(train_subjects) | set(test_subjects) != expected:
            raise AssertionError(f"Saved fold {row.fold} does not partition subjects 0-9.")
        train = frame["subject_id"].isin(train_subjects)
        test = frame["subject_id"].isin(test_subjects)
        if int(train.sum()) != int(row.n_train_epochs) or int(test.sum()) != int(row.n_test_epochs):
            raise AssertionError(f"Saved fold {row.fold} epoch counts differ from cache.")
        if len(train_subjects) != 8 or len(test_subjects) != 2 or int(row.subject_overlap_count) != 0:
            raise AssertionError(f"Saved fold {row.fold} metadata differs from the fixed design.")
        test_appearances.extend(test_subjects)
        audits.append({"fold": int(row.fold), "train_subjects": list(train_subjects), "test_subjects": list(test_subjects), "subject_overlap": []})
    if sorted(test_appearances) != list(FIXED_SUBJECTS):
        raise AssertionError("Every subject must occur as a test subject exactly once.")
    return audits


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(STAGE_CODES), zero_division=0
    )
    return float(accuracy_score(y_true, y_pred)), float(np.mean(f1)), precision, recall, f1, support


def run_models(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assignments = assignments.sort_values("fold").reset_index(drop=True)
    keys = ["subject_id", "epoch_id"]
    feature_sets = {
        "10D": list(feature_columns()),
        "12D + entropy": [*feature_columns(), f"spectral_entropy_{EEG_CHANNELS[0]}", f"spectral_entropy_{EEG_CHANNELS[1]}"],
    }
    if len(frame) != EXPECTED_EPOCHS or frame.duplicated(keys).any():
        raise AssertionError("Feature matrix must be the unique accepted 9,913 epochs.")
    y_all = frame["label"].to_numpy(dtype=int)
    fold_rows: list[dict[str, Any]] = []
    fold_cache: list[dict[str, Any]] = []

    # Complete and validate A before fitting any 12D model.
    for assignment in assignments.itertuples(index=False):
        fold = int(assignment.fold)
        train_subjects = read_subject_list(assignment.train_subjects)
        test_subjects = read_subject_list(assignment.test_subjects)
        train_mask = frame["subject_id"].isin(train_subjects).to_numpy()
        test_mask = frame["subject_id"].isin(test_subjects).to_numpy()
        train, test = frame.loc[train_mask], frame.loc[test_mask]
        if set(train["subject_id"]) != set(train_subjects) or set(test["subject_id"]) != set(test_subjects):
            raise AssertionError(f"Fold {fold} actual subjects differ from saved assignments.")
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        columns_10d = feature_sets["10D"]
        x_train = train.loc[:, columns_10d].to_numpy(dtype=np.float64)
        x_test = test.loc[:, columns_10d].to_numpy(dtype=np.float64)
        if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
            raise AssertionError(f"Fold {fold} 10D inputs have non-finite values.")
        model_10d = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        pred_10d = model_10d.fit(x_train, y_train).predict(x_test).astype(int)
        accuracy, macro, *_ = class_metrics(y_test, pred_10d)
        fold_rows.append({
            "fold": fold,
            "feature_set": "10D",
            "feature_dimension": len(columns_10d),
            "n_train_epochs": int(len(train)),
            "n_test_epochs": int(len(test)),
            "train_subjects": ";".join(map(str, train_subjects)),
            "test_subjects": ";".join(map(str, test_subjects)),
            "accuracy": accuracy,
            "macro_f1": macro,
        })
        fold_cache.append({
            "fold": fold,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train": train,
            "test": test,
            "y_train": y_train,
            "y_test": y_test,
            "pred_10d": pred_10d,
            "audit_10d": {
                "feature_columns": columns_10d,
                "train_feature_sha256": hashlib.sha256(np.ascontiguousarray(x_train).tobytes()).hexdigest(),
                "test_feature_sha256": hashlib.sha256(np.ascontiguousarray(x_test).tobytes()).hexdigest(),
                "rf_parameters": model_10d.get_params(deep=True),
            },
        })

    oof_10d = pd.concat([
        pd.DataFrame({
            "subject_id": test["subject_id"].to_numpy(dtype=int),
            "epoch_id": test["epoch_id"].to_numpy(dtype=int),
            "fold": fold_cache[fold_index]["fold"],
            "true_label": y_test,
            "true_class": [dict(zip(STAGE_CODES, STAGE_NAMES, strict=True))[label] for label in y_test],
            "pred_10d": pred_10d,
        })
        for fold_index, (test, y_test, pred_10d) in enumerate(
            (item["test"], item["y_test"], item["pred_10d"]) for item in fold_cache
        )
    ], ignore_index=True).sort_values(keys).reset_index(drop=True)
    baseline_gate = compare_baseline_to_accepted(root, oof_10d)

    # Only after A reproduces the accepted result, fit B on the same saved folds.
    oof_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for item in fold_cache:
        fold = item["fold"]
        train_subjects, test_subjects = item["train_subjects"], item["test_subjects"]
        train, test = item["train"], item["test"]
        y_train, y_test = item["y_train"], item["y_test"]
        columns_12d = feature_sets["12D + entropy"]
        x_train_12d = train.loc[:, columns_12d].to_numpy(dtype=np.float64)
        x_test_12d = test.loc[:, columns_12d].to_numpy(dtype=np.float64)
        model_12d = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        pred_12d = model_12d.fit(x_train_12d, y_train).predict(x_test_12d).astype(int)
        accuracy, macro, *_ = class_metrics(y_test, pred_12d)
        fold_rows.append({
            "fold": fold,
            "feature_set": "12D + entropy",
            "feature_dimension": len(columns_12d),
            "n_train_epochs": int(len(train)),
            "n_test_epochs": int(len(test)),
            "train_subjects": ";".join(map(str, train_subjects)),
            "test_subjects": ";".join(map(str, test_subjects)),
            "accuracy": accuracy,
            "macro_f1": macro,
        })
        audits.append({
            "fold": fold,
            "train_subjects": list(train_subjects),
            "test_subjects": list(test_subjects),
            "subject_overlap": [],
            "feature_sets": {
                "10D": item["audit_10d"],
                "12D + entropy": {
                    "feature_columns": columns_12d,
                    "train_feature_sha256": hashlib.sha256(np.ascontiguousarray(x_train_12d).tobytes()).hexdigest(),
                    "test_feature_sha256": hashlib.sha256(np.ascontiguousarray(x_test_12d).tobytes()).hexdigest(),
                    "rf_parameters": model_12d.get_params(deep=True),
                },
            },
        })
        oof_parts.append(pd.DataFrame({
            "subject_id": test["subject_id"].to_numpy(dtype=int),
            "epoch_id": test["epoch_id"].to_numpy(dtype=int),
            "fold": fold,
            "true_label": y_test,
            "true_class": [dict(zip(STAGE_CODES, STAGE_NAMES, strict=True))[label] for label in y_test],
            "pred_10d": item["pred_10d"],
            "pred_12d_entropy": pred_12d,
        }))

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(keys).reset_index(drop=True)
    expected_keys = set(zip(frame["subject_id"].astype(int), frame["epoch_id"].astype(int), strict=True))
    observed_keys = set(zip(oof["subject_id"].astype(int), oof["epoch_id"].astype(int), strict=True))
    duplicate_count = int(oof.duplicated(keys).sum())
    if len(oof) != EXPECTED_EPOCHS or observed_keys != expected_keys or duplicate_count:
        raise AssertionError("OOF table is not exactly one prediction per accepted epoch.")
    ordered_labels = frame.sort_values(keys)["label"].to_numpy(dtype=int)
    if not np.array_equal(oof["true_label"].to_numpy(dtype=int), ordered_labels):
        raise AssertionError("OOF labels do not match accepted cache labels.")
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["fold", "feature_set"]).reset_index(drop=True)
    return oof, fold_metrics, {
        "per_fold": audits,
        "duplicate_subject_epoch_keys": duplicate_count,
        "expected_oof_epochs": EXPECTED_EPOCHS,
        "observed_oof_epochs": int(len(oof)),
        "each_epoch_exactly_once": True,
        "subject_leakage_detected": False,
        "baseline_gate": baseline_gate,
    }


def summarize_metrics(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    overall_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    metric_lookup: dict[str, dict[str, float]] = {}
    for feature_set, pred_column, display, dimension in (
        ("10D", "pred_10d", "Balanced RF / 10D band-power", 10),
        ("12D + entropy", "pred_12d_entropy", "Balanced RF / 12D band-power + spectral entropy", 12),
    ):
        y_true = oof["true_label"].to_numpy(dtype=int)
        y_pred = oof[pred_column].to_numpy(dtype=int)
        accuracy, macro, precision, recall, f1, support = class_metrics(y_true, y_pred)
        metric_lookup[feature_set] = {"accuracy": accuracy, "macro_f1": macro}
        overall_rows.append({
            "feature_set": feature_set,
            "feature_set_label": display,
            "feature_dimension": dimension,
            "n_oof_epochs": len(oof),
            "accuracy": accuracy,
            "macro_f1": macro,
        })
        for code, stage, p, r, score, n in zip(STAGE_CODES, STAGE_NAMES, precision, recall, f1, support, strict=True):
            per_class_rows.append({
                "feature_set": feature_set,
                "class_label": int(code),
                "class_name": STAGE_LABELS[code],
                "precision": float(p),
                "recall": float(r),
                "f1": float(score),
                "support": int(n),
            })
        matrix = confusion_matrix(y_true, y_pred, labels=list(STAGE_CODES))
        for true_code, row in zip(STAGE_CODES, matrix, strict=True):
            for pred_code, count in zip(STAGE_CODES, row, strict=True):
                confusion_rows.append({
                    "feature_set": feature_set,
                    "true_label": int(true_code),
                    "true_class": STAGE_LABELS[true_code],
                    "pred_label": int(pred_code),
                    "pred_class": STAGE_LABELS[pred_code],
                    "count": int(count),
                })
    fold_summary = fold_metrics.groupby("feature_set", sort=False).agg(
        fold_accuracy_mean=("accuracy", "mean"),
        fold_accuracy_sd=("accuracy", "std"),
        fold_macro_f1_mean=("macro_f1", "mean"),
        fold_macro_f1_sd=("macro_f1", "std"),
    )
    overall = pd.DataFrame(overall_rows).join(fold_summary, on="feature_set")
    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(confusion_rows)
    class_f1 = per_class.pivot(index="class_name", columns="feature_set", values="f1").reindex(SHORT_STAGES)
    deltas = {
        "accuracy": float(metric_lookup["12D + entropy"]["accuracy"] - metric_lookup["10D"]["accuracy"]),
        "macro_f1": float(metric_lookup["12D + entropy"]["macro_f1"] - metric_lookup["10D"]["macro_f1"]),
    }
    for stage in SHORT_STAGES:
        deltas[f"{stage}_f1"] = float(class_f1.loc[stage, "12D + entropy"] - class_f1.loc[stage, "10D"])
    return overall, fold_metrics, per_class, {"confusion": confusion, "deltas": deltas}


def entropy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for channel in EEG_CHANNELS:
        column = f"spectral_entropy_{channel}"
        for code, stage in zip(STAGE_CODES, SHORT_STAGES, strict=True):
            values = frame.loc[frame["label"].eq(code), column].to_numpy(dtype=float)
            if not len(values):
                raise AssertionError(f"No entropy observations for stage {stage}.")
            rows.append({
                "channel": channel,
                "stage_label": stage,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            })
    return pd.DataFrame(rows)


def make_table4(overall: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    overall_index = overall.set_index("feature_set")
    class_pivot = per_class.pivot(index="class_name", columns="feature_set", values="f1").reindex(SHORT_STAGES)
    rows = []
    for feature_set, label, dimension in (
        ("10D", "Balanced RF / 10D band-power", 10),
        ("12D + entropy", "Balanced RF / 12D band-power + spectral entropy", 12),
    ):
        rows.append({
            "Feature set": label,
            "Feature dimension": dimension,
            "Accuracy": float(overall_index.loc[feature_set, "accuracy"]),
            "Macro-F1": float(overall_index.loc[feature_set, "macro_f1"]),
            "W F1": float(class_pivot.loc["W", feature_set]),
            "N1 F1": float(class_pivot.loc["N1", feature_set]),
            "N2 F1": float(class_pivot.loc["N2", feature_set]),
            "Stage 3+4 F1": float(class_pivot.loc["Stage 3+4", feature_set]),
            "REM F1": float(class_pivot.loc["REM", feature_set]),
        })
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#8A99A1",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
    })


def finish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def build_figure(frame: pd.DataFrame, per_class: pd.DataFrame, overall: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.28, 1.0]})
    stage_positions = np.arange(len(SHORT_STAGES))
    channel_offsets = (-0.18, 0.18)
    channel_colors = ("#4C78A8", "#E07B39")
    channel_hatches = ("", "///")
    for channel_index, channel in enumerate(EEG_CHANNELS):
        values = [
            frame.loc[frame["label"].eq(code), f"spectral_entropy_{channel}"].to_numpy(dtype=float)
            for code in STAGE_CODES
        ]
        positions = stage_positions + channel_offsets[channel_index]
        box = axes[0].boxplot(
            values,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.1},
            whiskerprops={"color": "#73838B", "linewidth": 0.85},
            capprops={"color": "#73838B", "linewidth": 0.85},
            boxprops={"edgecolor": channel_colors[channel_index], "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(channel_colors[channel_index])
            patch.set_alpha(0.25)
            patch.set_hatch(channel_hatches[channel_index])
        for median in box["medians"]:
            median.set_zorder(4)
    axes[0].set_xticks(stage_positions, SHORT_STAGES)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("True sleep stage")
    axes[0].set_ylabel("Normalized spectral entropy")
    axes[0].set_title("A  Entropy by stage and EEG channel", loc="left", fontweight="bold")
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[
            Patch(facecolor=channel_colors[i], edgecolor=channel_colors[i], alpha=0.25, hatch=channel_hatches[i], label=channel)
            for i, channel in enumerate(EEG_CHANNELS)
        ],
        frameon=False,
        loc="upper right",
    )
    finish_axis(axes[0])

    f1 = per_class.pivot(index="class_name", columns="feature_set", values="f1").reindex(SHORT_STAGES)
    macro = overall.set_index("feature_set")["macro_f1"]
    categories = [*SHORT_STAGES, "Macro-F1"]
    values_10d = [*f1["10D"].to_list(), float(macro["10D"])]
    values_12d = [*f1["12D + entropy"].to_list(), float(macro["12D + entropy"])]
    positions = np.arange(len(categories))
    width = 0.34
    bars_10d = axes[1].bar(positions - width / 2, values_10d, width, color=PALETTE["10D"], edgecolor=INK, linewidth=0.45, label="Balanced RF / 10D", zorder=3)
    bars_12d = axes[1].bar(positions + width / 2, values_12d, width, color=PALETTE["12D + entropy"], edgecolor=INK, linewidth=0.45, hatch="///", label="Balanced RF / 12D + entropy", zorder=3)
    axes[1].set_xticks(positions, categories, rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("F1 metric")
    axes[1].set_ylabel("OOF F1 score")
    axes[1].set_title("B  Classification performance", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")
    for bars in (bars_10d, bars_12d):
        for bar in bars:
            axes[1].annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    finish_axis(axes[1])
    fig.suptitle("Spectral entropy feature extension", fontweight="bold", color=INK)
    fig.text(
        0.5,
        0.015,
        "Panel A: 9,913 accepted 30-s epochs; box = median and IQR, whiskers = 1.5×IQR, outlier markers hidden. Panel B: pooled OOF F1.",
        ha="center",
        fontsize=8,
        color="#53636B",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compare_baseline_to_accepted(root: Path, oof: pd.DataFrame) -> dict[str, Any]:
    accepted_oof = pd.read_csv(root / ACCEPTED_OOF_RELATIVE_PATH).sort_values(["subject_id", "epoch_id"]).reset_index(drop=True)
    expected_oof = accepted_oof[["subject_id", "epoch_id", "fold", "true_label", "pred_balanced"]]
    actual_oof = oof[["subject_id", "epoch_id", "fold", "true_label", "pred_10d"]]
    keys_and_truth_match = expected_oof[["subject_id", "epoch_id", "fold", "true_label"]].equals(actual_oof[["subject_id", "epoch_id", "fold", "true_label"]])
    prediction_exact = bool(np.array_equal(expected_oof["pred_balanced"].to_numpy(dtype=int), actual_oof["pred_10d"].to_numpy(dtype=int)))
    accepted_overall = pd.read_csv(root / ACCEPTED_OVERALL_RELATIVE_PATH).set_index("model").loc["balanced"]
    actual_accuracy, actual_macro, actual_precision, actual_recall, actual_f1, _ = class_metrics(
        oof["true_label"].to_numpy(dtype=int), oof["pred_10d"].to_numpy(dtype=int)
    )
    metric_deltas = {
        "accuracy": float(actual_accuracy - accepted_overall["accuracy"]),
        "macro_f1": float(actual_macro - accepted_overall["macro_f1"]),
    }
    accepted_per_class = pd.read_csv(root / ACCEPTED_CLASS_RELATIVE_PATH)
    accepted_per_class = accepted_per_class.loc[accepted_per_class["model"].eq("balanced")].sort_values("class_label")
    actual_per_class_values = np.column_stack((actual_precision, actual_recall, actual_f1))
    class_metrics_equal = bool(np.allclose(
        accepted_per_class[["precision", "recall", "f1"]].to_numpy(dtype=float),
        actual_per_class_values,
        rtol=0.0,
        atol=METRIC_ATOL,
    ))
    pooled_metrics_equal = all(abs(value) <= METRIC_ATOL for value in metric_deltas.values())

    sklearn_version_differs = None
    # Accepted result files do not store the sklearn version, so exact-prediction
    # differences are diagnosed from the local package and their observed pattern.
    reason = None
    if not prediction_exact:
        sklearn_version_differs = "not recorded in accepted result metadata"
        changed_count = int(np.count_nonzero(expected_oof["pred_balanced"].to_numpy(dtype=int) != actual_oof["pred_10d"].to_numpy(dtype=int)))
        reason = (
            f"Predictions differ for {changed_count} epochs although cache, keys, labels, folds, parameters, "
            "and pooled metrics were checked; accepted files do not record sklearn version, so the exact "
            "source of the tiny prediction difference cannot be attributed to a library version."
        )

    checks = {
        "accepted_baseline_oof_keys_fold_and_truth_match": bool(keys_and_truth_match),
        "accepted_baseline_oof_predictions_exact_match": prediction_exact,
        "accepted_balanced_accuracy": float(accepted_overall["accuracy"]),
        "accepted_balanced_macro_f1": float(accepted_overall["macro_f1"]),
        "rerun_10d_accuracy": actual_accuracy,
        "rerun_10d_macro_f1": actual_macro,
        "baseline_metric_deltas": metric_deltas,
        "baseline_per_class_metrics_within_tolerance": class_metrics_equal,
        "baseline_metrics_within_tolerance": pooled_metrics_equal,
        "metric_tolerance_absolute": METRIC_ATOL,
        "local_sklearn_version": sklearn.__version__,
        "exact_match_deviation_reason": reason,
        "accepted_sklearn_version": sklearn_version_differs,
    }
    if not keys_and_truth_match:
        raise AssertionError("10D baseline OOF keys/folds/truth do not match accepted result; stopping.")
    if not prediction_exact and not (class_metrics_equal and pooled_metrics_equal):
        raise AssertionError("10D baseline does not reproduce accepted pooled/per-class metrics; stopping.")
    return checks


def write_summary(path: Path, overall: pd.DataFrame, per_class: pd.DataFrame, fold_metrics: pd.DataFrame, deltas: dict[str, float], baseline_checks: dict[str, Any], entropy_diag: dict[str, Any]) -> None:
    index = overall.set_index("feature_set")
    class_pivot = per_class.pivot(index="class_name", columns="feature_set", values="f1").reindex(SHORT_STAGES)
    lines = [
        "# Spectral entropy exploratory extension",
        "",
        "This is a fixed feature-extension comparison on the accepted Sleep-EDF subjects 0–9, recording 1, and 9,913 30-s EEG epochs. The accepted five-fold subject-wise fold assignments and Balanced RF settings (`n_estimators=100`, `random_state=241`, `class_weight=balanced`) were reused.",
        "",
        "## Pooled OOF results",
        "",
        "| Feature set | Dimension | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for feature_set, label in (("10D", "Balanced RF / 10D band-power"), ("12D + entropy", "Balanced RF / 12D + spectral entropy")):
        row = index.loc[feature_set]
        lines.append(f"| {label} | {int(row['feature_dimension'])} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} |")
    lines.extend(["", "## Five-fold mean ± SD", "", "| Feature set | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for feature_set, label in (("10D", "10D"), ("12D + entropy", "12D + entropy")):
        row = index.loc[feature_set]
        lines.append(f"| {label} | {row['fold_accuracy_mean']:.6f} ± {row['fold_accuracy_sd']:.6f} | {row['fold_macro_f1_mean']:.6f} ± {row['fold_macro_f1_sd']:.6f} |")
    lines.extend(["", "## Change (12D + entropy minus 10D)", "", "| Metric | Delta |", "|---|---:|"])
    for key in ("accuracy", "macro_f1", "W_f1", "N1_f1", "N2_f1", "Stage 3+4_f1", "REM_f1"):
        lines.append(f"| {key.replace('_', ' ')} | {deltas[key]:+.6f} |")
    lines.extend(["", "## Per-class pooled OOF F1", "", "| Stage | 10D | 12D + entropy | Delta |", "|---|---:|---:|---:|"])
    for stage in SHORT_STAGES:
        left, right = class_pivot.loc[stage, "10D"], class_pivot.loc[stage, "12D + entropy"]
        lines.append(f"| {stage} | {left:.6f} | {right:.6f} | {right-left:+.6f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The extension result is descriptive for this fixed subject-independent setting. No hypothesis test or parameter search was conducted. The observed changes are reported as measured; they do not establish statistical significance or general effectiveness.",
        "",
        "## Validation notes",
        "",
        f"- Recomputed Core 10D features exact match accepted cache: `{entropy_diag['recomputed_10d_exact_match']}`; maximum absolute error {entropy_diag['max_absolute_error']:.3g}, maximum relative error {entropy_diag['max_relative_error']:.3g}.",
        f"- 10D OOF predictions exact match accepted result: `{baseline_checks['accepted_baseline_oof_predictions_exact_match']}`.",
        f"- Accepted pooled baseline metric tolerance check: `{baseline_checks['baseline_metrics_within_tolerance']}`.",
        f"- Five folds, 9,913 epochs, and one OOF prediction per epoch are validated in `validation_checks.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, Path]:
    root = root.resolve()
    output_dir = root / OUTPUT_RELATIVE_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_cache_path = root / CACHE_RELATIVE_PATH
    folds_path = root / FOLDS_RELATIVE_PATH
    fetched_path = root / FETCHED_RELATIVE_PATH
    accepted_oof_path = root / ACCEPTED_OOF_RELATIVE_PATH
    accepted_overall_path = root / ACCEPTED_OVERALL_RELATIVE_PATH
    accepted_class_path = root / ACCEPTED_CLASS_RELATIVE_PATH
    for required_path in (accepted_cache_path, folds_path, fetched_path, accepted_oof_path, accepted_overall_path, accepted_class_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    # Preserve the accepted CSV's decimal round-trip values exactly when the
    # immutable 10D columns are carried into the extension cache.
    cache = pd.read_csv(accepted_cache_path, float_precision="round_trip")
    assignments = pd.read_csv(folds_path)
    fetched = json.loads(fetched_path.read_text(encoding="utf-8"))
    fold_audits = validate_assignments(assignments, cache)
    extended, feature_diagnostics = extract_extension_features(root, cache, fetched)

    extended_cache_path = root / "data" / "processed" / EXTENDED_CACHE_FILENAME
    extended.to_csv(extended_cache_path, index=False, float_format="%.17g")
    cache_hash = sha256_file(accepted_cache_path)
    folds_hash = sha256_file(folds_path)
    accepted_oof_hash = sha256_file(accepted_oof_path)

    # Run A first, then gate B on exact predictions or full data/fold/parameter
    # agreement plus accepted pooled and per-class metric reproduction.
    oof, fold_metrics, oof_checks = run_models(extended, assignments, root)
    overall, fold_metrics, per_class, extras = summarize_metrics(oof, fold_metrics)
    baseline_checks = oof_checks["baseline_gate"]

    entropy = entropy_summary(extended)
    deltas = extras["deltas"]
    confusion = extras["confusion"]
    table4 = make_table4(overall, per_class)

    config = {
        "experiment": "Spectral Entropy exploratory feature extension",
        "dataset": "Sleep-EDF Expanded / Sleep Cassette",
        "subjects": list(FIXED_SUBJECTS),
        "recording": 1,
        "channels": list(EEG_CHANNELS),
        "sampling_frequency_hz": 100,
        "epoch_duration_s": 30,
        "n_epochs": EXPECTED_EPOCHS,
        "stages": list(SHORT_STAGES),
        "core_features": list(feature_columns()),
        "extension_features": [f"spectral_entropy_{channel}" for channel in EEG_CHANNELS],
        "feature_dimensions": {"10D": 10, "12D + entropy": 12},
        "psd": {
            "method": feature_diagnostics["psd_method"],
            "api": "Epochs.compute_psd(picks='eeg', fmin=0.5, fmax=30.0) with the same MNE defaults as Core",
            "range_hz": list(PSD_RANGE_HZ),
            "frequency_bin_count_K": feature_diagnostics["psd_frequency_bin_count_K"],
            "frequency_grid_first_hz": feature_diagnostics["psd_frequency_first_hz"],
            "frequency_grid_last_hz": feature_diagnostics["psd_frequency_last_hz"],
            "entropy_zero_probability_convention": "0 log 0 = 0; positive mask only avoids log(0)",
            "entropy_denominator": "log(K), with fixed K equal to all valid bins in the 0.5-30 Hz PSD grid",
        },
        "fold_assignment_path": folds_path.relative_to(root).as_posix(),
        "fold_assignment_sha256": folds_hash,
        "folds_read_without_regeneration": True,
        "model": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "other_parameters": "scikit-learn defaults",
            "scikit_learn_version": sklearn.__version__,
        },
        "input_cache_sha256": cache_hash,
        "accepted_oof_sha256": accepted_oof_hash,
        "mne_version": mne.__version__,
        "feature_validation": feature_diagnostics,
        "baseline_reproduction": baseline_checks,
    }
    write_json(output_dir / "experiment_config.json", config)

    checks = {
        "fixed_subjects": list(FIXED_SUBJECTS),
        "fixed_recording": 1,
        "expected_epochs": EXPECTED_EPOCHS,
        "observed_epochs": int(len(extended)),
        "feature_dimension_10d": len(feature_columns()),
        "feature_dimension_12d": len(feature_columns()) + len(EEG_CHANNELS),
        "accepted_10d_cache_sha256_before_run": cache_hash,
        "accepted_fold_assignments_sha256": folds_hash,
        "accepted_fold_assignments_read_unchanged": True,
        "fold_assignments": fold_audits,
        "recomputed_feature_checks": feature_diagnostics,
        "entropy_columns": [f"spectral_entropy_{channel}" for channel in EEG_CHANNELS],
        "entropy_finite": bool(np.isfinite(extended.iloc[:, -2:].to_numpy(dtype=float)).all()),
        "entropy_min": feature_diagnostics["entropy_min"],
        "entropy_max": feature_diagnostics["entropy_max"],
        "entropy_in_unit_interval": 0 <= feature_diagnostics["entropy_min"] <= feature_diagnostics["entropy_max"] <= 1,
        "oof": oof_checks,
        "baseline_reproduction": baseline_checks,
        "all_folds_subject_disjoint": True,
        "rf_parameters_identical_between_feature_sets": True,
        "no_hyperparameter_search": True,
        "no_hypothesis_test": True,
    }

    # Save all small result files only after both experiment arms pass validation.
    entropy.to_csv(output_dir / "entropy_summary_by_stage.csv", index=False, float_format="%.10g")
    overall.to_csv(output_dir / "overall_metrics.csv", index=False, float_format="%.12g")
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, float_format="%.12g")
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False, float_format="%.12g")
    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    table4_path = root / "results" / TABLE_FILENAME
    table4.to_csv(table4_path, index=False, float_format="%.12g")
    figure_path = root / "figures" / FIGURE_FILENAME
    build_figure(extended, per_class, overall, figure_path)
    write_summary(output_dir / "experiment_summary.md", overall, per_class, fold_metrics, deltas, baseline_checks, feature_diagnostics)
    write_json(output_dir / "validation_checks.json", checks)

    return {
        "extended_cache": extended_cache_path,
        "output_dir": output_dir,
        "table4": table4_path,
        "figure6": figure_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    output_paths = run(args.project_root)
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
