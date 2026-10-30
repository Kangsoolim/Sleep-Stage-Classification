"""Build the fixed multi-subject Sleep-EDF feature cache without fitting models.

This command is deliberately limited to data preparation and subject-disjoint
fold assignment.  Standard RF and balanced RF must both load the saved cache
and reuse ``results/main_experiment/fold_assignments.csv`` unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import EPOCH_DURATION_S, EVENT_ID, STAGE_CODES, STAGE_NAMES, fetch_sleep_edf_files, load_sleep_edf_epochs
from evaluation import make_subject_group_kfold
from features import EXPECTED_FEATURE_DIM, feature_columns, normalized_psd_band_power


FIXED_SUBJECTS = tuple(range(10))
FIXED_RECORDING = (1,)
N_SPLITS = 5
CACHE_FILENAME = "sleep_edf_subjects_0_9_recording_1_epoch_features.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage_count_dict(labels: pd.Series | np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(labels, dtype=int).tolist())
    return {stage_name: int(counts[stage_code]) for stage_name, stage_code in EVENT_ID.items()}


def validate_epoch_frame(frame: pd.DataFrame) -> None:
    """Check the saved epoch-level table has exactly the fixed experiment schema."""
    expected_columns = ["subject_id", "epoch_id", "label", *feature_columns()]
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(f"Unexpected cache columns: {frame.columns.tolist()}")
    if frame.shape[1] - 3 != EXPECTED_FEATURE_DIM:
        raise AssertionError(f"Expected 10 feature columns, got {frame.shape[1] - 3}")
    if tuple(sorted(frame["subject_id"].unique().tolist())) != FIXED_SUBJECTS:
        raise AssertionError(f"Expected subjects {FIXED_SUBJECTS}, got {frame['subject_id'].unique()}")
    if not set(frame["label"].unique()).issubset(STAGE_CODES):
        raise AssertionError(f"Unexpected labels: {sorted(frame['label'].unique().tolist())}")
    if set(frame["label"].unique()) != set(STAGE_CODES):
        raise AssertionError("All five fixed sleep-stage labels must be present.")
    if frame.isna().any().any():
        raise AssertionError("The epoch-level cache contains NaN values.")
    feature_values = frame.loc[:, list(feature_columns())].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise AssertionError("The epoch-level cache contains Inf or non-finite features.")
    for subject_id, subject_frame in frame.groupby("subject_id", sort=True):
        expected_epoch_ids = np.arange(len(subject_frame), dtype=int)
        observed_epoch_ids = subject_frame["epoch_id"].to_numpy(dtype=int)
        if not np.array_equal(observed_epoch_ids, expected_epoch_ids):
            raise AssertionError(f"Subject {subject_id} has non-contiguous epoch_id values.")


def make_fold_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create and validate the one fixed 5-fold subject-disjoint split plan."""
    groups = frame["subject_id"].to_numpy(dtype=int)
    splitter = make_subject_group_kfold(n_splits=N_SPLITS)
    assignment_rows: list[dict[str, object]] = []
    test_appearance_counts: Counter[int] = Counter()

    # The all-zero X merely supplies GroupKFold an indexable feature matrix;
    # neither a model nor a random epoch-level split is involved here.
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros((len(frame), 1)), frame["label"].to_numpy(), groups), start=1
    ):
        train_subjects = tuple(sorted(np.unique(groups[train_index]).tolist()))
        test_subjects = tuple(sorted(np.unique(groups[test_index]).tolist()))
        overlap = set(train_subjects).intersection(test_subjects)
        if overlap:
            raise AssertionError(f"Fold {fold} subject overlap: {sorted(overlap)}")
        if len(test_subjects) != len(FIXED_SUBJECTS) // N_SPLITS:
            raise AssertionError(f"Fold {fold} must contain two test subjects, got {test_subjects}")
        test_appearance_counts.update(test_subjects)
        assignment_rows.append(
            {
                "fold": fold,
                "train_subjects": ";".join(map(str, train_subjects)),
                "test_subjects": ";".join(map(str, test_subjects)),
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "n_train_epochs": len(train_index),
                "n_test_epochs": len(test_index),
                "subject_overlap_count": len(overlap),
            }
        )

    expected_appearances = {subject: 1 for subject in FIXED_SUBJECTS}
    observed_appearances = {subject: test_appearance_counts[subject] for subject in FIXED_SUBJECTS}
    if observed_appearances != expected_appearances:
        raise AssertionError(f"Each subject must be a test group exactly once: {observed_appearances}")
    return pd.DataFrame(assignment_rows)


def build_main_experiment_data(project_root: Path) -> dict[str, Path]:
    """Fetch fixed files, make the cache, and save all auditable small outputs."""
    cache_dir = project_root / "data" / "mne_data"
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results" / "main_experiment"
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    file_records = fetch_sleep_edf_files(
        subjects=FIXED_SUBJECTS, recording=FIXED_RECORDING, path=cache_dir, verbose=True
    )
    frames: list[pd.DataFrame] = []
    per_subject_epochs: dict[str, int] = {}
    fetched_files: list[dict[str, object]] = []
    columns = list(feature_columns())

    for file_record in file_records:
        subject_id = int(file_record["subject"])
        raw, _, _, epochs = load_sleep_edf_epochs(
            file_record["psg_edf"], file_record["hypnogram_edf"]
        )
        features = normalized_psd_band_power(epochs)
        labels = epochs.events[:, 2].astype(int, copy=False)
        if len(features) != len(labels):
            raise AssertionError(f"Subject {subject_id}: feature/label count mismatch")
        subject_frame = pd.DataFrame(features, columns=columns)
        subject_frame.insert(0, "label", labels)
        subject_frame.insert(0, "epoch_id", np.arange(len(subject_frame), dtype=int))
        subject_frame.insert(0, "subject_id", subject_id)
        frames.append(subject_frame)
        per_subject_epochs[str(subject_id)] = int(len(subject_frame))
        fetched_files.append(
            {
                "subject_id": subject_id,
                "recording": FIXED_RECORDING[0],
                "psg_filename": Path(file_record["psg_edf"]).name,
                "hypnogram_filename": Path(file_record["hypnogram_edf"]).name,
            }
        )
        raw.close()
        del raw, epochs

    epoch_frame = pd.concat(frames, ignore_index=True)
    validate_epoch_frame(epoch_frame)
    fold_assignments = make_fold_assignments(epoch_frame)

    feature_cache_path = processed_dir / CACHE_FILENAME
    epoch_frame.to_csv(feature_cache_path, index=False)
    fold_path = results_dir / "fold_assignments.csv"
    fold_assignments.to_csv(fold_path, index=False)
    fetched_files_path = results_dir / "fetched_files.json"
    fetched_files_path.write_text(
        json.dumps(
            {
                "subjects": list(FIXED_SUBJECTS),
                "recording": list(FIXED_RECORDING),
                "mne_cache_dir": cache_dir.relative_to(project_root).as_posix(),
                "files": fetched_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "subjects": list(FIXED_SUBJECTS),
        "recording": list(FIXED_RECORDING),
        "processing": {
            "eeg_channels": ["Fpz-Cz", "Pz-Oz"],
            "wake_trimming": "official tutorial: retain 30 minutes before/after sleep",
            "epoch_duration_s": EPOCH_DURATION_S,
            "labels": list(STAGE_NAMES),
            "feature_method": "official normalized PSD relative band-power",
            "feature_bands": ["delta", "theta", "alpha", "sigma", "beta"],
        },
        "epoch_count_by_subject": per_subject_epochs,
        "label_count_all_subjects": _stage_count_dict(epoch_frame["label"]),
        "feature_columns": columns,
        "feature_dimension": len(columns),
        "total_samples": int(len(epoch_frame)),
        "nan_count": int(epoch_frame.isna().sum().sum()),
        "inf_count_features": int(np.isinf(epoch_frame.loc[:, columns].to_numpy(dtype=float)).sum()),
        "group_kfold": {
            "n_splits": N_SPLITS,
            "groups": "subject_id",
            "all_fold_subject_overlap_count": fold_assignments["subject_overlap_count"].tolist(),
            "each_subject_tested_once": True,
        },
        "feature_cache": feature_cache_path.relative_to(project_root).as_posix(),
    }
    summary_path = results_dir / "data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "feature_cache": feature_cache_path,
        "fold_assignments": fold_path,
        "summary": summary_path,
        "fetched_files": fetched_files_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_project_root(),
        help="Repository root (defaults to the parent of this src directory).",
    )
    args = parser.parse_args()
    output_paths = build_main_experiment_data(args.project_root.resolve())
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
