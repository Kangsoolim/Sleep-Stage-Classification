"""Minimal evaluation primitives shared by later subject-grouped experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from data_utils import STAGE_CODES, STAGE_NAMES


def make_subject_group_kfold(n_splits: int = 5) -> GroupKFold:
    """Create the fixed subject-disjoint splitter used by both future RF models."""
    return GroupKFold(n_splits=n_splits)


def evaluate_stage_predictions(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> dict[str, object]:
    """Calculate the fixed five-class metrics used for baseline parity checks."""
    labels = list(STAGE_CODES)
    target_names = list(STAGE_NAMES)
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, labels=labels, average="macro", zero_division=0)
        ),
        "report": classification_report(
            y_true_array,
            y_pred_array,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred_array, labels=labels),
    }
