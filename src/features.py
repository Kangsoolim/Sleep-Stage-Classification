"""Normalized PSD band-power features for the fixed Sleep-EDF baseline."""

from __future__ import annotations

from collections.abc import Mapping

import mne
import numpy as np

from data_utils import EEG_CHANNELS


FREQ_BANDS: Mapping[str, tuple[float, float]] = {
    "delta": (0.5, 4.5),
    "theta": (4.5, 8.5),
    "alpha": (8.5, 11.5),
    "sigma": (11.5, 15.5),
    "beta": (15.5, 30.0),
}
PSD_RANGE_HZ = (0.5, 30.0)
EXPECTED_FEATURE_DIM = len(EEG_CHANNELS) * len(FREQ_BANDS)


def normalized_psd_band_power(epochs: mne.Epochs) -> np.ndarray:
    """Return the official tutorial's normalized PSD band-power matrix.

    Columns are band-major then channel-major, exactly as in the original loop:
    delta/Fpz-Cz, delta/Pz-Oz, ..., beta/Fpz-Cz, beta/Pz-Oz.
    """
    spectrum = epochs.compute_psd(picks="eeg", fmin=PSD_RANGE_HZ[0], fmax=PSD_RANGE_HZ[1])
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)

    feature_blocks = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        feature_blocks.append(psds_band.reshape(len(psds), -1))
    features = np.concatenate(feature_blocks, axis=1)
    validate_feature_matrix(features)
    return features


def feature_columns() -> tuple[str, ...]:
    """Return stable names matching the baseline feature-matrix column order."""
    return tuple(f"{band}_{channel}" for band in FREQ_BANDS for channel in EEG_CHANNELS)


def validate_feature_matrix(features: np.ndarray) -> None:
    """Assert the fixed 2 EEG channels × 5 bands matrix has finite values."""
    if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_DIM:
        raise AssertionError(
            f"Expected feature shape (n_epochs, {EXPECTED_FEATURE_DIM}), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise AssertionError("Normalized PSD band-power features contain NaN or Inf.")
