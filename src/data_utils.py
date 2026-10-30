"""Shared data preparation for the fixed Sleep-EDF baseline.

The processing in this module intentionally mirrors the accepted MNE tutorial
reproduction: only the two EEG channels, the original five sleep-stage labels,
and 30-second annotation chunks are used. It does not filter the signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne
import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data
from mne.datasets.sleep_physionet import _utils as sleep_physionet_utils


EPOCH_DURATION_S = 30.0
WAKE_TRIM_BUFFER_S = 30 * 60
EEG_CHANNELS = ("Fpz-Cz", "Pz-Oz")
# MNE's bundled Pooch downloader uses 1 KiB chunks by default.  A larger
# transfer chunk changes only the transport behavior of ``fetch_data()``, not
# its URL, requested records, checksums, or returned data.
FETCH_CHUNK_SIZE_BYTES = 1_048_576
# Official public object-store mirror documented by PhysioNet for this exact
# Sleep-EDF Expanded release.  ``fetch_data()`` still supplies filenames and
# validates each downloaded file against MNE's pinned SHA-1 record.
PHYSIONET_SLEEP_EDF_S3_BASE_URL = (
    "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette"
)

# These two dictionaries are deliberately separate: stages 3 and 4 are first
# mapped to the same event code, then represented by one merged epoch label.
ANNOTATION_DESC_TO_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3/4": 4,
    "Sleep stage R": 5,
}
STAGE_NAMES = tuple(EVENT_ID)
STAGE_CODES = tuple(EVENT_ID.values())


def fetch_sleep_edf_files(
    subjects: Sequence[int],
    recording: Sequence[int] = (1,),
    *,
    path: str | Path | None = None,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Fetch Sleep-EDF PSG/hypnogram pairs and return validated file records."""
    original_downloader_params = sleep_physionet_utils._downloader_params

    def downloader_params_with_larger_chunks(**kwargs: object) -> dict[str, object]:
        params = original_downloader_params(**kwargs)
        params["chunk_size"] = FETCH_CHUNK_SIZE_BYTES
        return params

    # The public MNE fetch_data() API remains the sole data acquisition path.
    # This narrow temporary patch avoids a transport-only 1 KiB chunk bottleneck
    # on high-latency connections and is restored even if fetching raises.
    sleep_physionet_utils._downloader_params = downloader_params_with_larger_chunks
    try:
        files_by_subject = fetch_data(
            subjects=list(subjects),
            recording=list(recording),
            path=path,
            base_url=PHYSIONET_SLEEP_EDF_S3_BASE_URL,
            verbose=verbose,
        )
    finally:
        sleep_physionet_utils._downloader_params = original_downloader_params
    if len(recording) != 1:
        raise ValueError("This fixed baseline expects exactly one recording per subject.")

    records: list[dict[str, object]] = []
    for subject, files in zip(subjects, files_by_subject, strict=True):
        if len(files) != 2:
            raise RuntimeError(f"Expected PSG and Hypnogram for subject {subject}, got {files}")
        psg_path, hypnogram_path = map(Path, files)
        if not psg_path.name.endswith("-PSG.edf"):
            raise AssertionError(f"Unexpected PSG filename: {psg_path}")
        if not hypnogram_path.name.endswith("-Hypnogram.edf"):
            raise AssertionError(f"Unexpected Hypnogram filename: {hypnogram_path}")
        if not psg_path.is_file() or not hypnogram_path.is_file():
            raise FileNotFoundError((psg_path, hypnogram_path))
        records.append(
            {
                "subject": subject,
                "recording": recording[0],
                "psg_edf": psg_path.resolve(),
                "hypnogram_edf": hypnogram_path.resolve(),
            }
        )
    return records


def read_psg_and_hypnogram(
    psg_path: str | Path, hypnogram_path: str | Path
) -> tuple[mne.io.BaseRaw, mne.Annotations]:
    """Read a PSG EDF and its hypnogram using the official tutorial settings."""
    raw = mne.io.read_raw_edf(
        psg_path,
        stim_channel="Event marker",
        infer_types=True,
        preload=True,
        verbose="error",
    )
    annotations = mne.read_annotations(hypnogram_path)
    return raw, annotations


def trim_wake_annotations(
    annotations: mne.Annotations,
    *,
    wake_buffer_s: float = WAKE_TRIM_BUFFER_S,
) -> mne.Annotations:
    """Keep the official 30-minute Wake context before and after sleep."""
    if len(annotations) < 3:
        raise ValueError("At least three annotations are required for Wake trimming.")
    trimmed = annotations.copy()
    # This is intentionally the same crop boundary as the official tutorial.
    trimmed.crop(
        trimmed[1]["onset"] - wake_buffer_s,
        trimmed[-2]["onset"] + wake_buffer_s,
    )
    return trimmed


def validate_baseline_epochs(epochs: mne.Epochs) -> None:
    """Assert the fixed EEG, label, and epoch-duration invariants."""
    eeg_channels = tuple(
        channel
        for channel, channel_type in zip(epochs.ch_names, epochs.get_channel_types(), strict=True)
        if channel_type == "eeg"
    )
    if not set(EEG_CHANNELS).issubset(eeg_channels):
        raise AssertionError(f"Required EEG channels {EEG_CHANNELS}, got {eeg_channels}")

    epoch_duration_s = epochs.times[-1] + 1.0 / epochs.info["sfreq"]
    if not np.isclose(epoch_duration_s, EPOCH_DURATION_S, rtol=0.0, atol=1e-12):
        raise AssertionError(f"Expected {EPOCH_DURATION_S}-s epochs, got {epoch_duration_s}")

    observed_codes = set(np.unique(epochs.events[:, 2]).tolist())
    allowed_codes = set(STAGE_CODES)
    if not observed_codes or not observed_codes.issubset(allowed_codes):
        raise AssertionError(f"Only stage codes {allowed_codes} are allowed, got {observed_codes}")


def annotations_to_30s_epochs(
    raw: mne.io.BaseRaw, annotations: mne.Annotations
) -> tuple[mne.Annotations, np.ndarray, mne.Epochs]:
    """Trim annotations, map labels, and construct the fixed 30-second epochs."""
    trimmed_annotations = trim_wake_annotations(annotations)
    raw.set_annotations(trimmed_annotations, emit_warning=False)
    events, _ = mne.events_from_annotations(
        raw,
        event_id=ANNOTATION_DESC_TO_EVENT_ID,
        chunk_duration=EPOCH_DURATION_S,
    )
    tmax = EPOCH_DURATION_S - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=EVENT_ID,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    validate_baseline_epochs(epochs)
    return trimmed_annotations, events, epochs


def load_sleep_edf_epochs(
    psg_path: str | Path, hypnogram_path: str | Path
) -> tuple[mne.io.BaseRaw, mne.Annotations, np.ndarray, mne.Epochs]:
    """Read one pair of EDF files and return its validated baseline epochs."""
    raw, annotations = read_psg_and_hypnogram(psg_path, hypnogram_path)
    trimmed_annotations, events, epochs = annotations_to_30s_epochs(raw, annotations)
    return raw, trimmed_annotations, events, epochs
