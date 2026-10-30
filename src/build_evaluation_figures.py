"""Build QA-checked evaluation figures and tables for the locked RF comparison.

This script is deliberately report-only: it reads the accepted result CSVs and
OOF predictions, and it derives the raw/PSD illustrations from the same ten
Sleep-EDF recordings without writing features or fitting a classifier.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from data_utils import EEG_CHANNELS, EPOCH_DURATION_S, STAGE_CODES, STAGE_NAMES, load_sleep_edf_epochs, read_psg_and_hypnogram
from features import EXPECTED_FEATURE_DIM, FREQ_BANDS, PSD_RANGE_HZ


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SHORT_STAGES = ("W", "N1", "N2", "Stage 3+4", "REM")
MODEL_LABELS = {"standard": "Standard RF", "balanced": "Balanced RF"}
STAGE_LABELS = dict(zip(STAGE_CODES, SHORT_STAGES, strict=True))
PALETTE = {
    "W": "#4C78A8",
    "N1": "#F58518",
    "N2": "#54A24B",
    "Stage 3+4": "#B279A2",
    "REM": "#E45756",
    "standard": "#4C78A8",
    "balanced": "#E07B39",
}
INK = "#263238"
GRID = "#D9E1E5"
PSD_BAND_BOUNDARIES_HZ = (4.5, 8.5, 11.5, 15.5)
PSD_BANDS = (
    (0.5, 4.5, "delta"),
    (4.5, 8.5, "theta"),
    (8.5, 11.5, "alpha"),
    (11.5, 15.5, "sigma"),
    (15.5, 30.0, "beta"),
)


def load_csv_inputs() -> dict[str, pd.DataFrame]:
    """Load every numeric model result from its tracked CSV source."""
    return {
        "oof": pd.read_csv(RESULTS / "oof_predictions.csv"),
        "fold": pd.read_csv(RESULTS / "fold_metrics.csv"),
        "overall": pd.read_csv(RESULTS / "overall_metrics.csv"),
        "per_class": pd.read_csv(RESULTS / "per_class_metrics.csv"),
        "confusion": pd.read_csv(RESULTS / "confusion_matrices.csv"),
        "assignments": pd.read_csv(RESULTS / "main_experiment" / "fold_assignments.csv"),
    }


def assert_result_integrity(data: dict[str, pd.DataFrame], summary: dict[str, object]) -> None:
    """Fail early if a report value does not trace to the accepted OOF results."""
    if tuple(SHORT_STAGES) != ("W", "N1", "N2", "Stage 3+4", "REM"):
        raise AssertionError("Figure labels must use the fixed five-stage naming, including Stage 3+4.")
    if len(STAGE_CODES) != 5 or tuple(PSD_RANGE_HZ) != (0.5, 30.0):
        raise AssertionError("Evaluation must retain the fixed five stages and 0.5-30 Hz PSD range.")
    if int(summary["total_samples"]) != 9_913:
        raise AssertionError("Evaluation must retain the accepted 9,913 epochs.")
    oof, fold, overall, per_class, confusion = (
        data["oof"], data["fold"], data["overall"], data["per_class"], data["confusion"]
    )
    required_oof = {"subject_id", "epoch_id", "fold", "true_label", "true_class", "pred_standard", "pred_balanced"}
    if set(oof.columns) != required_oof:
        raise AssertionError(f"Unexpected OOF schema: {oof.columns.tolist()}")
    if oof.duplicated(["subject_id", "epoch_id"]).any() or len(oof) != int(summary["total_samples"]):
        raise AssertionError("OOF predictions are not one-to-one with the accepted epoch cache.")
    if sorted(oof["subject_id"].unique().tolist()) != list(summary["subjects"]):
        raise AssertionError("OOF subjects differ from the accepted ten-subject data summary.")
    if set(oof["true_label"].unique()) != set(STAGE_CODES):
        raise AssertionError("OOF labels do not contain the fixed five stages.")
    if sorted(fold["fold"].unique().tolist()) != [1, 2, 3, 4, 5] or set(fold["model"]) != set(MODEL_LABELS):
        raise AssertionError("Fold metrics do not represent the fixed five-fold two-model comparison.")

    expected_counts = {name: int(count) for name, count in summary["label_count_all_subjects"].items()}
    observed_counts = oof["true_class"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise AssertionError(f"OOF class counts differ from data summary: {observed_counts} != {expected_counts}")

    for model, prediction_col in (("standard", "pred_standard"), ("balanced", "pred_balanced")):
        y_true = oof["true_label"].to_numpy(dtype=int)
        y_pred = oof[prediction_col].to_numpy(dtype=int)
        source_overall = overall.set_index("model").loc[model]
        if not np.isclose(source_overall["accuracy"], accuracy_score(y_true, y_pred)):
            raise AssertionError(f"{model} accuracy does not agree with OOF predictions.")
        if not np.isclose(source_overall["macro_f1"], f1_score(y_true, y_pred, labels=STAGE_CODES, average="macro", zero_division=0)):
            raise AssertionError(f"{model} macro-F1 does not agree with OOF predictions.")
        _, _, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=STAGE_CODES, zero_division=0)
        source_class = per_class.loc[per_class["model"].eq(model)].sort_values("class_label")
        if not (np.allclose(source_class["f1"].to_numpy(), f1) and np.array_equal(source_class["support"].to_numpy(), support)):
            raise AssertionError(f"{model} per-class metrics do not agree with OOF predictions.")
        source_matrix = (
            confusion.loc[confusion["model"].eq(model)]
            .pivot(index="true_label", columns="pred_label", values="count")
            .reindex(index=STAGE_CODES, columns=STAGE_CODES)
            .to_numpy(dtype=int)
        )
        if not np.array_equal(source_matrix, confusion_matrix(y_true, y_pred, labels=STAGE_CODES)):
            raise AssertionError(f"{model} confusion matrix does not agree with OOF predictions.")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#8A99A1",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
    })


def finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, filename: str, figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    path = FIGURES / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    figures_for_pdf.append((filename, fig))


def add_caption(fig: plt.Figure, caption: str, *, bottom: float = 0.10, apply_layout: bool = True) -> None:
    """Reserve a real footer band so captions never collide with axis labels."""
    if apply_layout:
        fig.tight_layout(rect=(0.02, bottom, 0.98, 0.93))
    fig.text(0.5, bottom / 2, caption, ha="center", va="center", color="#53636B")


def plot_raw_eeg(fetched: dict[str, object], figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    """Plot the first 60 s of subject 0's real PSG, without filtering."""
    record = next(item for item in fetched["files"] if int(item["subject_id"]) == 0)
    cache_root = Path(fetched["mne_cache_dir"])
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    cache = cache_root / "physionet-sleep-data"
    raw, _ = read_psg_and_hypnogram(cache / record["psg_filename"], cache / record["hypnogram_filename"])
    start_s, duration_s = 0.0, 60.0
    samples = raw.get_data(picks=list(EEG_CHANNELS), start=0, stop=int(duration_s * raw.info["sfreq"])) * 1e6
    times = np.arange(samples.shape[1]) / raw.info["sfreq"] + start_s

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    for axis, channel, values in zip(axes, EEG_CHANNELS, samples, strict=True):
        axis.plot(times, values, color="#315C7B", linewidth=0.65)
        axis.set_ylabel(f"{channel}\nAmplitude (µV)")
        axis.set_xlim(start_s, start_s + duration_s)
        finish_axis(axis)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Representative raw EEG", fontweight="bold")
    add_caption(fig, "Sleep-EDF subject 0, recording 1; first 60 s; unfiltered Fpz-Cz and Pz-Oz")
    save_figure(fig, "figure_1_representative_raw_eeg.png", figures_for_pdf)


def plot_class_distribution(oof: pd.DataFrame, figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    counts = (
        oof.groupby(["subject_id", "true_label"]).size().unstack(fill_value=0)
        .reindex(index=sorted(oof["subject_id"].unique()), columns=STAGE_CODES, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bottom = np.zeros(len(counts), dtype=int)
    for code, stage in zip(STAGE_CODES, SHORT_STAGES, strict=True):
        values = counts[code].to_numpy()
        ax.bar(counts.index.astype(str), values, bottom=bottom, label=stage, color=PALETTE[stage], edgecolor="white", linewidth=0.45, zorder=3)
        bottom += values
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of 30-s epochs")
    ax.set_title("Sleep-stage distribution across 10 subjects", fontweight="bold")
    ax.legend(title="Stage", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    finish_axis(ax)
    add_caption(fig, "Counts are derived from the accepted OOF truth labels (n = 9,913 epochs).", bottom=0.19)
    save_figure(fig, "figure_2_subject_stage_distribution.png", figures_for_pdf)


def mean_relative_psd_by_stage(fetched: dict[str, object], expected_counts: dict[str, int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Aggregate the baseline's 0.5-30 Hz epoch-normalized PSD by true stage."""
    cache_root = Path(fetched["mne_cache_dir"])
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    cache = cache_root / "physionet-sleep-data"
    sums: dict[int, np.ndarray] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    frequencies: np.ndarray | None = None
    for record in fetched["files"]:
        _, _, _, epochs = load_sleep_edf_epochs(
            cache / record["psg_filename"], cache / record["hypnogram_filename"]
        )
        spectrum = epochs.compute_psd(picks="eeg", fmin=PSD_RANGE_HZ[0], fmax=PSD_RANGE_HZ[1], verbose="error")
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds /= psds.sum(axis=-1, keepdims=True)
        if frequencies is None:
            frequencies = freqs
        elif not np.array_equal(frequencies, freqs):
            raise AssertionError("PSD frequency grids are inconsistent across subjects.")
        for code in STAGE_CODES:
            selected = psds[epochs.events[:, 2] == code]
            if not len(selected):
                continue
            sums[code] = sums.get(code, np.zeros_like(selected[0])) + selected.sum(axis=0)
            counts[code] += len(selected)
        del epochs, spectrum, psds
    if frequencies is None or set(sums) != set(STAGE_CODES):
        raise AssertionError("Could not derive PSD summaries for every fixed stage.")
    if not np.allclose((frequencies[0], frequencies[-1]), PSD_RANGE_HZ):
        raise AssertionError("PSD frequencies differ from the locked 0.5-30 Hz range.")
    expected_by_code = {code: int(expected_counts[name]) for code, name in zip(STAGE_CODES, STAGE_NAMES, strict=True)}
    if dict(counts) != expected_by_code:
        raise AssertionError(f"PSD epoch counts differ from accepted data counts: {dict(counts)} != {expected_by_code}")
    if sum(counts.values()) != 9_913:
        raise AssertionError("PSD epoch total differs from the accepted 9,913 epochs.")
    return frequencies, {code: sums[code] / counts[code] for code in STAGE_CODES}


def plot_psd(frequencies: np.ndarray, mean_psd: dict[int, np.ndarray], figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for channel_index, (axis, channel) in enumerate(zip(axes, EEG_CHANNELS, strict=True)):
        for boundary_hz in PSD_BAND_BOUNDARIES_HZ:
            axis.axvline(boundary_hz, color="#B8C4C9", linestyle=(0, (2, 3)), linewidth=0.65, zorder=1)
        for code, stage in zip(STAGE_CODES, SHORT_STAGES, strict=True):
            axis.plot(frequencies, mean_psd[code][channel_index], label=stage, color=PALETTE[stage], linewidth=1.35, zorder=3)
        for lower_hz, upper_hz, band_name in PSD_BANDS:
            axis.text(
                (lower_hz + upper_hz) / 2,
                1.018,
                band_name,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#62727A",
                clip_on=False,
            )
        axis.set_title(channel, pad=24)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_xlim(*PSD_RANGE_HZ)
        finish_axis(axis)
    axes[0].set_ylabel("Mean relative PSD")
    axes[1].legend(title="True stage", frameon=False, loc="upper right")
    fig.suptitle("Mean normalized PSD by sleep stage", fontweight="bold")
    add_caption(fig, "All 9,913 accepted 30-s epochs; relative PSD normalized within 0.5-30 Hz per epoch and channel.")
    save_figure(fig, "figure_3_stage_psd_signatures.png", figures_for_pdf)


def confusion_from_csv(confusion: pd.DataFrame, model: str) -> np.ndarray:
    return (
        confusion.loc[confusion["model"].eq(model)]
        .pivot(index="true_label", columns="pred_label", values="count")
        .reindex(index=STAGE_CODES, columns=STAGE_CODES)
        .to_numpy(dtype=int)
    )


def plot_confusion_matrices(confusion: pd.DataFrame, figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    standard, balanced = (confusion_from_csv(confusion, model) for model in ("standard", "balanced"))
    vmax = max(standard.max(), balanced.max())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharex=True, sharey=True)
    image = None
    for axis, matrix, model in zip(axes, (standard, balanced), ("standard", "balanced"), strict=True):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xticks(range(len(SHORT_STAGES)), SHORT_STAGES, rotation=35, ha="right")
        axis.set_yticks(range(len(SHORT_STAGES)), SHORT_STAGES)
        axis.set_xlabel("Predicted stage")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(column, row, str(value), ha="center", va="center", fontsize=8, color="white" if value > vmax * 0.54 else INK)
    axes[0].set_ylabel("True stage")
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.24, top=0.84, wspace=0.32)
    colorbar = fig.colorbar(image, cax=fig.add_axes([0.87, 0.29, 0.025, 0.52]))
    colorbar.set_label("OOF epoch count")
    fig.suptitle("OOF confusion matrices", fontweight="bold")
    add_caption(fig, "Rows are true stages and columns are predicted stages; both panels use the same count scale.", bottom=0.07, apply_layout=False)
    save_figure(fig, "figure_4_oof_confusion_matrices.png", figures_for_pdf)


def plot_f1(per_class: pd.DataFrame, overall: pd.DataFrame, figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    pivot = per_class.pivot(index="class_label", columns="model", values="f1").reindex(STAGE_CODES)
    macro = overall.set_index("model")["macro_f1"].reindex(["standard", "balanced"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [3.3, 1]})
    positions = np.arange(len(SHORT_STAGES))
    width = 0.36
    axes[0].bar(positions - width / 2, pivot["standard"], width, label="Standard RF", color=PALETTE["standard"], zorder=3)
    axes[0].bar(positions + width / 2, pivot["balanced"], width, label="Balanced RF", color=PALETTE["balanced"], zorder=3)
    axes[0].set_xticks(positions, SHORT_STAGES)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlabel("Sleep stage")
    axes[0].set_ylabel("F1 score")
    axes[0].set_title("Per-class OOF F1")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.06), borderaxespad=0)
    finish_axis(axes[0])
    macro_positions = np.arange(2)
    macro_bars = axes[1].bar(macro_positions, macro.to_numpy(), color=[PALETTE["standard"], PALETTE["balanced"]], width=0.58, zorder=3)
    axes[1].bar_label(macro_bars, labels=[f"{value:.3f}" for value in macro], padding=3, fontsize=9)
    axes[1].set_xticks(macro_positions, ["Standard\nRF", "Balanced\nRF"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("OOF Macro-F1")
    finish_axis(axes[1])
    fig.suptitle("Per-class and Macro-F1 comparison: Standard RF vs Balanced RF", fontweight="bold")
    add_caption(fig, "F1 values are calculated from the same 9,913 accepted OOF predictions.")
    save_figure(fig, "figure_5_f1_comparison.png", figures_for_pdf)


def write_tables(data: dict[str, pd.DataFrame], summary: dict[str, object], checks: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments, fold, per_class = data["assignments"], data["fold"], data["per_class"]
    model_params = checks["model_parameters"]
    config = pd.DataFrame([
        ("Dataset", "Sleep-EDF Expanded / Sleep Cassette"),
        ("Subjects", "; ".join(str(value) for value in summary["subjects"])),
        ("Number of subjects", len(summary["subjects"])),
        ("Recording per subject", "; ".join(str(value) for value in summary["recording"])),
        ("Total 30-s epochs", int(summary["total_samples"])),
        ("EEG channels", "; ".join(summary["processing"]["eeg_channels"])),
        ("Stages", "; ".join(SHORT_STAGES)),
        ("PSD range (Hz)", f"{PSD_RANGE_HZ[0]}-{PSD_RANGE_HZ[1]}"),
        ("Feature method", summary["processing"]["feature_method"]),
        ("Feature dimension", int(summary["feature_dimension"])),
        ("Cross-validation", f"{len(assignments)}-fold GroupKFold by subject"),
        ("RF estimators", int(model_params["standard"]["n_estimators"])),
        ("RF random state", int(model_params["standard"]["random_state"])),
        ("Standard RF class weight", str(model_params["standard"]["class_weight"])),
        ("Balanced RF class weight", str(model_params["balanced"]["class_weight"])),
    ], columns=["Item", "Value"])
    if int(summary["feature_dimension"]) != EXPECTED_FEATURE_DIM:
        raise AssertionError("Data summary feature dimension does not match the locked feature definition.")
    fold_summary = (
        fold.groupby("model", sort=False)[["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .reindex(["standard", "balanced"])
    )
    performance = pd.DataFrame({
        "Model": [MODEL_LABELS[name] for name in fold_summary.index],
        "Accuracy mean": fold_summary[("accuracy", "mean")].to_numpy(),
        "Accuracy SD": fold_summary[("accuracy", "std")].to_numpy(),
        "Macro-F1 mean": fold_summary[("macro_f1", "mean")].to_numpy(),
        "Macro-F1 SD": fold_summary[("macro_f1", "std")].to_numpy(),
        "Folds": fold.groupby("model")["fold"].nunique().reindex(fold_summary.index).to_numpy(),
    })
    f1 = per_class.pivot(index="class_label", columns="model", values="f1").reindex(STAGE_CODES)
    class_f1 = pd.DataFrame({
        "Stage": SHORT_STAGES,
        "Baseline F1": f1["standard"].to_numpy(),
        "Balanced F1": f1["balanced"].to_numpy(),
        "Change (Balanced - Baseline)": (f1["balanced"] - f1["standard"]).to_numpy(),
    })
    config.to_csv(RESULTS / "table_data_experiment_configuration.csv", index=False)
    performance.to_csv(RESULTS / "table_foldwise_model_performance.csv", index=False, float_format="%.6f")
    class_f1.to_csv(RESULTS / "table_per_class_f1_comparison.csv", index=False, float_format="%.6f")
    return config, performance, class_f1


def write_key_findings(data: dict[str, pd.DataFrame], performance: pd.DataFrame, class_f1: pd.DataFrame) -> None:
    overall = data["overall"].set_index("model")
    standard, balanced = (overall.loc[name] for name in ("standard", "balanced"))
    n1, rem = (class_f1.set_index("Stage").loc[name] for name in ("N1", "REM"))
    standard_matrix = confusion_from_csv(data["confusion"], "standard")
    matrix_without_diagonal = standard_matrix.copy()
    np.fill_diagonal(matrix_without_diagonal, -1)
    row, column = np.unravel_index(np.argmax(matrix_without_diagonal), matrix_without_diagonal.shape)
    major_count = int(matrix_without_diagonal[row, column])
    major_confusion = f"{SHORT_STAGES[row]} to {SHORT_STAGES[column]} ({major_count} epochs)"
    text = "\n".join([
        "# Key findings",
        "",
        f"- On the accepted OOF predictions (n = {int(standard['n_oof_epochs']):,}), Standard RF had Accuracy {standard['accuracy']:.4f} and Macro-F1 {standard['macro_f1']:.4f}; Balanced RF had Accuracy {balanced['accuracy']:.4f} and Macro-F1 {balanced['macro_f1']:.4f}.",
        f"- Across the five held-out subject folds, Accuracy was {performance.loc[0, 'Accuracy mean']:.4f} ± {performance.loc[0, 'Accuracy SD']:.4f} for Standard RF and {performance.loc[1, 'Accuracy mean']:.4f} ± {performance.loc[1, 'Accuracy SD']:.4f} for Balanced RF; fold-wise Macro-F1 was {performance.loc[0, 'Macro-F1 mean']:.4f} ± {performance.loc[0, 'Macro-F1 SD']:.4f} and {performance.loc[1, 'Macro-F1 mean']:.4f} ± {performance.loc[1, 'Macro-F1 SD']:.4f}, respectively.",
        f"- With class weighting, N1 F1 changed from {n1['Baseline F1']:.4f} to {n1['Balanced F1']:.4f} ({n1['Change (Balanced - Baseline)']:+.4f}), and REM F1 changed from {rem['Baseline F1']:.4f} to {rem['Balanced F1']:.4f} ({rem['Change (Balanced - Baseline)']:+.4f}).",
        f"- The largest off-diagonal error for Standard RF was {major_confusion}; Figure 4 provides the full count matrices for both models.",
        f"- The balanced setting has higher OOF Macro-F1 ({balanced['macro_f1'] - standard['macro_f1']:+.4f}) alongside lower OOF Accuracy ({balanced['accuracy'] - standard['accuracy']:+.4f}), an observed trade-off in this fixed comparison.",
        "- Results use one 10-subject, recording-1, five-fold grouped evaluation and no statistical hypothesis test; observed differences are not interpreted as statistically significant or as evidence beyond this setting.",
        "",
    ])
    (RESULTS / "key_findings.md").write_text(text, encoding="utf-8")


def write_pdf(figures_for_pdf: list[tuple[str, plt.Figure]]) -> None:
    with PdfPages(FIGURES / "evaluation_figures.pdf") as pdf:
        for _, figure in figures_for_pdf:
            pdf.savefig(figure, bbox_inches="tight")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    configure_style()
    data = load_csv_inputs()
    summary = json.loads((RESULTS / "main_experiment" / "data_summary.json").read_text(encoding="utf-8"))
    checks = json.loads((RESULTS / "validation_checks.json").read_text(encoding="utf-8"))
    fetched = json.loads((RESULTS / "main_experiment" / "fetched_files.json").read_text(encoding="utf-8"))
    assert_result_integrity(data, summary)
    config, performance, class_f1 = write_tables(data, summary, checks)
    write_key_findings(data, performance, class_f1)

    figures_for_pdf: list[tuple[str, plt.Figure]] = []
    plot_raw_eeg(fetched, figures_for_pdf)
    plot_class_distribution(data["oof"], figures_for_pdf)
    frequencies, mean_psd = mean_relative_psd_by_stage(fetched, summary["label_count_all_subjects"])
    plot_psd(frequencies, mean_psd, figures_for_pdf)
    plot_confusion_matrices(data["confusion"], figures_for_pdf)
    plot_f1(data["per_class"], data["overall"], figures_for_pdf)
    write_pdf(figures_for_pdf)
    for _, figure in figures_for_pdf:
        plt.close(figure)
    print("QA passed: CSV metrics, OOF predictions, class counts, and confusion matrices agree.")
    print("Wrote 5 PNG figures, 1 multi-page PDF, 3 CSV tables, and key_findings.md.")


if __name__ == "__main__":
    main()
