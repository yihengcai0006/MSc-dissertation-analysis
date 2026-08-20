#!/usr/bin/env python3

"""
Regenerate the final dissertation classifier figures using only
previously completed analysis outputs.

IMPORTANT
---------
This script is a FIGURE-ONLY final-pass script.

It does NOT:
- refit any classifier;
- rerun nested cross-validation;
- perform feature selection;
- tune hyperparameters;
- rerun label permutations;
- recalculate feature-selection stability;
- modify the completed nested-CV results;
- modify the completed permutation results;
- overwrite the original classifier figures;
- overwrite the original result tables or figure-source tables.

It reads the formally completed outputs from the existing workflow and
creates only dissertation-formatted figure copies with the suffix:

    _final

The underlying numerical values, samples, fold results, permutation
scores and statistical conclusions are therefore unchanged.

Scientific interpretation
-------------------------
The low/high labels in the 100-ligand dataset are derived from existing
(old) RF-predicted barriers.

These classifier figures therefore describe structural patterns
associated with the behaviour of the existing RF model. They do not
show classification of measured or improved-QM/MM activation barriers.

Repeated outer-CV folds are not independent observations. Their
distribution in Figure 08A is descriptive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# General configuration
# ============================================================

BATCH = "batch0100"

FORMAL_FEATURE_SETS = (
    "residue_ligand",
    "residue_cofactor",
    "residue_residue",
    "combined_all",
)

# ------------------------------------------------------------
# Thesis-facing terminology only.
#
# IMPORTANT:
# Internal feature-set identifiers are NOT changed.
# Only the labels displayed in the figures are changed.
# ------------------------------------------------------------

FEATURE_SET_LABELS = {
    "residue_ligand": "Protein–ligand",
    "residue_cofactor": "Protein–GTP",
    "residue_residue": "Protein–protein",
    "combined_all": "Combined",
}

EXPECTED_OUTER_FOLDS_PER_SET = 25
EXPECTED_PERMUTATIONS_PER_SET = 30

PRIMARY_METRIC = "balanced_accuracy"

# Same value as the original plotting script.
# This preserves the same deterministic jitter pattern.
RANDOM_STATE = 42

PNG_DPI = 300


# ============================================================
# Final thesis figure formatting
# ============================================================
# Supervisor guidance was approximately font size 18.
#
# Main titles and axis labels therefore start at 18 pt.
# Tick labels, legends and annotations are slightly smaller
# where necessary to avoid overlap.
#
# Final readability should still be judged after insertion
# into the dissertation PDF.
# ============================================================

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 15
ANNOTATION_SIZE = 14
FOOTNOTE_SIZE = 13


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

NESTED_CV_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "nested_cv"
    / "final"
)

PERMUTATION_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "permutation"
    / "final_perm0001_0030"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "results"
    / BATCH
    / "classifier_figures"
)

FIGURE_DIR = (
    OUTPUT_ROOT
    / "figures"
)


# ============================================================
# Existing completed-analysis input files
# ============================================================

NESTED_METRICS_FILE = (
    NESTED_CV_DIR
    / "nested_cv_fold_metrics.tsv"
)

DUMMY_METRICS_FILE = (
    NESTED_CV_DIR
    / "nested_cv_dummy_metrics.tsv"
)

PERMUTATION_SCORES_FILE = (
    PERMUTATION_DIR
    / "permutation_scores.tsv"
)

PERMUTATION_SUMMARY_FILE = (
    PERMUTATION_DIR
    / "permutation_test_summary.tsv"
)

OBSERVED_SCORES_FILE = (
    PERMUTATION_DIR
    / "observed_scores.tsv"
)


# ============================================================
# Console helper
# ============================================================

def log(message: str = "") -> None:
    """Print one console message immediately."""
    print(
        message,
        flush=True,
    )


# ============================================================
# Input validation
# ============================================================

def validate_input_paths() -> None:
    """
    Confirm that the completed analysis outputs required for plotting
    already exist.
    """
    required_files = [
        NESTED_METRICS_FILE,
        DUMMY_METRICS_FILE,
        PERMUTATION_SCORES_FILE,
        PERMUTATION_SUMMARY_FILE,
        OBSERVED_SCORES_FILE,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required completed-analysis files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing_files
            )
        )


def prepare_output_directory() -> None:
    """Create the figure directory if required."""
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Load COMPLETED nested-CV results
# ============================================================

def load_nested_cv_metrics() -> pd.DataFrame:
    """
    Read the completed logistic-regression outer-fold metrics.

    No cross-validation is rerun here.
    """
    metrics = pd.read_csv(
        NESTED_METRICS_FILE,
        sep="\t",
    )

    required_columns = {
        "model_type",
        "feature_set",
        "balanced_accuracy",
    }

    missing_columns = (
        required_columns
        - set(metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Nested-CV metric table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    metrics = metrics.loc[
        metrics["model_type"]
        == "logistic_regression"
    ].copy()

    for feature_set in FORMAL_FEATURE_SETS:

        subset = metrics.loc[
            metrics["feature_set"]
            == feature_set
        ]

        if len(subset) != EXPECTED_OUTER_FOLDS_PER_SET:
            raise ValueError(
                f"{feature_set}: expected "
                f"{EXPECTED_OUTER_FOLDS_PER_SET} "
                "completed outer-fold rows, but observed "
                f"{len(subset)}."
            )

        if not subset[
            PRIMARY_METRIC
        ].between(
            0,
            1,
            inclusive="both",
        ).all():
            raise ValueError(
                f"{feature_set}: balanced accuracy "
                "outside [0, 1]."
            )

    expected_total_rows = (
        len(FORMAL_FEATURE_SETS)
        * EXPECTED_OUTER_FOLDS_PER_SET
    )

    if len(metrics) != expected_total_rows:
        raise ValueError(
            "Unexpected total number of completed nested-CV rows: "
            f"expected {expected_total_rows}, "
            f"observed {len(metrics)}."
        )

    return metrics


# ============================================================
# Load COMPLETED dummy results
# ============================================================

def load_dummy_metrics() -> pd.DataFrame:
    """
    Read the completed dummy-classifier outer-fold results.

    The dummy classifier is NOT rerun.
    """
    dummy = pd.read_csv(
        DUMMY_METRICS_FILE,
        sep="\t",
    )

    if "balanced_accuracy" not in dummy.columns:
        raise ValueError(
            "Dummy metric table does not contain "
            "'balanced_accuracy'."
        )

    if len(dummy) != EXPECTED_OUTER_FOLDS_PER_SET:
        raise ValueError(
            "Expected "
            f"{EXPECTED_OUTER_FOLDS_PER_SET} "
            "completed dummy outer-fold rows, "
            f"observed {len(dummy)}."
        )

    if not dummy[
        "balanced_accuracy"
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Dummy balanced accuracy outside [0, 1]."
        )

    return dummy


# ============================================================
# Load COMPLETED permutation results
# ============================================================

def load_permutation_scores() -> pd.DataFrame:
    """
    Read the previously completed permutation-level scores.

    No permutation analysis is rerun.
    """
    scores = pd.read_csv(
        PERMUTATION_SCORES_FILE,
        sep="\t",
    )

    required_columns = {
        "permutation_id",
        "feature_set",
        "mean_balanced_accuracy",
    }

    missing_columns = (
        required_columns
        - set(scores.columns)
    )

    if missing_columns:
        raise ValueError(
            "Permutation-score table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    for feature_set in FORMAL_FEATURE_SETS:

        subset = scores.loc[
            scores["feature_set"]
            == feature_set
        ]

        if len(subset) != EXPECTED_PERMUTATIONS_PER_SET:
            raise ValueError(
                f"{feature_set}: expected "
                f"{EXPECTED_PERMUTATIONS_PER_SET} "
                "completed permutation rows, observed "
                f"{len(subset)}."
            )

        observed_ids = sorted(
            subset[
                "permutation_id"
            ].astype(int).unique()
        )

        expected_ids = list(
            range(
                1,
                EXPECTED_PERMUTATIONS_PER_SET + 1,
            )
        )

        if observed_ids != expected_ids:
            raise ValueError(
                f"{feature_set}: completed permutation IDs "
                "do not exactly cover 1–30."
            )

        if not subset[
            "mean_balanced_accuracy"
        ].between(
            0,
            1,
            inclusive="both",
        ).all():
            raise ValueError(
                f"{feature_set}: permutation balanced "
                "accuracy outside [0, 1]."
            )

    return scores


def load_permutation_summary() -> pd.DataFrame:
    """Read the completed permutation-test summary."""
    summary = pd.read_csv(
        PERMUTATION_SUMMARY_FILE,
        sep="\t",
    )

    required_columns = {
        "feature_set",
        "observed_mean_balanced_accuracy",
        "n_permutations",
        "n_null_ge_observed",
        "permutation_p_value",
    }

    missing_columns = (
        required_columns
        - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Permutation summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if set(
        summary["feature_set"]
    ) != set(FORMAL_FEATURE_SETS):
        raise ValueError(
            "Permutation summary does not contain "
            "exactly the four formal feature sets."
        )

    if not (
        summary["n_permutations"]
        == EXPECTED_PERMUTATIONS_PER_SET
    ).all():
        raise ValueError(
            "Unexpected permutation count in "
            "the completed summary."
        )

    return summary


def load_observed_scores() -> pd.DataFrame:
    """
    Read the observed scores previously saved with the permutation
    analysis.
    """
    observed = pd.read_csv(
        OBSERVED_SCORES_FILE,
        sep="\t",
    )

    required_columns = {
        "feature_set",
        "observed_mean_balanced_accuracy",
    }

    missing_columns = (
        required_columns
        - set(observed.columns)
    )

    if missing_columns:
        raise ValueError(
            "Observed-score table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if set(
        observed["feature_set"]
    ) != set(FORMAL_FEATURE_SETS):
        raise ValueError(
            "Observed-score table does not contain "
            "exactly the four formal feature sets."
        )

    return observed


# ============================================================
# Consistency checks
# ============================================================

def validate_observed_score_consistency(
    nested_metrics: pd.DataFrame,
    observed_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> None:
    """
    Check that the previously saved observed scores are unchanged
    across the completed result files.

    This does NOT calculate a new model result.
    It only checks numerical agreement.
    """
    tolerance = 1e-10

    for feature_set in FORMAL_FEATURE_SETS:

        nested_mean = float(
            nested_metrics.loc[
                nested_metrics["feature_set"]
                == feature_set,
                "balanced_accuracy",
            ].mean()
        )

        observed_value = float(
            observed_scores.loc[
                observed_scores["feature_set"]
                == feature_set,
                "observed_mean_balanced_accuracy",
            ].iloc[0]
        )

        summary_value = float(
            permutation_summary.loc[
                permutation_summary["feature_set"]
                == feature_set,
                "observed_mean_balanced_accuracy",
            ].iloc[0]
        )

        if not np.isclose(
            nested_mean,
            observed_value,
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"{feature_set}: completed nested-CV mean "
                "does not match the previously stored "
                "observed permutation-test score."
            )

        if not np.isclose(
            observed_value,
            summary_value,
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"{feature_set}: stored observed score "
                "does not match the completed "
                "permutation-summary score."
            )


def validate_permutation_summary_consistency(
    permutation_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> None:
    """
    Check the already-completed permutation p-values against the
    stored null scores.

    This is a consistency check only. No permutations are rerun.
    """
    tolerance = 1e-10

    for feature_set in FORMAL_FEATURE_SETS:

        null_scores = permutation_scores.loc[
            permutation_scores["feature_set"]
            == feature_set,
            "mean_balanced_accuracy",
        ]

        summary_row = permutation_summary.loc[
            permutation_summary["feature_set"]
            == feature_set
        ].iloc[0]

        observed = float(
            summary_row[
                "observed_mean_balanced_accuracy"
            ]
        )

        n_ge = int(
            (
                null_scores
                >= observed
            ).sum()
        )

        p_value = (
            n_ge + 1
        ) / (
            len(null_scores) + 1
        )

        stored_n_ge = int(
            summary_row[
                "n_null_ge_observed"
            ]
        )

        stored_p_value = float(
            summary_row[
                "permutation_p_value"
            ]
        )

        if n_ge != stored_n_ge:
            raise ValueError(
                f"{feature_set}: completed null-score count "
                "does not match the stored permutation summary."
            )

        if not np.isclose(
            p_value,
            stored_p_value,
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"{feature_set}: completed null scores "
                "do not reproduce the stored permutation "
                "p-value."
            )


# ============================================================
# Figure helper
# ============================================================

def save_figure(
    figure: plt.Figure,
    base_name: str,
) -> tuple[Path, Path]:
    """
    Save dissertation-final copies only.

    IMPORTANT:
    The '_final' suffix prevents the original figures from being
    overwritten.
    """
    png_path = (
        FIGURE_DIR
        / f"{base_name}_final.png"
    )

    svg_path = (
        FIGURE_DIR
        / f"{base_name}_final.svg"
    )

    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return (
        png_path,
        svg_path,
    )


def deterministic_jitter(
    n_points: int,
    seed_offset: int,
    width: float = 0.12,
) -> np.ndarray:
    """
    Generate the same deterministic descriptive jitter pattern
    used by the original plotting script.
    """
    generator = np.random.default_rng(
        RANDOM_STATE
        + seed_offset
    )

    return generator.uniform(
        -width,
        width,
        size=n_points,
    )


# ============================================================
# Figure 08A
# Final nested-CV performance figure
# ============================================================

def plot_nested_cv_balanced_accuracy(
    nested_metrics: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Redraw the existing outer-fold balanced-accuracy figure.

    The plotted values are read directly from the completed
    nested-CV outputs.

    No classifier is refitted.
    """

    figure, axis = plt.subplots(
        figsize=(
            10.5,
            7.2,
        )
    )

    positions = np.arange(
        1,
        len(FORMAL_FEATURE_SETS) + 1,
    )

    distributions = []

    for feature_set in FORMAL_FEATURE_SETS:

        values = (
            nested_metrics.loc[
                nested_metrics["feature_set"]
                == feature_set,
                "balanced_accuracy",
            ]
            .to_numpy(
                dtype=float
            )
        )

        distributions.append(
            values
        )

    # --------------------------------------------------------
    # Same boxplot data as the original figure
    # --------------------------------------------------------

    axis.boxplot(
        distributions,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.7,
        },
    )

    # --------------------------------------------------------
    # Same fold-level points and deterministic jitter
    # --------------------------------------------------------

    for index, (
        position,
        feature_set,
        values,
    ) in enumerate(
        zip(
            positions,
            FORMAL_FEATURE_SETS,
            distributions,
        ),
        start=1,
    ):

        jitter = deterministic_jitter(
            n_points=len(values),
            seed_offset=index,
        )

        axis.scatter(
            position + jitter,
            values,
            s=42,
            alpha=0.65,
            zorder=3,
        )

        mean_value = float(
            np.mean(
                values
            )
        )

        axis.scatter(
            [position],
            [mean_value],
            marker="D",
            s=80,
            zorder=4,
            label=(
                "Observed mean"
                if index == 1
                else None
            ),
        )

        # Same numerical mean as original figure.
        # Only annotation size is increased.
        axis.text(
            position,
            min(
                0.99,
                mean_value + 0.035,
            ),
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )

    # --------------------------------------------------------
    # Same completed dummy baseline
    # --------------------------------------------------------

    dummy_mean = float(
        dummy_metrics[
            "balanced_accuracy"
        ].mean()
    )

    axis.axhline(
        dummy_mean,
        linestyle="--",
        linewidth=1.6,
        label=(
            "Dummy prior mean = "
            f"{dummy_mean:.3f}"
        ),
    )

    # --------------------------------------------------------
    # Thesis-facing display labels
    # --------------------------------------------------------

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            FEATURE_SET_LABELS[
                feature_set
            ]
            for feature_set
            in FORMAL_FEATURE_SETS
        ],
        rotation=0,
        fontsize=TICK_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Outer-fold balanced accuracy",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_xlabel(
        "Feature set",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Nested-CV classification performance",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    # Same y-axis limits as original figure.
    axis.set_ylim(
        0.25,
        1.0,
    )

    axis.tick_params(
        axis="y",
        labelsize=TICK_LABEL_SIZE,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="upper left",
        fontsize=LEGEND_SIZE,
    )

    # --------------------------------------------------------
    # Same interpretation note, larger and wrapped
    # --------------------------------------------------------

    axis.text(
        0.5,
        -0.17,
        (
            "Points show the 25 repeated outer-CV test-fold scores "
            "per feature set; they are descriptive and are not "
            "independent replicates."
        ),
        transform=axis.transAxes,
        fontsize=FOOTNOTE_SIZE,
        ha="center",
        va="top",
        wrap=True,
    )

    # Leave additional bottom space for the enlarged note.
    figure.subplots_adjust(
        bottom=0.22,
        left=0.13,
        right=0.97,
        top=0.90,
    )

    return save_figure(
        figure=figure,
        base_name=(
            "08A_nested_cv_balanced_accuracy"
        ),
    )


# ============================================================
# Figure 08B
# Final permutation figure
# ============================================================

def plot_permutation_distributions(
    permutation_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Redraw the completed label-permutation figure.

    Every plotted null point is read directly from the completed
    permutation output.

    No labels are permuted here.
    No nested CV is rerun here.
    """

    figure, axis = plt.subplots(
        figsize=(
            10.5,
            7.2,
        )
    )

    positions = np.arange(
        1,
        len(FORMAL_FEATURE_SETS) + 1,
    )

    distributions = []

    for feature_set in FORMAL_FEATURE_SETS:

        values = (
            permutation_scores.loc[
                permutation_scores["feature_set"]
                == feature_set,
                "mean_balanced_accuracy",
            ]
            .to_numpy(
                dtype=float
            )
        )

        distributions.append(
            values
        )

    # --------------------------------------------------------
    # Same completed null distributions
    # --------------------------------------------------------

    axis.boxplot(
        distributions,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.7,
        },
    )

    for index, (
        position,
        feature_set,
        null_values,
    ) in enumerate(
        zip(
            positions,
            FORMAL_FEATURE_SETS,
            distributions,
        ),
        start=1,
    ):

        jitter = deterministic_jitter(
            n_points=len(
                null_values
            ),
            seed_offset=100 + index,
        )

        axis.scatter(
            position + jitter,
            null_values,
            s=42,
            alpha=0.65,
            zorder=3,
        )

        summary_row = (
            permutation_summary.loc[
                permutation_summary["feature_set"]
                == feature_set
            ]
            .iloc[0]
        )

        observed = float(
            summary_row[
                "observed_mean_balanced_accuracy"
            ]
        )

        p_value = float(
            summary_row[
                "permutation_p_value"
            ]
        )

        n_ge = int(
            summary_row[
                "n_null_ge_observed"
            ]
        )

        n_permutations = int(
            summary_row[
                "n_permutations"
            ]
        )

        axis.scatter(
            [position],
            [observed],
            marker="D",
            s=90,
            zorder=5,
            label=(
                "Observed nested-CV mean"
                if index == 1
                else None
            ),
        )

        # Same annotation values as original.
        # Only visual spacing and font size are changed.
        annotation_y = min(
            0.735,
            max(
                observed,
                float(
                    np.max(
                        null_values
                    )
                ),
            )
            + 0.030,
        )

        axis.text(
            position,
            annotation_y,
            (
                f"{n_ge}/{n_permutations} ≥ observed\n"
                f"p = {p_value:.3f}"
            ),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )

    # --------------------------------------------------------
    # Same balanced-accuracy reference line
    # --------------------------------------------------------

    axis.axhline(
        0.5,
        linestyle="--",
        linewidth=1.5,
        label="Balanced accuracy = 0.5",
    )

    # --------------------------------------------------------
    # Thesis-facing terminology
    # --------------------------------------------------------

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            FEATURE_SET_LABELS[
                feature_set
            ]
            for feature_set
            in FORMAL_FEATURE_SETS
        ],
        fontsize=TICK_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Mean outer-fold balanced accuracy",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_xlabel(
        "Feature set",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Observed performance versus label-permutation null distributions",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    # Same scientific display range as original.
    axis.set_ylim(
        0.30,
        0.75,
    )

    axis.tick_params(
        axis="y",
        labelsize=TICK_LABEL_SIZE,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="lower left",
        fontsize=LEGEND_SIZE,
    )

    # --------------------------------------------------------
    # Same explanatory note, larger for thesis readability
    # --------------------------------------------------------

    axis.text(
        0.5,
        -0.17,
        (
            "Each null point represents one complete label permutation "
            "with the full nested-CV procedure rerun.\n"
            "Thirty permutations were used per feature set."
        ),
        transform=axis.transAxes,
        fontsize=FOOTNOTE_SIZE,
        ha="center",
        va="top",
    )

    figure.subplots_adjust(
        bottom=0.22,
        left=0.13,
        right=0.97,
        top=0.90,
    )

    return save_figure(
        figure=figure,
        base_name=(
            "08B_label_permutation_balanced_accuracy"
        ),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Regenerate only the final dissertation-formatted figures.

    Completed scientific results are read from disk.
    No model analysis is rerun.
    """

    log("=" * 72)
    log("Final dissertation classifier figure regeneration")
    log("=" * 72)

    log()
    log("This script will NOT:")
    log("  - refit classifiers")
    log("  - rerun nested CV")
    log("  - rerun feature selection")
    log("  - rerun label permutations")
    log("  - overwrite completed result tables")
    log("  - overwrite original figures")

    log()
    log("Checking completed analysis files...")

    validate_input_paths()
    prepare_output_directory()

    log("Completed input files found.")

    # --------------------------------------------------------
    # Read completed results
    # --------------------------------------------------------

    log()
    log("Reading completed nested-CV results...")

    nested_metrics = (
        load_nested_cv_metrics()
    )

    dummy_metrics = (
        load_dummy_metrics()
    )

    log(
        "Nested-CV logistic-regression rows: "
        f"{len(nested_metrics)}"
    )

    log(
        "Dummy outer-fold rows: "
        f"{len(dummy_metrics)}"
    )

    log()
    log("Reading completed permutation results...")

    permutation_scores = (
        load_permutation_scores()
    )

    permutation_summary = (
        load_permutation_summary()
    )

    observed_scores = (
        load_observed_scores()
    )

    log(
        "Permutation-score rows: "
        f"{len(permutation_scores)}"
    )

    # --------------------------------------------------------
    # Consistency checks only
    # --------------------------------------------------------

    log()
    log(
        "Checking consistency of previously completed "
        "observed scores..."
    )

    validate_observed_score_consistency(
        nested_metrics=nested_metrics,
        observed_scores=observed_scores,
        permutation_summary=permutation_summary,
    )

    log("Observed-score consistency check passed.")

    log()
    log(
        "Checking consistency of previously completed "
        "permutation summaries..."
    )

    validate_permutation_summary_consistency(
        permutation_scores=(
            permutation_scores
        ),
        permutation_summary=(
            permutation_summary
        ),
    )

    log("Permutation-summary consistency check passed.")

    # --------------------------------------------------------
    # Display numerical values before plotting
    #
    # This provides an easy visual check that the same completed
    # results are being used.
    # --------------------------------------------------------

    log()
    log("=" * 72)
    log("Completed observed balanced-accuracy values")
    log("=" * 72)

    for feature_set in FORMAL_FEATURE_SETS:

        values = nested_metrics.loc[
            nested_metrics["feature_set"]
            == feature_set,
            "balanced_accuracy",
        ]

        log(
            f"{FEATURE_SET_LABELS[feature_set]}: "
            f"{values.mean():.3f} ± "
            f"{values.std(ddof=1):.3f}"
        )

    dummy_mean = float(
        dummy_metrics[
            "balanced_accuracy"
        ].mean()
    )

    log(
        f"Dummy prior mean: "
        f"{dummy_mean:.3f}"
    )

    # --------------------------------------------------------
    # Figure 08A
    # --------------------------------------------------------

    log()
    log("Regenerating Figure 08A...")

    figure_08a_png, figure_08a_svg = (
        plot_nested_cv_balanced_accuracy(
            nested_metrics=nested_metrics,
            dummy_metrics=dummy_metrics,
        )
    )

    log(
        f"Written: {figure_08a_png}"
    )

    log(
        f"Written: {figure_08a_svg}"
    )

    # --------------------------------------------------------
    # Figure 08B
    # --------------------------------------------------------

    log()
    log("Regenerating Figure 08B...")

    figure_08b_png, figure_08b_svg = (
        plot_permutation_distributions(
            permutation_scores=(
                permutation_scores
            ),
            permutation_summary=(
                permutation_summary
            ),
        )
    )

    log(
        f"Written: {figure_08b_png}"
    )

    log(
        f"Written: {figure_08b_svg}"
    )

    # --------------------------------------------------------
    # Final safety summary
    # --------------------------------------------------------

    log()
    log("=" * 72)
    log("Final figure regeneration completed")
    log("=" * 72)

    log()
    log("Generated:")
    log(
        "  08A_nested_cv_balanced_accuracy_final.png"
    )
    log(
        "  08A_nested_cv_balanced_accuracy_final.svg"
    )
    log(
        "  08B_label_permutation_balanced_accuracy_final.png"
    )
    log(
        "  08B_label_permutation_balanced_accuracy_final.svg"
    )

    log()
    log("Scientific analysis unchanged:")
    log("  models refitted: False")
    log("  nested CV rerun: False")
    log("  feature selection rerun: False")
    log("  permutations rerun: False")
    log("  completed TSV results modified: False")
    log("  original figures overwritten: False")

    log()
    log("Done.")


if __name__ == "__main__":
    main()