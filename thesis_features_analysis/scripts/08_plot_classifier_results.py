"""
Create final classifier-result figures for batch0100.

This script is a plotting and result-packaging step only.

It does NOT:
- refit any classifier;
- rerun nested cross-validation;
- perform feature selection;
- tune hyperparameters;
- rerun label permutations;
- recalculate feature-selection stability.

Instead, it reads the formally completed outputs from:

05_run_four_set_nested_cv.py
06_analyse_feature_stability.py
07_run_label_permutation_test.py

and creates a small number of dissertation-ready summary figures.

Figures
-------
Figure 08A
    Outer-fold balanced-accuracy distributions for the four formal
    feature sets, with the prior dummy baseline shown as a reference.

Figure 08B
    Label-permutation null distributions for the four feature sets,
    with the observed nested-CV mean balanced accuracy overlaid.

Important interpretation
------------------------
The batch0100 low/high labels are derived from old RF-predicted barriers.
These classifier results therefore characterise structural patterns
associated with the behaviour of the old RF model rather than measured
or improved-QM/MM activation barriers.

Repeated outer-CV folds are not independent observations. The fold-level
distributions shown in Figure 08A are therefore descriptive and should
not be interpreted as independent-sample inferential distributions.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn


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

FEATURE_SET_LABELS = {
    "residue_ligand": "Residue–ligand",
    "residue_cofactor": "Residue–GTP",
    "residue_residue": "Residue–residue",
    "combined_all": "Combined",
}

EXPECTED_OUTER_FOLDS_PER_SET = 25
EXPECTED_PERMUTATIONS_PER_SET = 30

PRIMARY_METRIC = "balanced_accuracy"

RANDOM_STATE = 42

PNG_DPI = 300


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

STABILITY_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "stability"
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

SOURCE_DATA_DIR = (
    OUTPUT_ROOT
    / "figure_source_data"
)

TABLE_DIR = (
    OUTPUT_ROOT
    / "tables"
)


NESTED_METRICS_FILE = (
    NESTED_CV_DIR
    / "nested_cv_fold_metrics.tsv"
)

DUMMY_METRICS_FILE = (
    NESTED_CV_DIR
    / "nested_cv_dummy_metrics.tsv"
)

STABILITY_CANDIDATE_FILE = (
    STABILITY_DIR
    / "tables"
    / "candidate_feature_stability.tsv"
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
    """Print a message immediately."""
    print(
        message,
        flush=True,
    )


# ============================================================
# Input validation
# ============================================================

def validate_input_paths() -> None:
    """Confirm that all required completed-analysis files exist."""
    required_files = [
        NESTED_METRICS_FILE,
        DUMMY_METRICS_FILE,
        STABILITY_CANDIDATE_FILE,
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


def prepare_output_directories() -> None:
    """Create output directories."""
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Load completed results
# ============================================================

def load_nested_cv_metrics() -> pd.DataFrame:
    """Load and validate the formal script-05 model metrics."""
    metrics = pd.read_csv(
        NESTED_METRICS_FILE,
        sep="\t",
    )

    required_columns = {
        "model_type",
        "feature_set",
        "balanced_accuracy",
        "roc_auc_low",
        "mcc",
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
                f"outer-fold metric rows, observed "
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

    expected_rows = (
        len(FORMAL_FEATURE_SETS)
        * EXPECTED_OUTER_FOLDS_PER_SET
    )

    if len(metrics) != expected_rows:
        raise ValueError(
            "Unexpected total number of formal model rows: "
            f"expected {expected_rows}, observed {len(metrics)}."
        )

    return metrics


def load_dummy_metrics() -> pd.DataFrame:
    """Load and validate the formal script-05 dummy baseline."""
    dummy = pd.read_csv(
        DUMMY_METRICS_FILE,
        sep="\t",
    )

    required_columns = {
        "balanced_accuracy",
    }

    missing_columns = (
        required_columns
        - set(dummy.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dummy metric table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dummy) != EXPECTED_OUTER_FOLDS_PER_SET:
        raise ValueError(
            "Expected 25 dummy outer-fold metric rows, "
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


def load_permutation_scores() -> pd.DataFrame:
    """Load and validate permutation-level null scores from script 07."""
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
                f"permutation rows, observed {len(subset)}."
            )

        unique_ids = sorted(
            subset[
                "permutation_id"
            ].unique()
        )

        expected_ids = list(
            range(
                1,
                EXPECTED_PERMUTATIONS_PER_SET + 1,
            )
        )

        if unique_ids != expected_ids:
            raise ValueError(
                f"{feature_set}: permutation IDs "
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
                f"{feature_set}: permutation BA "
                "outside [0, 1]."
            )

    return scores


def load_permutation_summary() -> pd.DataFrame:
    """Load the formal empirical permutation-test summary."""
    summary = pd.read_csv(
        PERMUTATION_SUMMARY_FILE,
        sep="\t",
    )

    required_columns = {
        "feature_set",
        "observed_mean_balanced_accuracy",
        "null_mean_balanced_accuracy",
        "null_std_balanced_accuracy",
        "null_max_balanced_accuracy",
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
            "Unexpected permutation count in summary."
        )

    return summary


def load_observed_scores() -> pd.DataFrame:
    """Load observed script-05 means saved by script 07."""
    observed = pd.read_csv(
        OBSERVED_SCORES_FILE,
        sep="\t",
    )

    required_columns = {
        "feature_set",
        "observed_mean_balanced_accuracy",
        "observed_mean_roc_auc_low",
        "observed_mean_mcc",
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


def load_candidate_stability() -> pd.DataFrame:
    """Load the script-06 candidate stability table for provenance."""
    candidates = pd.read_csv(
        STABILITY_CANDIDATE_FILE,
        sep="\t",
    )

    required_columns = {
        "feature_set",
        "feature",
        "selection_frequency",
    }

    missing_columns = (
        required_columns
        - set(candidates.columns)
    )

    if missing_columns:
        raise ValueError(
            "Candidate-stability table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return candidates


# ============================================================
# Cross-analysis consistency checks
# ============================================================

def validate_observed_score_consistency(
    nested_metrics: pd.DataFrame,
    observed_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> None:
    """
    Confirm that script-05 means, script-07 observed values and
    permutation-summary observed values are numerically identical.
    """
    tolerance = 1e-10

    for feature_set in FORMAL_FEATURE_SETS:
        nested_mean = (
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
                f"{feature_set}: script-05 nested-CV mean "
                "does not match script-07 observed score."
            )

        if not np.isclose(
            observed_value,
            summary_value,
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"{feature_set}: observed score does not match "
                "permutation-summary observed score."
            )


def validate_permutation_summary_consistency(
    permutation_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> None:
    """Independently reproduce the script-07 empirical p-values."""
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

        if n_ge != int(
            summary_row[
                "n_null_ge_observed"
            ]
        ):
            raise ValueError(
                f"{feature_set}: independently calculated "
                "n_null_ge_observed does not match script 07."
            )

        if not np.isclose(
            p_value,
            float(
                summary_row[
                    "permutation_p_value"
                ]
            ),
            atol=tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"{feature_set}: independently calculated "
                "permutation p-value does not match script 07."
            )


# ============================================================
# Figure-source tables
# ============================================================

def build_performance_source_table(
    nested_metrics: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare exact source data used for Figure 08A."""
    rows = []

    for feature_set in FORMAL_FEATURE_SETS:
        subset = (
            nested_metrics.loc[
                nested_metrics["feature_set"]
                == feature_set
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        for index, row in subset.iterrows():
            rows.append(
                {
                    "source_type": (
                        "logistic_regression_outer_fold"
                    ),
                    "feature_set": (
                        feature_set
                    ),
                    "display_label": (
                        FEATURE_SET_LABELS[
                            feature_set
                        ]
                    ),
                    "fold_record": (
                        index + 1
                    ),
                    "balanced_accuracy": float(
                        row[
                            "balanced_accuracy"
                        ]
                    ),
                }
            )

    dummy_mean = float(
        dummy_metrics[
            "balanced_accuracy"
        ].mean()
    )

    rows.append(
        {
            "source_type": (
                "dummy_reference_mean"
            ),
            "feature_set": (
                "dummy_prior"
            ),
            "display_label": (
                "Dummy prior"
            ),
            "fold_record": (
                np.nan
            ),
            "balanced_accuracy": (
                dummy_mean
            ),
        }
    )

    return pd.DataFrame(
        rows
    )


def build_permutation_source_table(
    permutation_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare exact source data used for Figure 08B."""
    rows = []

    for feature_set in FORMAL_FEATURE_SETS:
        subset = permutation_scores.loc[
            permutation_scores["feature_set"]
            == feature_set
        ].copy()

        summary_row = permutation_summary.loc[
            permutation_summary["feature_set"]
            == feature_set
        ].iloc[0]

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

        for _, row in subset.iterrows():
            rows.append(
                {
                    "feature_set": (
                        feature_set
                    ),
                    "display_label": (
                        FEATURE_SET_LABELS[
                            feature_set
                        ]
                    ),
                    "permutation_id": int(
                        row[
                            "permutation_id"
                        ]
                    ),
                    "null_mean_balanced_accuracy": float(
                        row[
                            "mean_balanced_accuracy"
                        ]
                    ),
                    "observed_mean_balanced_accuracy": (
                        observed
                    ),
                    "n_null_ge_observed": (
                        n_ge
                    ),
                    "permutation_p_value": (
                        p_value
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plot helpers
# ============================================================

def save_figure(
    figure: plt.Figure,
    base_name: str,
) -> tuple[Path, Path]:
    """Save one figure as PNG and SVG."""
    png_path = (
        FIGURE_DIR
        / f"{base_name}.png"
    )

    svg_path = (
        FIGURE_DIR
        / f"{base_name}.svg"
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
    """Generate deterministic horizontal jitter for descriptive points."""
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
# ============================================================

def plot_nested_cv_balanced_accuracy(
    nested_metrics: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Plot descriptive outer-fold balanced-accuracy distributions.

    Individual repeated-CV folds are shown descriptively.
    The figure does not imply that the 25 folds are independent samples.
    """
    figure, axis = plt.subplots(
        figsize=(
            9.2,
            5.8,
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

    axis.boxplot(
        distributions,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.5,
        },
    )

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
            s=28,
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
            s=58,
            zorder=4,
            label=(
                "Observed mean"
                if index == 1
                else None
            ),
        )

        axis.text(
            position,
            min(
                0.99,
                mean_value + 0.035,
            ),
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    dummy_mean = float(
        dummy_metrics[
            "balanced_accuracy"
        ].mean()
    )

    axis.axhline(
        dummy_mean,
        linestyle="--",
        linewidth=1.4,
        label=(
            f"Dummy prior mean = "
            f"{dummy_mean:.3f}"
        ),
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            FEATURE_SET_LABELS[
                feature_set
            ]
            for feature_set in FORMAL_FEATURE_SETS
        ],
        rotation=0,
    )

    axis.set_ylabel(
        "Outer-fold balanced accuracy"
    )

    axis.set_xlabel(
        "Feature set"
    )

    axis.set_title(
        "Nested-CV classification performance"
    )

    axis.set_ylim(
        0.25,
        1.0,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="upper left",
    )

    axis.text(
        0.01,
        -0.16,
        (
            "Points show the 25 repeated outer-CV test-fold scores per "
            "feature set; they are descriptive and are not independent "
            "replicates."
        ),
        transform=axis.transAxes,
        fontsize=8.5,
        va="top",
    )

    figure.tight_layout()

    return save_figure(
        figure=figure,
        base_name=(
            "08A_nested_cv_balanced_accuracy"
        ),
    )


# ============================================================
# Figure 08B
# ============================================================

def plot_permutation_distributions(
    permutation_scores: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Plot permutation-level mean balanced-accuracy null distributions.

    Each point is one complete label permutation, not one outer fold.
    """
    figure, axis = plt.subplots(
        figsize=(
            9.2,
            5.8,
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

    axis.boxplot(
        distributions,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=False,
        medianprops={
            "linewidth": 1.5,
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
            s=28,
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
            s=70,
            zorder=5,
            label=(
                "Observed nested-CV mean"
                if index == 1
                else None
            ),
        )

        annotation_y = min(
            0.94,
            max(
                observed,
                float(
                    np.max(
                        null_values
                    )
                ),
            )
            + 0.045,
        )

        axis.text(
            position,
            annotation_y,
            (
                f"{n_ge}/{n_permutations} ≥ observed\n"
                f"p={p_value:.3f}"
            ),
            ha="center",
            va="bottom",
            fontsize=8.7,
        )

    axis.axhline(
        0.5,
        linestyle="--",
        linewidth=1.3,
        label="Balanced accuracy = 0.5",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            FEATURE_SET_LABELS[
                feature_set
            ]
            for feature_set in FORMAL_FEATURE_SETS
        ]
    )

    axis.set_ylabel(
        "Mean outer-fold balanced accuracy"
    )

    axis.set_xlabel(
        "Feature set"
    )

    axis.set_title(
        "Observed performance versus label-permutation null distributions"
    )

    axis.set_ylim(
        0.30,
        0.75,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        frameon=False,
        loc="lower left",
    )

    axis.text(
        0.5,
        -0.16,
        (
            "Each null point represents one complete label permutation "
            "with the full nested-CV procedure rerun.\n"
            "Thirty permutations were used per feature set."
        ),
        transform=axis.transAxes,
        fontsize=8.5,
        ha="center",
        va="top",
    )

    figure.tight_layout()

    return save_figure(
        figure=figure,
        base_name=(
            "08B_label_permutation_balanced_accuracy"
        ),
    )


# ============================================================
# Summary tables
# ============================================================

def build_classifier_result_summary(
    nested_metrics: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one concise dissertation-oriented numerical summary."""
    rows = []

    for feature_set in FORMAL_FEATURE_SETS:
        nested_subset = nested_metrics.loc[
            nested_metrics["feature_set"]
            == feature_set
        ]

        permutation_row = (
            permutation_summary.loc[
                permutation_summary["feature_set"]
                == feature_set
            ]
            .iloc[0]
        )

        rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "display_label": (
                    FEATURE_SET_LABELS[
                        feature_set
                    ]
                ),
                "n_outer_folds": (
                    len(
                        nested_subset
                    )
                ),
                "mean_balanced_accuracy": (
                    nested_subset[
                        "balanced_accuracy"
                    ].mean()
                ),
                "std_balanced_accuracy": (
                    nested_subset[
                        "balanced_accuracy"
                    ].std(
                        ddof=1
                    )
                ),
                "mean_roc_auc_low": (
                    nested_subset[
                        "roc_auc_low"
                    ].mean()
                ),
                "std_roc_auc_low": (
                    nested_subset[
                        "roc_auc_low"
                    ].std(
                        ddof=1
                    )
                ),
                "mean_mcc": (
                    nested_subset[
                        "mcc"
                    ].mean()
                ),
                "std_mcc": (
                    nested_subset[
                        "mcc"
                    ].std(
                        ddof=1
                    )
                ),
                "permutation_null_mean_ba": float(
                    permutation_row[
                        "null_mean_balanced_accuracy"
                    ]
                ),
                "permutation_null_std_ba": float(
                    permutation_row[
                        "null_std_balanced_accuracy"
                    ]
                ),
                "permutation_null_max_ba": float(
                    permutation_row[
                        "null_max_balanced_accuracy"
                    ]
                ),
                "n_permutations": int(
                    permutation_row[
                        "n_permutations"
                    ]
                ),
                "n_null_ge_observed": int(
                    permutation_row[
                        "n_null_ge_observed"
                    ]
                ),
                "permutation_p_value": float(
                    permutation_row[
                        "permutation_p_value"
                    ]
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary[
        "dummy_mean_balanced_accuracy"
    ] = float(
        dummy_metrics[
            "balanced_accuracy"
        ].mean()
    )

    return summary


def build_figure_manifest(
    figure_records: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    """Create a machine-readable figure manifest."""
    return pd.DataFrame(
        figure_records
    )


def save_run_summary(
    output_file: Path,
    nested_metrics: pd.DataFrame,
    permutation_scores: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
) -> None:
    """Save plotting provenance and scientific-use notes."""
    settings: dict[str, Any] = {
        "batch": BATCH,
        "script_role": (
            "final classifier plotting and result packaging only"
        ),
        "models_refitted": False,
        "nested_cv_rerun": False,
        "feature_selection_rerun": False,
        "permutations_rerun": False,
        "stability_rerun": False,
        "primary_metric": (
            "balanced accuracy"
        ),
        "n_feature_sets": (
            len(
                FORMAL_FEATURE_SETS
            )
        ),
        "n_outer_fold_model_records": (
            len(
                nested_metrics
            )
        ),
        "n_outer_folds_per_feature_set": (
            EXPECTED_OUTER_FOLDS_PER_SET
        ),
        "n_permutation_records": (
            len(
                permutation_scores
            )
        ),
        "n_permutations_per_feature_set": (
            EXPECTED_PERMUTATIONS_PER_SET
        ),
        "dummy_mean_balanced_accuracy": float(
            dummy_metrics[
                "balanced_accuracy"
            ].mean()
        ),
        "figure_08a_interpretation": (
            "descriptive repeated outer-CV balanced-accuracy "
            "distribution; folds are not independent replicates"
        ),
        "figure_08b_interpretation": (
            "permutation-level null distributions; each point "
            "is one complete label permutation"
        ),
        "label_source": (
            "old RF-predicted barriers"
        ),
        "scientific_limitation": (
            "classifier characterises old-model behaviour rather "
            "than measured or improved-QM/MM activation barriers"
        ),
        "python_version": (
            sys.version.replace(
                "\n",
                " ",
            )
        ),
        "platform": (
            platform.platform()
        ),
        "numpy_version": (
            np.__version__
        ),
        "pandas_version": (
            pd.__version__
        ),
        "matplotlib_version": (
            matplotlib.__version__
        ),
        "scikit_learn_version": (
            sklearn.__version__
        ),
    }

    pd.DataFrame(
        [
            {
                "setting": key,
                "value": value,
            }
            for key, value in settings.items()
        ]
    ).to_csv(
        output_file,
        sep="\t",
        index=False,
    )


# ============================================================
# Output validation
# ============================================================

def validate_output_files(
    figure_manifest: pd.DataFrame,
    performance_source: pd.DataFrame,
    permutation_source: pd.DataFrame,
    classifier_summary: pd.DataFrame,
) -> None:
    """Validate expected output dimensions and generated files."""
    expected_performance_rows = (
        len(FORMAL_FEATURE_SETS)
        * EXPECTED_OUTER_FOLDS_PER_SET
        + 1
    )

    if len(
        performance_source
    ) != expected_performance_rows:
        raise RuntimeError(
            "Unexpected Figure 08A source-data row count: "
            f"expected {expected_performance_rows}, "
            f"observed {len(performance_source)}."
        )

    expected_permutation_rows = (
        len(FORMAL_FEATURE_SETS)
        * EXPECTED_PERMUTATIONS_PER_SET
    )

    if len(
        permutation_source
    ) != expected_permutation_rows:
        raise RuntimeError(
            "Unexpected Figure 08B source-data row count: "
            f"expected {expected_permutation_rows}, "
            f"observed {len(permutation_source)}."
        )

    if len(
        classifier_summary
    ) != len(
        FORMAL_FEATURE_SETS
    ):
        raise RuntimeError(
            "Classifier summary must contain "
            "exactly four feature-set rows."
        )

    if len(
        figure_manifest
    ) != 2:
        raise RuntimeError(
            "Expected exactly two formal figures."
        )

    for _, row in figure_manifest.iterrows():
        for column in (
            "png_file",
            "svg_file",
            "source_data_file",
        ):
            path = Path(
                row[
                    column
                ]
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing generated output: {path}"
                )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Create the two final classifier figures."""
    log("=" * 72)
    log(
        "Batch0100 final classifier figures"
    )
    log("=" * 72)

    validate_input_paths()
    prepare_output_directories()

    log()
    log(
        "Loading completed 05, 06 and 07 results..."
    )

    nested_metrics = (
        load_nested_cv_metrics()
    )

    dummy_metrics = (
        load_dummy_metrics()
    )

    permutation_scores = (
        load_permutation_scores()
    )

    permutation_summary = (
        load_permutation_summary()
    )

    observed_scores = (
        load_observed_scores()
    )

    candidate_stability = (
        load_candidate_stability()
    )

    validate_observed_score_consistency(
        nested_metrics=(
            nested_metrics
        ),
        observed_scores=(
            observed_scores
        ),
        permutation_summary=(
            permutation_summary
        ),
    )

    validate_permutation_summary_consistency(
        permutation_scores=(
            permutation_scores
        ),
        permutation_summary=(
            permutation_summary
        ),
    )

    log(
        "Cross-analysis consistency checks passed."
    )

    log()
    log(
        "Creating Figure 08A..."
    )

    performance_source = (
        build_performance_source_table(
            nested_metrics=(
                nested_metrics
            ),
            dummy_metrics=(
                dummy_metrics
            ),
        )
    )

    performance_source_file = (
        SOURCE_DATA_DIR
        / "08A_nested_cv_balanced_accuracy_source.tsv"
    )

    performance_source.to_csv(
        performance_source_file,
        sep="\t",
        index=False,
    )

    figure_08a_png, figure_08a_svg = (
        plot_nested_cv_balanced_accuracy(
            nested_metrics=(
                nested_metrics
            ),
            dummy_metrics=(
                dummy_metrics
            ),
        )
    )

    log(
        f"Saved: {figure_08a_png.name}"
    )

    log(
        f"Saved: {figure_08a_svg.name}"
    )

    log()
    log(
        "Creating Figure 08B..."
    )

    permutation_source = (
        build_permutation_source_table(
            permutation_scores=(
                permutation_scores
            ),
            permutation_summary=(
                permutation_summary
            ),
        )
    )

    permutation_source_file = (
        SOURCE_DATA_DIR
        / "08B_label_permutation_balanced_accuracy_source.tsv"
    )

    permutation_source.to_csv(
        permutation_source_file,
        sep="\t",
        index=False,
    )

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
        f"Saved: {figure_08b_png.name}"
    )

    log(
        f"Saved: {figure_08b_svg.name}"
    )

    classifier_summary = (
        build_classifier_result_summary(
            nested_metrics=(
                nested_metrics
            ),
            dummy_metrics=(
                dummy_metrics
            ),
            permutation_summary=(
                permutation_summary
            ),
        )
    )

    classifier_summary_file = (
        TABLE_DIR
        / "08_classifier_result_summary.tsv"
    )

    classifier_summary.to_csv(
        classifier_summary_file,
        sep="\t",
        index=False,
    )

    figure_records = [
        {
            "figure_id": "08A",
            "title": (
                "Nested-CV classification performance"
            ),
            "scientific_question": (
                "How does balanced accuracy differ across "
                "the four formal structural feature sets?"
            ),
            "png_file": str(
                figure_08a_png
            ),
            "svg_file": str(
                figure_08a_svg
            ),
            "source_data_file": str(
                performance_source_file
            ),
            "primary_measure": (
                "outer-fold balanced accuracy"
            ),
            "interpretation_note": (
                "25 repeated outer-fold values per feature set "
                "are descriptive and not independent replicates"
            ),
        },
        {
            "figure_id": "08B",
            "title": (
                "Observed performance versus "
                "label-permutation null distributions"
            ),
            "scientific_question": (
                "Was the observed mean balanced accuracy "
                "reproduced under randomly permuted labels?"
            ),
            "png_file": str(
                figure_08b_png
            ),
            "svg_file": str(
                figure_08b_svg
            ),
            "source_data_file": str(
                permutation_source_file
            ),
            "primary_measure": (
                "permutation-level mean outer-fold "
                "balanced accuracy"
            ),
            "interpretation_note": (
                "30 complete label permutations per "
                "feature set"
            ),
        },
    ]

    figure_manifest = (
        build_figure_manifest(
            figure_records
        )
    )

    figure_manifest_file = (
        TABLE_DIR
        / "08_figure_manifest.tsv"
    )

    figure_manifest.to_csv(
        figure_manifest_file,
        sep="\t",
        index=False,
    )

    candidate_reference_file = (
        TABLE_DIR
        / "08_stable_candidate_reference.tsv"
    )

    candidate_stability.to_csv(
        candidate_reference_file,
        sep="\t",
        index=False,
    )

    run_summary_file = (
        TABLE_DIR
        / "08_plot_classifier_results_summary.tsv"
    )

    save_run_summary(
        output_file=(
            run_summary_file
        ),
        nested_metrics=(
            nested_metrics
        ),
        permutation_scores=(
            permutation_scores
        ),
        dummy_metrics=(
            dummy_metrics
        ),
    )

    validate_output_files(
        figure_manifest=(
            figure_manifest
        ),
        performance_source=(
            performance_source
        ),
        permutation_source=(
            permutation_source
        ),
        classifier_summary=(
            classifier_summary
        ),
    )

    log()
    log("=" * 72)
    log(
        "Final classifier numerical summary"
    )
    log("=" * 72)

    display_columns = [
        "feature_set",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_roc_auc_low",
        "mean_mcc",
        "permutation_null_mean_ba",
        "permutation_null_max_ba",
        "n_null_ge_observed",
        "permutation_p_value",
    ]

    log(
        classifier_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    log()
    log(
        "Generated formal figures: 2"
    )

    log(
        "  08A_nested_cv_balanced_accuracy"
    )

    log(
        "  08B_label_permutation_balanced_accuracy"
    )

    log()
    log(
        "Figure source rows:"
    )

    log(
        f"  Figure 08A: "
        f"{len(performance_source)}"
    )

    log(
        f"  Figure 08B: "
        f"{len(permutation_source)}"
    )

    log()
    log(
        "Saved results to:"
    )

    log(
        str(
            OUTPUT_ROOT
        )
    )

    log()
    log(
        "No models were refitted."
    )

    log(
        "No feature selection was rerun."
    )

    log(
        "No permutations were rerun."
    )

    log()
    log(
        "Done."
    )


if __name__ == "__main__":
    main()