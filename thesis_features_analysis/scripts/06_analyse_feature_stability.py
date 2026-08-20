"""
Analyse feature-selection stability from the final batch0100 nested CV.

This script does not train or refit any machine-learning model.

It reads the feature selections saved by script 05 and asks:

1. How often was each feature selected across the outer folds?
2. When a feature was selected, was its logistic-regression coefficient
   direction consistent?
3. How similar were the selected feature sets between outer folds?
4. Were the three pre-specified residue-residue candidates stable?
5. Which residue-residue features are most suitable for subsequent
   structural interpretation?

The analysis uses the final repeated nested-CV results:

    5 outer folds x 5 repeats = 25 outer models per feature set.

Important
---------
The batch0100 low/high labels originate from old RF-predicted barriers.
Feature stability therefore describes reproducibility with respect to
the old-model classification task, not causal determinants of measured
or improved QM/MM activation barriers.

Jaccard similarity is reported as a supplementary whole-set stability
measure. Because the number of selected features k varies between outer
folds, feature-level selection frequency remains the main stability
measure.
"""

from __future__ import annotations

import itertools
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# General configuration
# ============================================================

BATCH = "batch0100"

EXPECTED_OUTER_SPLITS_PER_SET = 25

FORMAL_FEATURE_SETS = (
    "residue_ligand",
    "residue_cofactor",
    "residue_residue",
    "combined_all",
)

TARGET_FEATURES = (
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
)

TOP_N_FOR_FIGURES = 15


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FINAL_CV_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "nested_cv"
    / "final"
)

SELECTED_FEATURES_FILE = (
    FINAL_CV_DIR
    / "nested_cv_selected_features.tsv"
)

BEST_PARAMETERS_FILE = (
    FINAL_CV_DIR
    / "nested_cv_best_parameters.tsv"
)

FEATURE_SET_MANIFEST_FILE = (
    FINAL_CV_DIR
    / "feature_set_manifest.tsv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "stability"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

FIGURE_SOURCE_DIR = (
    OUTPUT_DIR
    / "figure_source_data"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_SOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

OUT_STABILITY_ALL = (
    TABLE_DIR
    / "feature_selection_stability_all.tsv"
)

OUT_CANDIDATES = (
    TABLE_DIR
    / "candidate_feature_stability.tsv"
)

OUT_JACCARD_PAIRS = (
    TABLE_DIR
    / "pairwise_selected_feature_jaccard.tsv"
)

OUT_JACCARD_SUMMARY = (
    TABLE_DIR
    / "feature_set_jaccard_summary.tsv"
)

OUT_FEATURE_SET_SUMMARY = (
    TABLE_DIR
    / "feature_set_stability_summary.tsv"
)

OUT_FIGURE_MANIFEST = (
    TABLE_DIR
    / "06_figure_manifest.tsv"
)

OUT_RUN_SUMMARY = (
    TABLE_DIR
    / "06_feature_stability_summary.tsv"
)


# ============================================================
# Console helper
# ============================================================

def log(
    message: str = "",
) -> None:
    """Print a console message immediately."""
    print(
        message,
        flush=True,
    )


# ============================================================
# Input validation
# ============================================================

def validate_input_paths() -> None:
    """Confirm that all required script-05 outputs exist."""
    required_files = [
        SELECTED_FEATURES_FILE,
        BEST_PARAMETERS_FILE,
        FEATURE_SET_MANIFEST_FILE,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required script-05 outputs are missing:\n"
            + "\n".join(
                str(file_path)
                for file_path in missing_files
            )
        )


def validate_selected_features(
    selected: pd.DataFrame,
) -> None:
    """Validate the selected-feature records from the final nested CV."""
    required_columns = {
        "feature_set",
        "outer_repeat",
        "outer_fold",
        "outer_split",
        "feature",
        "coefficient_for_low_class",
        "absolute_coefficient",
        "coefficient_direction",
        "f_score_outer_training",
    }

    missing_columns = (
        required_columns
        - set(selected.columns)
    )

    if missing_columns:
        raise ValueError(
            "Selected-feature table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    unexpected_sets = (
        set(
            selected[
                "feature_set"
            ].dropna()
        )
        - set(FORMAL_FEATURE_SETS)
    )

    if unexpected_sets:
        raise ValueError(
            "Unexpected feature sets in selected-feature table: "
            f"{sorted(unexpected_sets)}"
        )

    duplicate_rows = selected.duplicated(
        subset=[
            "feature_set",
            "outer_split",
            "feature",
        ]
    )

    if duplicate_rows.any():
        duplicated = selected.loc[
            duplicate_rows,
            [
                "feature_set",
                "outer_split",
                "feature",
            ],
        ]

        raise ValueError(
            "The same feature was recorded more than once "
            "within an outer split:\n"
            + duplicated.to_string(
                index=False
            )
        )


def validate_best_parameters(
    parameters: pd.DataFrame,
) -> None:
    """Confirm exactly 25 final outer models for each feature set."""
    required_columns = {
        "feature_set",
        "outer_repeat",
        "outer_fold",
        "outer_split",
        "best_k",
        "best_c",
        "n_selected_features",
    }

    missing_columns = (
        required_columns
        - set(parameters.columns)
    )

    if missing_columns:
        raise ValueError(
            "Best-parameter table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_outer_models = (
        parameters.duplicated(
            subset=[
                "feature_set",
                "outer_split",
            ]
        )
    )

    if duplicate_outer_models.any():
        raise ValueError(
            "Duplicated feature-set/outer-split rows "
            "were found in best parameters."
        )

    for feature_set in FORMAL_FEATURE_SETS:
        subset = parameters[
            parameters[
                "feature_set"
            ]
            == feature_set
        ]

        n_outer_splits = subset[
            "outer_split"
        ].nunique()

        if (
            n_outer_splits
            != EXPECTED_OUTER_SPLITS_PER_SET
        ):
            raise ValueError(
                f"{feature_set}: expected "
                f"{EXPECTED_OUTER_SPLITS_PER_SET} "
                f"outer splits, observed "
                f"{n_outer_splits}."
            )


def validate_feature_manifest(
    manifest: pd.DataFrame,
) -> None:
    """Validate the row-level feature-set manifest from script 05."""
    required_columns = {
        "feature_set",
        "feature",
        "n_features_in_set",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature-set manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    duplicates = manifest.duplicated(
        subset=[
            "feature_set",
            "feature",
        ]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicated feature membership was found "
            "in feature_set_manifest.tsv."
        )

    for feature_set in FORMAL_FEATURE_SETS:
        if feature_set not in set(
            manifest["feature_set"]
        ):
            raise ValueError(
                "Feature set missing from manifest: "
                f"{feature_set}"
            )


# ============================================================
# Load data
# ============================================================

def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and validate the three script-05 input tables."""
    validate_input_paths()

    selected = pd.read_csv(
        SELECTED_FEATURES_FILE,
        sep="\t",
    )

    parameters = pd.read_csv(
        BEST_PARAMETERS_FILE,
        sep="\t",
    )

    manifest = pd.read_csv(
        FEATURE_SET_MANIFEST_FILE,
        sep="\t",
    )

    validate_selected_features(
        selected
    )

    validate_best_parameters(
        parameters
    )

    validate_feature_manifest(
        manifest
    )

    return (
        selected,
        parameters,
        manifest,
    )


# ============================================================
# Feature-level stability
# ============================================================

def determine_dominant_direction(
    positive_fraction: float,
    negative_fraction: float,
    zero_fraction: float,
) -> tuple[str, float]:
    """Return the most frequent coefficient direction and its fraction."""
    direction_fractions = {
        "towards_low": (
            positive_fraction
        ),
        "towards_high": (
            negative_fraction
        ),
        "zero": (
            zero_fraction
        ),
    }

    dominant_direction = max(
        direction_fractions,
        key=direction_fractions.get,
    )

    dominant_fraction = float(
        direction_fractions[
            dominant_direction
        ]
    )

    return (
        dominant_direction,
        dominant_fraction,
    )


def summarise_selected_feature(
    selected_feature_rows: pd.DataFrame,
    total_outer_splits: int,
) -> dict[str, Any]:
    """Summarise stability statistics for one selected feature."""
    n_selected = selected_feature_rows[
        "outer_split"
    ].nunique()

    coefficients = pd.to_numeric(
        selected_feature_rows[
            "coefficient_for_low_class"
        ],
        errors="coerce",
    )

    absolute_coefficients = pd.to_numeric(
        selected_feature_rows[
            "absolute_coefficient"
        ],
        errors="coerce",
    )

    f_scores = pd.to_numeric(
        selected_feature_rows[
            "f_score_outer_training"
        ],
        errors="coerce",
    )

    directions = selected_feature_rows[
        "coefficient_direction"
    ].astype(str)

    positive_fraction = float(
        (
            directions
            == "towards_low"
        ).mean()
    )

    negative_fraction = float(
        (
            directions
            == "towards_high"
        ).mean()
    )

    zero_fraction = float(
        (
            directions
            == "zero"
        ).mean()
    )

    (
        dominant_direction,
        dominant_direction_fraction,
    ) = determine_dominant_direction(
        positive_fraction=(
            positive_fraction
        ),
        negative_fraction=(
            negative_fraction
        ),
        zero_fraction=(
            zero_fraction
        ),
    )

    return {
        "selected_outer_folds": (
            int(n_selected)
        ),
        "total_outer_folds": (
            int(total_outer_splits)
        ),
        "selection_frequency": (
            n_selected
            / total_outer_splits
        ),
        "mean_coefficient_for_low": (
            coefficients.mean()
        ),
        "median_coefficient_for_low": (
            coefficients.median()
        ),
        "coefficient_std": (
            coefficients.std(
                ddof=1
            )
        ),
        "mean_absolute_coefficient": (
            absolute_coefficients.mean()
        ),
        "median_absolute_coefficient": (
            absolute_coefficients.median()
        ),
        "positive_direction_fraction": (
            positive_fraction
        ),
        "negative_direction_fraction": (
            negative_fraction
        ),
        "zero_direction_fraction": (
            zero_fraction
        ),
        "dominant_direction": (
            dominant_direction
        ),
        "dominant_direction_fraction": (
            dominant_direction_fraction
        ),
        "mean_outer_training_f_score": (
            f_scores.mean()
        ),
        "median_outer_training_f_score": (
            f_scores.median()
        ),
    }


def build_feature_stability_table(
    selected: pd.DataFrame,
    parameters: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate feature-level selection stability.

    All manifest features are retained, including features that were
    never selected. This makes zero selection frequencies explicit.
    """
    rows = []

    for feature_set in FORMAL_FEATURE_SETS:
        manifest_subset = manifest[
            manifest[
                "feature_set"
            ]
            == feature_set
        ].copy()

        selected_subset = selected[
            selected[
                "feature_set"
            ]
            == feature_set
        ].copy()

        parameter_subset = parameters[
            parameters[
                "feature_set"
            ]
            == feature_set
        ]

        total_outer_splits = (
            parameter_subset[
                "outer_split"
            ].nunique()
        )

        for _, manifest_row in (
            manifest_subset.iterrows()
        ):
            feature = manifest_row[
                "feature"
            ]

            feature_rows = (
                selected_subset[
                    selected_subset[
                        "feature"
                    ]
                    == feature
                ]
            )

            if feature_rows.empty:
                summary = {
                    "selected_outer_folds": 0,
                    "total_outer_folds": (
                        total_outer_splits
                    ),
                    "selection_frequency": 0.0,
                    "mean_coefficient_for_low": (
                        np.nan
                    ),
                    "median_coefficient_for_low": (
                        np.nan
                    ),
                    "coefficient_std": (
                        np.nan
                    ),
                    "mean_absolute_coefficient": (
                        np.nan
                    ),
                    "median_absolute_coefficient": (
                        np.nan
                    ),
                    "positive_direction_fraction": (
                        np.nan
                    ),
                    "negative_direction_fraction": (
                        np.nan
                    ),
                    "zero_direction_fraction": (
                        np.nan
                    ),
                    "dominant_direction": (
                        "not_selected"
                    ),
                    "dominant_direction_fraction": (
                        np.nan
                    ),
                    "mean_outer_training_f_score": (
                        np.nan
                    ),
                    "median_outer_training_f_score": (
                        np.nan
                    ),
                }

            else:
                summary = (
                    summarise_selected_feature(
                        selected_feature_rows=(
                            feature_rows
                        ),
                        total_outer_splits=(
                            total_outer_splits
                        ),
                    )
                )

            rows.append(
                {
                    "feature_set": (
                        feature_set
                    ),
                    "feature": (
                        feature
                    ),
                    "n_features_in_set": int(
                        manifest_row[
                            "n_features_in_set"
                        ]
                    ),
                    **summary,
                }
            )

    stability = pd.DataFrame(
        rows
    )

    stability = (
        stability.sort_values(
            by=[
                "feature_set",
                "selection_frequency",
                "dominant_direction_fraction",
                "median_absolute_coefficient",
                "feature",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    stability[
        "stability_rank_within_feature_set"
    ] = (
        stability.groupby(
            "feature_set"
        )
        .cumcount()
        + 1
    )

    return stability


# ============================================================
# Selected-set Jaccard stability
# ============================================================

def build_selected_sets(
    selected: pd.DataFrame,
    parameters: pd.DataFrame,
    feature_set: str,
) -> dict[int, set[str]]:
    """
    Build a selected-feature set for each outer split.

    The best-parameter table supplies the complete list of outer splits,
    ensuring that even an unexpectedly empty selected set would be
    represented explicitly.
    """
    parameter_subset = (
        parameters[
            parameters[
                "feature_set"
            ]
            == feature_set
        ]
        .sort_values(
            "outer_split"
        )
    )

    selected_subset = selected[
        selected[
            "feature_set"
        ]
        == feature_set
    ]

    selected_sets: dict[
        int,
        set[str],
    ] = {}

    for outer_split in (
        parameter_subset[
            "outer_split"
        ].astype(int)
    ):
        feature_names = set(
            selected_subset.loc[
                selected_subset[
                    "outer_split"
                ]
                == outer_split,
                "feature",
            ].astype(str)
        )

        selected_sets[
            outer_split
        ] = feature_names

    return selected_sets


def calculate_jaccard(
    first_set: set[str],
    second_set: set[str],
) -> float:
    """Calculate Jaccard similarity between two selected-feature sets."""
    union = (
        first_set
        | second_set
    )

    if not union:
        return 1.0

    intersection = (
        first_set
        & second_set
    )

    return (
        len(intersection)
        / len(union)
    )


def build_pairwise_jaccard_table(
    selected: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate all pairwise outer-fold Jaccard similarities."""
    rows = []

    for feature_set in FORMAL_FEATURE_SETS:
        selected_sets = (
            build_selected_sets(
                selected=selected,
                parameters=parameters,
                feature_set=feature_set,
            )
        )

        for (
            first_split,
            second_split,
        ) in itertools.combinations(
            sorted(
                selected_sets
            ),
            2,
        ):
            first_features = (
                selected_sets[
                    first_split
                ]
            )

            second_features = (
                selected_sets[
                    second_split
                ]
            )

            intersection_size = len(
                first_features
                & second_features
            )

            union_size = len(
                first_features
                | second_features
            )

            jaccard = calculate_jaccard(
                first_set=first_features,
                second_set=second_features,
            )

            rows.append(
                {
                    "feature_set": (
                        feature_set
                    ),
                    "outer_split_1": (
                        first_split
                    ),
                    "outer_split_2": (
                        second_split
                    ),
                    "n_selected_split_1": (
                        len(first_features)
                    ),
                    "n_selected_split_2": (
                        len(second_features)
                    ),
                    "intersection_size": (
                        intersection_size
                    ),
                    "union_size": (
                        union_size
                    ),
                    "jaccard_similarity": (
                        jaccard
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def summarise_jaccard(
    pairwise_jaccard: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise whole-set Jaccard stability for each feature set."""
    summary = (
        pairwise_jaccard
        .groupby(
            "feature_set"
        )
        .agg(
            n_pairwise_comparisons=(
                "jaccard_similarity",
                "size",
            ),
            mean_jaccard=(
                "jaccard_similarity",
                "mean",
            ),
            std_jaccard=(
                "jaccard_similarity",
                "std",
            ),
            median_jaccard=(
                "jaccard_similarity",
                "median",
            ),
            min_jaccard=(
                "jaccard_similarity",
                "min",
            ),
            max_jaccard=(
                "jaccard_similarity",
                "max",
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# Candidate-feature table
# ============================================================

def build_candidate_table(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the three pre-specified residue-residue candidates.

    They are reported separately in residue_residue and combined_all
    where available.
    """
    candidate_sets = (
        "residue_residue",
        "combined_all",
    )

    candidate_table = stability[
        stability[
            "feature_set"
        ].isin(
            candidate_sets
        )
        & stability[
            "feature"
        ].isin(
            TARGET_FEATURES
        )
    ].copy()

    expected_pairs = {
        (
            feature_set,
            feature,
        )
        for feature_set in (
            candidate_sets
        )
        for feature in (
            TARGET_FEATURES
        )
    }

    observed_pairs = set(
        zip(
            candidate_table[
                "feature_set"
            ],
            candidate_table[
                "feature"
            ],
        )
    )

    missing_pairs = (
        expected_pairs
        - observed_pairs
    )

    if missing_pairs:
        raise ValueError(
            "Expected candidate features are missing "
            "from the stability table:\n"
            + "\n".join(
                f"{feature_set}: {feature}"
                for (
                    feature_set,
                    feature,
                ) in sorted(
                    missing_pairs
                )
            )
        )

    candidate_table = (
        candidate_table.sort_values(
            by=[
                "feature",
                "feature_set",
            ]
        )
    )

    return candidate_table


# ============================================================
# Feature-set summary
# ============================================================

def build_feature_set_summary(
    stability: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Create compact descriptive stability summaries."""
    rows = []

    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        subset = stability[
            stability[
                "feature_set"
            ]
            == feature_set
        ]

        parameter_subset = (
            parameters[
                parameters[
                    "feature_set"
                ]
                == feature_set
            ]
        )

        selection_frequencies = (
            subset[
                "selection_frequency"
            ]
        )

        rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "n_candidate_features": (
                    len(subset)
                ),
                "n_outer_splits": (
                    parameter_subset[
                        "outer_split"
                    ].nunique()
                ),
                "mean_selected_features_per_outer_fold": (
                    parameter_subset[
                        "n_selected_features"
                    ].mean()
                ),
                "median_selected_features_per_outer_fold": (
                    parameter_subset[
                        "n_selected_features"
                    ].median()
                ),
                "n_features_selected_at_least_once": int(
                    (
                        selection_frequencies
                        > 0
                    ).sum()
                ),
                "n_features_frequency_ge_0_20": int(
                    (
                        selection_frequencies
                        >= 0.20
                    ).sum()
                ),
                "n_features_frequency_ge_0_50": int(
                    (
                        selection_frequencies
                        >= 0.50
                    ).sum()
                ),
                "n_features_frequency_ge_0_80": int(
                    (
                        selection_frequencies
                        >= 0.80
                    ).sum()
                ),
                "n_features_frequency_eq_1": int(
                    np.isclose(
                        selection_frequencies,
                        1.0,
                    ).sum()
                ),
                "maximum_selection_frequency": (
                    selection_frequencies.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Figures
# ============================================================

def save_figure(
    figure: plt.Figure,
    output_stem: str,
) -> tuple[Path, Path]:
    """Save a figure as 300-dpi PNG and vector SVG."""
    png_file = (
        FIGURE_DIR
        / f"{output_stem}.png"
    )

    svg_file = (
        FIGURE_DIR
        / f"{output_stem}.svg"
    )

    figure.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_file,
        bbox_inches="tight",
    )

    return (
        png_file,
        svg_file,
    )


def select_top_stable_features(
    stability: pd.DataFrame,
    feature_set: str,
    top_n: int,
) -> pd.DataFrame:
    """Select the most frequently selected features for plotting."""
    subset = stability[
        stability[
            "feature_set"
        ]
        == feature_set
    ].copy()

    subset = (
        subset.sort_values(
            by=[
                "selection_frequency",
                "dominant_direction_fraction",
                "median_absolute_coefficient",
                "feature",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .head(
            top_n
        )
        .copy()
    )

    if subset.empty:
        raise ValueError(
            "No stability results available for "
            f"{feature_set}."
        )

    return subset


def plot_selection_frequency(
    plot_data: pd.DataFrame,
    feature_set: str,
    output_stem: str,
) -> tuple[Path, Path]:
    """
    Plot selection frequency for the most stable features.

    The annotation also reports the dominant coefficient-direction
    consistency among the folds in which each feature was selected.
    """
    display_data = (
        plot_data.sort_values(
            by=[
                "selection_frequency",
                "dominant_direction_fraction",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .copy()
    )

    figure_height = max(
        6.0,
        0.40 * len(
            display_data
        )
        + 1.8,
    )

    figure, axis = plt.subplots(
        figsize=(
            9.5,
            figure_height,
        )
    )

    axis.barh(
        display_data[
            "feature"
        ],
        display_data[
            "selection_frequency"
        ],
    )

    axis.set_xlim(
        0,
        1.10,
    )

    axis.set_xlabel(
        "Selection frequency across 25 outer folds"
    )

    axis.set_ylabel(
        "Structural distance feature"
    )

    readable_feature_set = (
        feature_set.replace(
            "_",
            "–",
        )
    )

    axis.set_title(
        "Most stable "
        f"{readable_feature_set} "
        "features"
    )

    axis.axvline(
        0.5,
        linewidth=1,
        linestyle="--",
    )

    axis.axvline(
        0.8,
        linewidth=1,
        linestyle=":",
    )

    for position, (_, row) in enumerate(
        display_data.iterrows()
    ):
        frequency = float(
            row[
                "selection_frequency"
            ]
        )

        consistency = row[
            "dominant_direction_fraction"
        ]

        if pd.isna(
            consistency
        ):
            annotation = (
                f"{frequency:.2f}"
            )

        else:
            annotation = (
                f"{frequency:.2f} "
                f"(dir. {consistency:.2f})"
            )

        axis.text(
            frequency + 0.015,
            position,
            annotation,
            va="center",
            ha="left",
            fontsize=8,
        )

    figure.tight_layout()

    output_files = save_figure(
        figure=figure,
        output_stem=(
            output_stem
        ),
    )

    plt.close(
        figure
    )

    return output_files


def create_stability_figures(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Create only the two stability figures needed for interpretation."""
    figure_rows = []

    figure_configuration = [
        {
            "feature_set": (
                "residue_residue"
            ),
            "output_stem": (
                "top15_residue_residue_"
                "selection_frequency"
            ),
        },
        {
            "feature_set": (
                "combined_all"
            ),
            "output_stem": (
                "top15_combined_all_"
                "selection_frequency"
            ),
        },
    ]

    for configuration in (
        figure_configuration
    ):
        feature_set = configuration[
            "feature_set"
        ]

        plot_data = (
            select_top_stable_features(
                stability=stability,
                feature_set=feature_set,
                top_n=TOP_N_FOR_FIGURES,
            )
        )

        source_file = (
            FIGURE_SOURCE_DIR
            / (
                f"06_{feature_set}_"
                "stability_source.tsv"
            )
        )

        plot_data.to_csv(
            source_file,
            sep="\t",
            index=False,
        )

        (
            png_file,
            svg_file,
        ) = plot_selection_frequency(
            plot_data=plot_data,
            feature_set=feature_set,
            output_stem=configuration[
                "output_stem"
            ],
        )

        figure_rows.append(
            {
                "figure_id": (
                    f"06_{feature_set}_stability"
                ),
                "figure_role": (
                    "feature_selection_stability"
                ),
                "feature_set": (
                    feature_set
                ),
                "n_features_shown": (
                    len(plot_data)
                ),
                "primary_measure": (
                    "selection_frequency"
                ),
                "secondary_annotation": (
                    "dominant coefficient-direction "
                    "fraction among selected folds"
                ),
                "source_data": (
                    str(source_file)
                ),
                "png_file": (
                    str(png_file)
                ),
                "svg_file": (
                    str(svg_file)
                ),
            }
        )

        log(
            f"Saved stability figure: "
            f"{feature_set}"
        )

    return pd.DataFrame(
        figure_rows
    )


# ============================================================
# Per-set table saving
# ============================================================

def save_per_feature_set_tables(
    stability: pd.DataFrame,
) -> None:
    """Save one convenient stability table for each formal feature set."""
    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        subset = stability[
            stability[
                "feature_set"
            ]
            == feature_set
        ].copy()

        output_file = (
            TABLE_DIR
            / (
                "feature_selection_stability_"
                f"{feature_set}.tsv"
            )
        )

        subset.to_csv(
            output_file,
            sep="\t",
            index=False,
        )


# ============================================================
# Run summary
# ============================================================

def save_run_summary(
    stability: pd.DataFrame,
    pairwise_jaccard: pd.DataFrame,
    candidate_table: pd.DataFrame,
) -> None:
    """Save key provenance and interpretation details."""
    settings: dict[
        str,
        Any,
    ] = {
        "batch": BATCH,
        "selected_features_source": str(
            SELECTED_FEATURES_FILE
        ),
        "best_parameters_source": str(
            BEST_PARAMETERS_FILE
        ),
        "feature_manifest_source": str(
            FEATURE_SET_MANIFEST_FILE
        ),
        "analysis_type": (
            "post-hoc analysis of nested-CV "
            "training-fold feature selections"
        ),
        "models_refitted_in_script_06": (
            False
        ),
        "outer_splits_per_feature_set": (
            EXPECTED_OUTER_SPLITS_PER_SET
        ),
        "primary_stability_measure": (
            "feature selection frequency"
        ),
        "supplementary_global_measure": (
            "pairwise Jaccard similarity"
        ),
        "jaccard_limitation": (
            "selected-set sizes vary because k "
            "was tuned independently within "
            "each outer-training fold"
        ),
        "coefficient_reference_class": (
            "low"
        ),
        "target_candidates": (
            "; ".join(
                TARGET_FEATURES
            )
        ),
        "n_scientific_figures": (
            2
        ),
        "n_stability_rows": (
            len(stability)
        ),
        "n_pairwise_jaccard_rows": (
            len(pairwise_jaccard)
        ),
        "n_candidate_rows": (
            len(candidate_table)
        ),
        "barrier_source": (
            "old RF-predicted barriers"
        ),
        "interpretation": (
            "selection reproducibility for "
            "old-model low/high classification; "
            "not causal barrier determinants"
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
    }

    pd.DataFrame(
        [
            {
                "setting": key,
                "value": value,
            }
            for (
                key,
                value,
            ) in settings.items()
        ]
    ).to_csv(
        OUT_RUN_SUMMARY,
        sep="\t",
        index=False,
    )


# ============================================================
# Output validation
# ============================================================

def validate_outputs(
    stability: pd.DataFrame,
    pairwise_jaccard: pd.DataFrame,
    candidate_table: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """Perform final consistency checks before accepting the results."""
    expected_stability_rows = len(
        manifest
    )

    if len(
        stability
    ) != expected_stability_rows:
        raise RuntimeError(
            "Unexpected stability-table row count: "
            f"expected {expected_stability_rows}, "
            f"observed {len(stability)}."
        )

    expected_pairwise_per_set = (
        EXPECTED_OUTER_SPLITS_PER_SET
        * (
            EXPECTED_OUTER_SPLITS_PER_SET
            - 1
        )
        // 2
    )

    expected_total_pairs = (
        expected_pairwise_per_set
        * len(
            FORMAL_FEATURE_SETS
        )
    )

    if len(
        pairwise_jaccard
    ) != expected_total_pairs:
        raise RuntimeError(
            "Unexpected Jaccard row count: "
            f"expected {expected_total_pairs}, "
            f"observed {len(pairwise_jaccard)}."
        )

    if len(
        candidate_table
    ) != 6:
        raise RuntimeError(
            "Expected six candidate rows "
            "(three features x two feature sets), "
            f"observed {len(candidate_table)}."
        )

    if len(
        figure_manifest
    ) != 2:
        raise RuntimeError(
            "Expected exactly two scientific figures."
        )

    invalid_frequencies = stability[
        ~stability[
            "selection_frequency"
        ].between(
            0,
            1,
            inclusive="both",
        )
    ]

    if not invalid_frequencies.empty:
        raise RuntimeError(
            "Selection frequencies outside [0, 1] "
            "were detected."
        )

    invalid_jaccard = pairwise_jaccard[
        ~pairwise_jaccard[
            "jaccard_similarity"
        ].between(
            0,
            1,
            inclusive="both",
        )
    ]

    if not invalid_jaccard.empty:
        raise RuntimeError(
            "Jaccard similarities outside [0, 1] "
            "were detected."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run the complete post-nested-CV stability analysis."""
    log(
        f"Analysing feature-selection stability "
        f"for {BATCH}..."
    )

    (
        selected,
        parameters,
        manifest,
    ) = load_inputs()

    log()
    log(
        "Input records:"
    )

    log(
        f"  selected-feature rows: "
        f"{len(selected)}"
    )

    log(
        f"  outer-model parameter rows: "
        f"{len(parameters)}"
    )

    log(
        f"  feature-manifest rows: "
        f"{len(manifest)}"
    )

    stability = (
        build_feature_stability_table(
            selected=selected,
            parameters=parameters,
            manifest=manifest,
        )
    )

    pairwise_jaccard = (
        build_pairwise_jaccard_table(
            selected=selected,
            parameters=parameters,
        )
    )

    jaccard_summary = (
        summarise_jaccard(
            pairwise_jaccard
        )
    )

    candidate_table = (
        build_candidate_table(
            stability
        )
    )

    feature_set_summary = (
        build_feature_set_summary(
            stability=stability,
            parameters=parameters,
        )
    )

    stability.to_csv(
        OUT_STABILITY_ALL,
        sep="\t",
        index=False,
    )

    save_per_feature_set_tables(
        stability
    )

    candidate_table.to_csv(
        OUT_CANDIDATES,
        sep="\t",
        index=False,
    )

    pairwise_jaccard.to_csv(
        OUT_JACCARD_PAIRS,
        sep="\t",
        index=False,
    )

    jaccard_summary.to_csv(
        OUT_JACCARD_SUMMARY,
        sep="\t",
        index=False,
    )

    feature_set_summary.to_csv(
        OUT_FEATURE_SET_SUMMARY,
        sep="\t",
        index=False,
    )

    figure_manifest = (
        create_stability_figures(
            stability
        )
    )

    figure_manifest.to_csv(
        OUT_FIGURE_MANIFEST,
        sep="\t",
        index=False,
    )

    validate_outputs(
        stability=stability,
        pairwise_jaccard=(
            pairwise_jaccard
        ),
        candidate_table=(
            candidate_table
        ),
        figure_manifest=(
            figure_manifest
        ),
        manifest=manifest,
    )

    save_run_summary(
        stability=stability,
        pairwise_jaccard=(
            pairwise_jaccard
        ),
        candidate_table=(
            candidate_table
        ),
    )

    log()
    log("=" * 72)
    log(
        "Candidate-feature stability"
    )
    log("=" * 72)

    candidate_columns = [
        "feature_set",
        "feature",
        "selected_outer_folds",
        "total_outer_folds",
        "selection_frequency",
        "median_coefficient_for_low",
        "positive_direction_fraction",
        "negative_direction_fraction",
        "dominant_direction",
        "dominant_direction_fraction",
    ]

    log(
        candidate_table[
            candidate_columns
        ].to_string(
            index=False
        )
    )

    log()
    log("=" * 72)
    log(
        "Feature-set stability summary"
    )
    log("=" * 72)

    log(
        feature_set_summary.to_string(
            index=False
        )
    )

    log()
    log("=" * 72)
    log(
        "Pairwise Jaccard summary"
    )
    log("=" * 72)

    log(
        jaccard_summary.to_string(
            index=False
        )
    )

    log()
    log(
        "Scientific figures created: 2"
    )

    log(
        "1. Top 15 residue-residue "
        "selection frequencies"
    )

    log(
        "2. Top 15 combined-all "
        "selection frequencies"
    )

    log()
    log(
        "Saved results to:"
    )

    log(
        str(OUTPUT_DIR)
    )

    log()
    log(
        "Done."
    )


if __name__ == "__main__":
    main()