#!/usr/bin/env python3
"""
03_analyse_prefixed_features_vs_qmmm.py

Exploratory analysis of three pre-fixed batch0100 stable residue-residue
features in the 47 complete improved-QM/MM structures.

IMPORTANT DESIGN PRINCIPLES
---------------------------
The three features were fixed in advance from the independent February
batch0100 nested-CV stability analysis:

    PRO62-PRO1279
    PRO1275-PRO1276
    PRO90-PRO92

This script does NOT:
- screen all 4086 structural features;
- select features using the 47-structure barriers;
- train a classifier or regression model;
- optimise any parameters.

For each pre-fixed feature, the script:
1. quantifies structural variation across the 47 structures;
2. determines whether correlation analysis is mathematically meaningful;
3. computes Pearson and Spearman correlations only for variable features;
4. generates exploratory scatter plots for testable features.

Invariant features are explicitly reported as not testable rather than
passing them to correlation functions.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "02_qmmm47_prefixed_feature_values.tsv"
)

STATS_OUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "03_qmmm47_prefixed_feature_statistics.tsv"
)

THESIS_OUT = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "03_qmmm47_prefixed_feature_thesis_summary.tsv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

LOG_OUT = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "03_analyse_prefixed_features_vs_qmmm.log"
)


# =============================================================================
# Pre-fixed features
# =============================================================================

PREFIXED_FEATURES = [
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
]


STRUCTURAL_LABELS = {
    "PRO62-PRO1279": "GLU62-SER1279",
    "PRO1275-PRO1276": "PHE1275-ARG1276",
    "PRO90-PRO92": "PHE90-ASP92",
}


BATCH0100_STABILITY = {
    "PRO62-PRO1279": {
        "rr_selection_frequency": 1.00,
        "combined_selection_frequency": 1.00,
    },
    "PRO1275-PRO1276": {
        "rr_selection_frequency": 0.88,
        "combined_selection_frequency": 0.84,
    },
    "PRO90-PRO92": {
        "rr_selection_frequency": 0.84,
        "combined_selection_frequency": 0.80,
    },
}


# =============================================================================
# Helper functions
# =============================================================================

def interquartile_range(values):
    """Return the interquartile range of finite numeric values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)


def coefficient_of_variation(values):
    """
    Return sample SD / absolute mean.

    The coefficient of variation is only descriptive here.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    mean = np.mean(values)
    sd = np.std(values, ddof=1)

    if np.isclose(mean, 0.0):
        return np.nan

    return float(sd / abs(mean))


def classify_variation(values):
    """
    Determine whether a feature has sufficient numerical variation
    for correlation analysis.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n_unique = len(np.unique(values))

    if len(values) < 3:
        return "not_testable_too_few_values"

    if n_unique <= 1:
        return "not_testable_invariant"

    if np.isclose(np.std(values, ddof=1), 0.0):
        return "not_testable_near_constant"

    return "testable_variable"


def safe_pearson(x, y):
    """Pearson correlation for finite values only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan, len(x)

    if np.isclose(np.std(x, ddof=1), 0.0):
        return np.nan, np.nan, len(x)

    r, p = stats.pearsonr(x, y)

    return float(r), float(p), len(x)


def safe_spearman(x, y):
    """Spearman rank correlation for finite values only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan, len(x)

    if len(np.unique(x)) <= 1:
        return np.nan, np.nan, len(x)

    rho, p = stats.spearmanr(x, y)

    return float(rho), float(p), len(x)


def plot_feature_vs_barrier(
    df,
    feature,
    structural_label,
    pearson_r,
    pearson_p,
    spearman_rho,
    spearman_p,
):
    """
    Plot one variable pre-fixed feature against improved-QM/MM barrier.
    """
    x = pd.to_numeric(
        df[feature],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        df["barrier_kcal_per_mol"],
        errors="coerce",
    ).to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    ax.scatter(
        x,
        y,
        s=42,
        alpha=0.8,
    )

    # Descriptive least-squares line for visualisation only.
    if len(x) >= 2 and not np.isclose(np.std(x, ddof=1), 0.0):
        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(
            np.min(x),
            np.max(x),
            200,
        )

        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            linewidth=1.5,
        )

    ax.set_xlabel(
        f"{structural_label} minimum inter-residue distance (Å)"
    )

    ax.set_ylabel(
        "Improved QM/MM barrier (kcal mol$^{-1}$)"
    )

    ax.set_title(
        f"47 complete structures: {structural_label}"
    )

    annotation = (
        f"Pearson r = {pearson_r:+.3f}"
        f" (p = {pearson_p:.3g})\n"
        f"Spearman ρ = {spearman_rho:+.3f}"
        f" (p = {spearman_p:.3g})"
    )

    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()

    safe_name = feature.replace("-", "_")

    png_path = (
        FIGURE_DIR
        / f"03_{safe_name}_vs_improved_qmmm_barrier.png"
    )

    svg_path = (
        FIGURE_DIR
        / f"03_{safe_name}_vs_improved_qmmm_barrier.svg"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, svg_path


# =============================================================================
# Main
# =============================================================================

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
    )

    required_columns = {
        "ligand_id",
        "barrier_kcal_per_mol",
        *PREFIXED_FEATURES,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if len(df) != 47:
        raise RuntimeError(
            f"Expected 47 rows, found {len(df)}"
        )

    if df["ligand_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate ligand IDs found in Script 02 output."
        )

    if df["barrier_kcal_per_mol"].isna().any():
        raise RuntimeError(
            "Missing improved-QM/MM barriers found."
        )

    print("=" * 80)
    print("QMMM47 PRE-FIXED FEATURE EXPLORATORY ANALYSIS")
    print("=" * 80)

    print(f"Input: {INPUT_FILE}")
    print(f"Samples: {len(df)}")
    print()

    barrier = pd.to_numeric(
        df["barrier_kcal_per_mol"],
        errors="raise",
    ).to_numpy(dtype=float)

    records = []
    figure_records = []

    for feature in PREFIXED_FEATURES:
        values = pd.to_numeric(
            df[feature],
            errors="raise",
        ).to_numpy(dtype=float)

        finite_values = values[np.isfinite(values)]

        n = len(finite_values)
        n_unique = len(np.unique(finite_values))

        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        feature_range = maximum - minimum

        mean = float(np.mean(finite_values))
        median = float(np.median(finite_values))

        if n >= 2:
            std = float(np.std(finite_values, ddof=1))
        else:
            std = np.nan

        iqr = interquartile_range(finite_values)
        cv = coefficient_of_variation(finite_values)

        status = classify_variation(finite_values)

        pearson_r = np.nan
        pearson_p = np.nan
        spearman_rho = np.nan
        spearman_p = np.nan
        correlation_n = 0

        if status == "testable_variable":
            pearson_r, pearson_p, pearson_n = safe_pearson(
                values,
                barrier,
            )

            spearman_rho, spearman_p, spearman_n = safe_spearman(
                values,
                barrier,
            )

            if pearson_n != spearman_n:
                raise RuntimeError(
                    f"Correlation sample-size mismatch for {feature}"
                )

            correlation_n = pearson_n

            png_path, svg_path = plot_feature_vs_barrier(
                df=df,
                feature=feature,
                structural_label=STRUCTURAL_LABELS[feature],
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_rho=spearman_rho,
                spearman_p=spearman_p,
            )

            figure_records.append(
                {
                    "feature": feature,
                    "png": str(png_path),
                    "svg": str(svg_path),
                }
            )

        record = {
            "feature": feature,
            "structural_label": STRUCTURAL_LABELS[feature],

            "batch0100_rr_selection_frequency":
                BATCH0100_STABILITY[feature][
                    "rr_selection_frequency"
                ],

            "batch0100_combined_selection_frequency":
                BATCH0100_STABILITY[feature][
                    "combined_selection_frequency"
                ],

            "n_structures": n,
            "n_unique": n_unique,

            "mean_A": mean,
            "median_A": median,
            "std_A": std,
            "min_A": minimum,
            "max_A": maximum,
            "range_A": feature_range,
            "iqr_A": iqr,
            "coefficient_of_variation": cv,

            "analysis_status": status,

            "correlation_n": correlation_n,

            "pearson_r_vs_improved_qmmm_barrier":
                pearson_r,

            "pearson_p_raw":
                pearson_p,

            "spearman_rho_vs_improved_qmmm_barrier":
                spearman_rho,

            "spearman_p_raw":
                spearman_p,
        }

        records.append(record)

    stats_df = pd.DataFrame(records)

    stats_df.to_csv(
        STATS_OUT,
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Thesis-facing concise summary
    # -------------------------------------------------------------------------

    thesis_df = stats_df[
        [
            "feature",
            "structural_label",
            "batch0100_rr_selection_frequency",
            "batch0100_combined_selection_frequency",
            "n_unique",
            "range_A",
            "std_A",
            "analysis_status",
            "pearson_r_vs_improved_qmmm_barrier",
            "spearman_rho_vs_improved_qmmm_barrier",
        ]
    ].copy()

    thesis_df.to_csv(
        THESIS_OUT,
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print("Results:")
    print()

    for _, row in stats_df.iterrows():
        print("-" * 80)
        print(
            f"{row['feature']}  "
            f"({row['structural_label']})"
        )

        print(
            f"variation: "
            f"n_unique={int(row['n_unique'])}, "
            f"range={row['range_A']:.6f} Å, "
            f"std={row['std_A']:.6f} Å"
        )

        print(
            f"status: {row['analysis_status']}"
        )

        if row["analysis_status"] == "testable_variable":
            print(
                f"Pearson r = "
                f"{row['pearson_r_vs_improved_qmmm_barrier']:+.6f}"
                f", p = {row['pearson_p_raw']:.6g}"
            )

            print(
                f"Spearman rho = "
                f"{row['spearman_rho_vs_improved_qmmm_barrier']:+.6f}"
                f", p = {row['spearman_p_raw']:.6g}"
            )

        else:
            print(
                "Correlation not calculated because "
                "the feature is invariant/near-constant."
            )

    # -------------------------------------------------------------------------
    # Log
    # -------------------------------------------------------------------------

    with open(LOG_OUT, "w", encoding="utf-8") as handle:
        handle.write(
            "QMMM47 pre-fixed feature exploratory analysis\n"
        )
        handle.write("=" * 60 + "\n\n")

        handle.write(
            "The three structural features were fixed in advance from "
            "the independent February batch0100 nested-CV stability "
            "analysis. No feature screening of the 47 improved-QM/MM "
            "structures was performed.\n\n"
        )

        handle.write(
            stats_df.to_string(index=False)
        )

        handle.write("\n\n")

        handle.write(
            "Invariant features were not subjected to Pearson or "
            "Spearman correlation because correlation is undefined "
            "for a constant predictor.\n"
        )

    print()
    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print(STATS_OUT)
    print(THESIS_OUT)

    for item in figure_records:
        print(item["png"])
        print(item["svg"])

    print(LOG_OUT)

    print()
    print("SCRIPT 03 COMPLETE.")


if __name__ == "__main__":
    main()
