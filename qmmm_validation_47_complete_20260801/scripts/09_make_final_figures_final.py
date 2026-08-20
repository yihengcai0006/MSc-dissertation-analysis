#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Project paths
# ============================================================
# Keep exactly the same project structure as the original
# 09_make_final_figures.py script.
#
# This script only reads existing final result files and
# regenerates figures. It does NOT rerun the RF model,
# feature generation, bootstrap analysis, or feature-importance
# calculations.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    ROOT / "predictions"
    / "old_rf_vs_qmmm_47_complete.csv"
)

FEATURE_FILE = (
    ROOT / "results"
    / "tables"
    / "feature_importance_variance_all.tsv"
)

TYPE_FILE = (
    ROOT / "results"
    / "tables"
    / "rf_importance_by_feature_type.tsv"
)

FIGURE_DIR = ROOT / "results" / "figures"


# ============================================================
# Final thesis figure font sizes
# ============================================================
# Supervisor guidance: use approximately font size 18.
#
# 18 pt is used as the starting point for titles and axis labels.
# Tick labels, legends, and annotations are slightly smaller
# to avoid overcrowding.
#
# Final judgement should be based on readability after insertion
# into the dissertation PDF.
# ============================================================

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 16
ANNOTATION_SIZE = 16

# The top-20 plots contain long feature labels, so their y-axis
# labels are kept slightly smaller while remaining clearly readable.
FEATURE_TICK_SIZE = 13


# ============================================================
# Save figures
# ============================================================
# IMPORTANT:
# "_final" is automatically added to every output filename.
#
# Therefore the original PNG/PDF files are NOT overwritten.
# ============================================================

def save_figure(fig, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    png_path = FIGURE_DIR / f"{filename}_final.png"
    pdf_path = FIGURE_DIR / f"{filename}_final.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("Written:", png_path)
    print("Written:", pdf_path)

    plt.close(fig)


def main() -> None:

    # ========================================================
    # Read EXISTING final result files
    # ========================================================

    prediction_df = pd.read_csv(PREDICTION_FILE)
    feature_df = pd.read_csv(FEATURE_FILE, sep="\t")
    type_df = pd.read_csv(TYPE_FILE, sep="\t")

    reference = prediction_df[
        "barrier_kcal_per_mol"
    ].astype(float)

    prediction = prediction_df[
        "old_rf_prediction_kcal_per_mol"
    ].astype(float)

    residual = prediction - reference

    # These reproduce the same descriptive values already shown
    # in the original figure. No model is fitted or rerun here.
    pearson = stats.pearsonr(
        reference,
        prediction,
    )

    spearman = stats.spearmanr(
        reference,
        prediction,
    )

    slope, intercept, _, _, _ = stats.linregress(
        reference,
        prediction,
    )


    # ========================================================
    # Figure 1
    # Existing RF prediction vs improved-QM/MM reference
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.0, 6.5)
    )

    ax.scatter(
        reference,
        prediction,
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    lower = min(
        reference.min(),
        prediction.min(),
    ) - 0.3

    upper = max(
        reference.max(),
        prediction.max(),
    ) + 0.3

    grid = np.linspace(
        lower,
        upper,
        200,
    )

    ax.plot(
        grid,
        grid,
        linestyle="--",
        linewidth=1.7,
        label="1:1 line",
    )

    regression_grid = np.linspace(
        reference.min(),
        reference.max(),
        200,
    )

    ax.plot(
        regression_grid,
        intercept + slope * regression_grid,
        linewidth=1.7,
        label="Linear fit",
    )

    annotation = (
        f"n = {len(prediction_df)}\n"
        f"Pearson r = {pearson.statistic:.3f}, "
        f"p = {pearson.pvalue:.3f}\n"
        f"Spearman ρ = {spearman.statistic:.3f}, "
        f"p = {spearman.pvalue:.3f}\n"
        f"Slope = {slope:.3f}"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_SIZE,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.set_xlabel(
        "Improved-QM/MM activation barrier (kcal/mol)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Existing RF prediction (kcal/mol)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_title(
        "Existing RF predictions versus improved-QM/MM barriers",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.set_xlim(
        lower,
        upper,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "old_rf_vs_qmmm_47_complete_scatter",
    )


    # ========================================================
    # Figure 2
    # Residuals
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9.0, 6.2)
    )

    ax.scatter(
        reference,
        residual,
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.7,
    )

    residual_slope, residual_intercept, _, _, _ = (
        stats.linregress(
            reference,
            residual,
        )
    )

    ax.plot(
        regression_grid,
        residual_intercept
        + residual_slope * regression_grid,
        linewidth=1.7,
        label=(
            "Residual trend, "
            f"slope = {residual_slope:.3f}"
        ),
    )

    ax.set_xlabel(
        "Improved-QM/MM activation barrier (kcal/mol)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Residual: RF prediction − QM/MM reference\n(kcal/mol)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_title(
        "Residuals of existing RF predictions",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "old_rf_vs_qmmm_47_complete_residuals",
    )


    # ========================================================
    # Figure 3
    # Distribution comparison
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.0, 6.2)
    )

    bins = np.linspace(
        min(
            reference.min(),
            prediction.min(),
        ) - 0.2,
        max(
            reference.max(),
            prediction.max(),
        ) + 0.2,
        14,
    )

    ax.hist(
        reference,
        bins=bins,
        alpha=0.55,
        label=(
            "Improved-QM/MM reference "
            f"(SD = {reference.std(ddof=1):.2f})"
        ),
    )

    ax.hist(
        prediction,
        bins=bins,
        alpha=0.55,
        label=(
            "Existing RF prediction "
            f"(SD = {prediction.std(ddof=1):.2f})"
        ),
    )

    ax.axvline(
        reference.mean(),
        linestyle="--",
        linewidth=1.5,
    )

    ax.axvline(
        prediction.mean(),
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Barrier / prediction (kcal/mol)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Number of ligands",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_title(
        "Distributions of improved-QM/MM barriers "
        "and existing RF predictions",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.legend(
        fontsize=LEGEND_SIZE,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "reference_vs_prediction_distribution",
    )


    # ========================================================
    # Figure 4
    # Top-20 existing RF feature importance
    # ========================================================

    top20 = (
        feature_df
        .sort_values(
            "rf_importance",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "rf_importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10.0, 9.0)
    )

    ax.barh(
        top20["feature_name"],
        top20["rf_importance"] * 100,
    )

    ax.set_xlabel(
        "RF feature importance (%)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Feature",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_title(
        "Top 20 existing RF features",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.tick_params(
        axis="x",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.tick_params(
        axis="y",
        labelsize=FEATURE_TICK_SIZE,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "top_20_rf_feature_importance",
    )


    # ========================================================
    # Figure 5
    # Variability of top-20 important RF features
    # ========================================================

    variability_top20 = (
        feature_df
        .sort_values(
            "rf_importance",
            ascending=False,
        )
        .head(20)
        .sort_values(
            "sd_distance_angstrom",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10.0, 9.0)
    )

    ax.barh(
        variability_top20["feature_name"],
        variability_top20[
            "sd_distance_angstrom"
        ],
    )

    for index, value in enumerate(
        variability_top20[
            "sd_distance_angstrom"
        ]
    ):
        ax.text(
            max(value, 0.002),
            index,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=ANNOTATION_SIZE,
        )

    ax.set_xlabel(
        "Standard deviation across 47 structures (Å)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Feature",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_title(
        "Variability of the 20 most important existing RF features",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.tick_params(
        axis="x",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.tick_params(
        axis="y",
        labelsize=FEATURE_TICK_SIZE,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "top_20_rf_feature_variability",
    )


    # ========================================================
    # Figure 6
    # RF importance by feature type
    # ========================================================

    type_plot = type_df.copy()

    # --------------------------------------------------------
    # Display labels only.
    #
    # This does NOT change the underlying data or analysis.
    # It only replaces internal/high-level plotting terminology
    # with thesis-consistent terminology.
    # --------------------------------------------------------

    display_name_map = {
        "protein-protein": "protein–protein",
        "protein-LIG": "protein–ligand",
        "protein-GTP": "protein–GTP",
        "GTP-LIG": "GTP–ligand",
        "other": "other",
    }

    type_plot["display_feature_type"] = (
        type_plot["feature_type"]
        .map(display_name_map)
        .fillna(type_plot["feature_type"])
    )

    type_plot = type_plot.sort_values(
        "importance_percentage",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.8)
    )

    ax.barh(
        type_plot["display_feature_type"],
        type_plot["importance_percentage"],
    )

    for index, value in enumerate(
        type_plot["importance_percentage"]
    ):
        ax.text(
            value + 0.3,
            index,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=ANNOTATION_SIZE,
        )

    ax.set_xlabel(
        "Total RF importance (%)",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Feature type",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_title(
        "Distribution of existing RF importance by feature type",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    ax.set_xlim(
        0,
        max(
            type_plot["importance_percentage"]
        ) * 1.10,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "rf_importance_by_feature_type",
    )


if __name__ == "__main__":
    main()