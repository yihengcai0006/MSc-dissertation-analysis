#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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


def save_figure(fig, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    png_path = FIGURE_DIR / f"{filename}.png"
    pdf_path = FIGURE_DIR / f"{filename}.pdf"

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

    pearson = stats.pearsonr(reference, prediction)
    spearman = stats.spearmanr(reference, prediction)

    slope, intercept, _, _, _ = stats.linregress(
        reference,
        prediction,
    )

    # ---------------------------------------------------------
    # Figure 1: RF prediction vs QM/MM reference
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    ax.scatter(
        reference,
        prediction,
        s=50,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    lower = min(reference.min(), prediction.min()) - 0.3
    upper = max(reference.max(), prediction.max()) + 0.3

    grid = np.linspace(lower, upper, 200)

    ax.plot(
        grid,
        grid,
        linestyle="--",
        linewidth=1.5,
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
        linewidth=1.5,
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
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.set_xlabel("Improved QM/MM barrier (kcal/mol)")
    ax.set_ylabel("Old RF prediction (kcal/mol)")
    ax.set_title(
        "Old RF predictions versus improved QM/MM barriers"
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(
        fig,
        "old_rf_vs_qmmm_47_complete_scatter",
    )

    # ---------------------------------------------------------
    # Figure 2: residuals
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    ax.scatter(
        reference,
        residual,
        s=50,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    residual_slope, residual_intercept, _, _, _ = (
        stats.linregress(reference, residual)
    )

    ax.plot(
        regression_grid,
        residual_intercept
        + residual_slope * regression_grid,
        linewidth=1.5,
        label=f"Residual trend, slope = {residual_slope:.3f}",
    )

    ax.set_xlabel("Improved QM/MM barrier (kcal/mol)")
    ax.set_ylabel(
        "Residual: RF prediction − QM/MM reference (kcal/mol)"
    )
    ax.set_title(
        "Residuals of old RF predictions"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(
        fig,
        "old_rf_vs_qmmm_47_complete_residuals",
    )

    # ---------------------------------------------------------
    # Figure 3: distributions
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    bins = np.linspace(
        min(reference.min(), prediction.min()) - 0.2,
        max(reference.max(), prediction.max()) + 0.2,
        14,
    )

    ax.hist(
        reference,
        bins=bins,
        alpha=0.55,
        label=(
            f"QM/MM reference "
            f"(SD={reference.std(ddof=1):.2f})"
        ),
    )

    ax.hist(
        prediction,
        bins=bins,
        alpha=0.55,
        label=(
            f"Old RF prediction "
            f"(SD={prediction.std(ddof=1):.2f})"
        ),
    )

    ax.axvline(
        reference.mean(),
        linestyle="--",
        linewidth=1.3,
    )

    ax.axvline(
        prediction.mean(),
        linestyle=":",
        linewidth=1.3,
    )

    ax.set_xlabel("Barrier / prediction (kcal/mol)")
    ax.set_ylabel("Number of ligands")
    ax.set_title(
        "Distribution of QM/MM barriers and old RF predictions"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        "reference_vs_prediction_distribution",
    )

    # ---------------------------------------------------------
    # Figure 4: top-20 RF feature importance
    # ---------------------------------------------------------
    top20 = (
        feature_df
        .sort_values("rf_importance", ascending=False)
        .head(20)
        .sort_values("rf_importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.5))

    ax.barh(
        top20["feature_name"],
        top20["rf_importance"] * 100,
    )

    ax.set_xlabel("RF feature importance (%)")
    ax.set_ylabel("Feature")
    ax.set_title("Top 20 old RF features")
    ax.grid(axis="x", alpha=0.25)

    save_figure(
        fig,
        "top_20_rf_feature_importance",
    )

    # ---------------------------------------------------------
    # Figure 5: variability of top-20 important features
    # ---------------------------------------------------------
    variability_top20 = (
        feature_df
        .sort_values("rf_importance", ascending=False)
        .head(20)
        .sort_values(
            "sd_distance_angstrom",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.5))

    ax.barh(
        variability_top20["feature_name"],
        variability_top20["sd_distance_angstrom"],
    )
    # ===========
    for index, value in enumerate(
    variability_top20["sd_distance_angstrom"]
):
        ax.text(
            max(value, 0.002),
            index,
            f"{value:.3f}",
            va="center",
            ha="left",
        )
    #============
    ax.set_xlabel(
        "Standard deviation across 47 structures (Å)"
    )
    ax.set_ylabel("Feature")
    ax.set_title(
        "Variability of the 20 most important RF features"
    )
    ax.grid(axis="x", alpha=0.25)

    save_figure(
        fig,
        "top_20_rf_feature_variability",
    )

    # ---------------------------------------------------------
    # Figure 6: importance by feature type
    # ---------------------------------------------------------
    type_plot = type_df.sort_values(
        "importance_percentage",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.barh(
        type_plot["feature_type"],
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
        )

    ax.set_xlabel("Total RF importance (%)")
    ax.set_ylabel("Feature type")
    ax.set_title(
        "Distribution of RF importance by feature type"
    )
    ax.set_xlim(
        0,
        max(type_plot["importance_percentage"]) * 1.08,
    )
    ax.grid(axis="x", alpha=0.25)

    save_figure(
        fig,
        "rf_importance_by_feature_type",
    )


if __name__ == "__main__":
    main()
