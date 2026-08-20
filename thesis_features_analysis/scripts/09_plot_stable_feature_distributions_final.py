#!/usr/bin/env python3

"""
Regenerate the dissertation-final stable-feature distribution figure.

IMPORTANT
---------
This is a figure-only final-pass script.

It does NOT:
- rerun the stable-feature analysis;
- select features again;
- rerun nested cross-validation;
- modify the completed result tables;
- overwrite the original figure files.

It reads the already saved Script 09 tables and redraws the figure with:
- larger, thesis-readable text;
- a vertical 3-panel layout;
- dissertation-consistent terminology.

Scientific scope
----------------
The low/high groups come from the old RF-predicted barriers in the
100-ligand dataset. Therefore, this figure shows structural differences
associated with the old RF model behaviour. It does not present improved-
QM/MM group labels and does not constitute a new feature-selection step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 20260819
PNG_DPI = 300

TITLE_SIZE = 18
PANEL_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14
ANNOTATION_SIZE = 13

GROUP_ORDER = ["low", "high"]

STABLE_FEATURES = [
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
]

DISPLAY_LABELS = {
    "PRO62-PRO1279": "GLU62(RAS)--SER1279(NF1)",
    "PRO1275-PRO1276": "PHE1275(NF1)--ARG1276(NF1)",
    "PRO90-PRO92": "PHE90(RAS)--ASP92(RAS)",
}

LOW_LABEL = "Low"
HIGH_LABEL = "High"


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "figures"
)

VALUES_FILE = TABLE_DIR / "09_stable_feature_values.tsv"
SUMMARY_FILE = TABLE_DIR / "09_stable_feature_group_statistics.tsv"

OUTPUT_STEM = "09_stable_feature_low_high_distributions_final"


# ============================================================
# Console helper
# ============================================================

def log(message: str = "") -> None:
    print(message, flush=True)


# ============================================================
# Input validation
# ============================================================

def validate_inputs() -> None:
    missing = []

    if not VALUES_FILE.exists():
        missing.append(str(VALUES_FILE))

    if not SUMMARY_FILE.exists():
        missing.append(str(SUMMARY_FILE))

    if missing:
        raise FileNotFoundError(
            "Required completed Script 09 table(s) not found:\n"
            + "\n".join(missing)
        )


def load_completed_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    values_df = pd.read_csv(VALUES_FILE, sep="\t")
    summary_df = pd.read_csv(SUMMARY_FILE, sep="\t")

    required_value_columns = {
        "feature",
        "group",
        "distance_A",
    }

    required_summary_columns = {
        "feature",
        "high_minus_low_mean_A",
        "cohens_d_low_minus_high",
    }

    missing_value_cols = required_value_columns - set(values_df.columns)
    missing_summary_cols = required_summary_columns - set(summary_df.columns)

    if missing_value_cols:
        raise ValueError(
            "Missing columns in values table: "
            f"{sorted(missing_value_cols)}"
        )

    if missing_summary_cols:
        raise ValueError(
            "Missing columns in summary table: "
            f"{sorted(missing_summary_cols)}"
        )

    values_df["feature"] = values_df["feature"].astype(str)
    values_df["group"] = (
        values_df["group"].astype(str).str.lower().str.strip()
    )

    values_df["distance_A"] = pd.to_numeric(
        values_df["distance_A"],
        errors="coerce",
    )

    summary_df["feature"] = summary_df["feature"].astype(str)

    if values_df["distance_A"].isna().any():
        raise ValueError(
            "Found non-numeric or missing values in distance_A."
        )

    observed_features = set(values_df["feature"].unique())
    expected_features = set(STABLE_FEATURES)

    if expected_features - observed_features:
        raise ValueError(
            "Values table is missing expected stable features: "
            f"{sorted(expected_features - observed_features)}"
        )

    observed_summary_features = set(summary_df["feature"].unique())

    if expected_features - observed_summary_features:
        raise ValueError(
            "Summary table is missing expected stable features: "
            f"{sorted(expected_features - observed_summary_features)}"
        )

    allowed_groups = set(GROUP_ORDER)
    observed_groups = set(values_df["group"].unique())

    if not observed_groups.issubset(allowed_groups):
        raise ValueError(
            "Unexpected groups found in values table: "
            f"{sorted(observed_groups - allowed_groups)}"
        )

    return values_df, summary_df


# ============================================================
# Plot helper
# ============================================================

def save_figure(fig: plt.Figure) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    png_path = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    svg_path = FIGURE_DIR / f"{OUTPUT_STEM}.svg"

    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, svg_path


def make_final_plot(
    values_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8.8, 13.8),
        sharex=False,
        sharey=False,
    )

    # A simple overall title without internal dataset naming.
    fig.suptitle(
        "Low/high distributions of the three stable protein--protein features",
        fontsize=TITLE_SIZE,
        y=0.995,
    )

    for ax, feature in zip(axes, STABLE_FEATURES):
        feature_values = values_df.loc[
            values_df["feature"] == feature
        ].copy()

        low = feature_values.loc[
            feature_values["group"] == "low",
            "distance_A",
        ].to_numpy(dtype=float)

        high = feature_values.loc[
            feature_values["group"] == "high",
            "distance_A",
        ].to_numpy(dtype=float)

        datasets = [low, high]

        # Boxplots for group summaries.
        bp = ax.boxplot(
            datasets,
            positions=[1, 2],
            widths=0.46,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(linewidth=1.6),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
        )

        # Light fill to distinguish the two groups.
        # This is presentation-only; no data are changed.
        facecolors = ["#cfe3f6", "#f9d9b4"]
        edgecolors = ["#4c78a8", "#f58518"]

        for patch, facecolor, edgecolor in zip(
            bp["boxes"], facecolors, edgecolors
        ):
            patch.set_facecolor(facecolor)
            patch.set_edgecolor(edgecolor)

        for median, edgecolor in zip(bp["medians"], edgecolors):
            median.set_color(edgecolor)

        # Overlay all individual observations with jitter.
        for x_position, values, color in zip(
            [1, 2],
            datasets,
            edgecolors,
        ):
            jitter = rng.normal(
                loc=0.0,
                scale=0.04,
                size=len(values),
            )

            ax.scatter(
                np.full(len(values), x_position) + jitter,
                values,
                s=24,
                alpha=0.75,
                color=color,
                edgecolors="none",
                zorder=3,
            )

        # Panel title: biological identity first, internal label second.
        ax.set_title(
            f"{DISPLAY_LABELS[feature]}\nInternal feature: {feature}",
            fontsize=PANEL_TITLE_SIZE,
            pad=8,
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [LOW_LABEL, HIGH_LABEL],
            fontsize=TICK_LABEL_SIZE,
        )

        ax.set_ylabel(
            "Minimum inter-residue distance (Å)",
            fontsize=AXIS_LABEL_SIZE,
        )

        ax.tick_params(
            axis="y",
            labelsize=TICK_LABEL_SIZE,
        )

        ax.grid(
            axis="y",
            alpha=0.22,
        )

        # Pull stored statistics from the completed summary table.
        stats_row = summary_df.loc[
            summary_df["feature"] == feature
        ].iloc[0]

        delta = float(stats_row["high_minus_low_mean_A"])
        d_value = float(stats_row["cohens_d_low_minus_high"])

        annotation = (
            f"High − low mean = {delta:+.3f} Å\n"
            f"Cohen's d = {d_value:+.2f}"
        )

        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_SIZE,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "0.75",
            },
        )

        # Add a little breathing space vertically.
        data_min = min(np.min(low), np.min(high))
        data_max = max(np.max(low), np.max(high))
        data_range = data_max - data_min

        if data_range == 0:
            pad = 0.05
        else:
            pad = max(0.06 * data_range, 0.02)

        ax.set_ylim(
            data_min - pad,
            data_max + pad,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    return save_figure(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    log("=" * 72)
    log("Final dissertation stable-feature distribution figure regeneration")
    log("=" * 72)

    log()
    log("This script will NOT:")
    log("  - rerun the stable-feature analysis")
    log("  - rerun nested CV")
    log("  - select features again")
    log("  - modify completed TSV files")
    log("  - overwrite original figure files")

    log()
    log("Checking completed Script 09 tables...")
    validate_inputs()
    log("Completed tables found.")

    log()
    log("Reading completed Script 09 tables...")
    values_df, summary_df = load_completed_tables()

    log(f"Values rows loaded: {len(values_df)}")
    log(f"Summary rows loaded: {len(summary_df)}")

    log()
    log("Regenerating final figure...")
    png_path, svg_path = make_final_plot(values_df, summary_df)

    log(f"Written: {png_path}")
    log(f"Written: {svg_path}")

    log()
    log("=" * 72)
    log("Final figure regeneration completed")
    log("=" * 72)

    log()
    log("Scientific analysis unchanged:")
    log("  stable-feature analysis rerun: False")
    log("  nested CV rerun: False")
    log("  features re-selected: False")
    log("  completed TSV results modified: False")
    log("  original figures overwritten: False")

    log()
    log("Done.")


if __name__ == "__main__":
    main()