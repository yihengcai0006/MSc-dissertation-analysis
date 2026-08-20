#!/usr/bin/env python3

"""
Regenerate the dissertation-final feature-stability figure.

IMPORTANT
---------
This is a figure-only final-pass script.

It does NOT:
- refit any classifier;
- rerun nested cross-validation;
- rerun feature selection;
- recalculate selection frequency;
- recalculate coefficient-direction consistency;
- modify any completed TSV result file;
- overwrite the original stability figure.

It reads the already saved source-data table for the residue-residue
(feature-selection stability) figure and redraws the figure with larger,
thesis-readable text.

Scientific scope
----------------
The batch0100 low/high labels are derived from the old RF-predicted barriers.

Therefore, this figure shows structural patterns associated with the
behaviour of the old RF model. It should not be interpreted as evidence
about measured activation barriers or improved-QM/MM barriers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# General configuration
# ============================================================

BATCH = "batch0100"
TOP_N = 15
PNG_DPI = 300

# ------------------------------------------------------------
# Final thesis figure formatting
# ------------------------------------------------------------

TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
ANNOTATION_SIZE = 14


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    BASE_DIR
    / "results"
    / BATCH
    / "stability"
    / "figure_source_data"
    / "06_residue_residue_stability_source.tsv"
)

FIGURE_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "stability"
    / "figures"
)

OUTPUT_STEM = "top15_residue_residue_selection_frequency_final"


# ============================================================
# Console helper
# ============================================================

def log(message: str = "") -> None:
    """Print one console message immediately."""
    print(message, flush=True)


# ============================================================
# Validation and loading
# ============================================================

def validate_input_path() -> None:
    """Confirm that the completed source-data table exists."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            "Required source-data file not found:\n"
            f"{SOURCE_FILE}"
        )


def load_plot_data() -> pd.DataFrame:
    """
    Read the completed source-data TSV.

    No feature-stability analysis is recalculated here.
    """
    plot_data = pd.read_csv(
        SOURCE_FILE,
        sep="\t",
    )

    required_columns = {
        "feature_set",
        "feature",
        "selection_frequency",
        "dominant_direction_fraction",
        "stability_rank_within_feature_set",
    }

    missing_columns = required_columns - set(plot_data.columns)

    if missing_columns:
        raise ValueError(
            "Source-data table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if plot_data.empty:
        raise ValueError(
            "Source-data table is empty."
        )

    feature_sets = set(
        plot_data["feature_set"].dropna().astype(str)
    )

    if feature_sets != {"residue_residue"}:
        raise ValueError(
            "Expected only feature_set = 'residue_residue' "
            "in the source-data table, but observed: "
            f"{sorted(feature_sets)}"
        )

    if not plot_data["selection_frequency"].between(
        0.0,
        1.0,
        inclusive="both",
    ).all():
        raise ValueError(
            "Selection frequencies must lie within [0, 1]."
        )

    if not plot_data["dominant_direction_fraction"].between(
        0.0,
        1.0,
        inclusive="both",
    ).all():
        raise ValueError(
            "Direction-consistency values must lie within [0, 1]."
        )

    # Keep the top 15 rows only, sorted exactly by stability rank.
    plot_data = (
        plot_data.sort_values(
            "stability_rank_within_feature_set",
            ascending=True,
        )
        .head(TOP_N)
        .copy()
    )

    if len(plot_data) == 0:
        raise ValueError(
            "No rows available after selecting the top features."
        )

    return plot_data


# ============================================================
# Figure saving
# ============================================================

def save_figure(figure: plt.Figure) -> tuple[Path, Path]:
    """
    Save only dissertation-final copies.

    The '_final' suffix ensures that the original figure is not overwritten.
    """
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    svg_path = FIGURE_DIR / f"{OUTPUT_STEM}.svg"

    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return png_path, svg_path


# ============================================================
# Plotting
# ============================================================

def make_final_plot(plot_data: pd.DataFrame) -> tuple[Path, Path]:
    """
    Redraw the residue-residue feature-stability figure
    with larger thesis-readable text.

    The plotted numerical values are exactly those already stored in the
    source-data TSV.
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
        8.2,
        0.52 * len(display_data) + 2.2,
    )

    figure, axis = plt.subplots(
        figsize=(11.0, figure_height)
    )

    axis.barh(
        display_data["feature"],
        display_data["selection_frequency"],
    )

    axis.set_xlim(0.0, 1.10)

    axis.set_xlabel(
        "Selection frequency across 25 outer folds",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Protein--protein distance feature",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Most stable protein--protein features",
        fontsize=TITLE_SIZE,
        pad=14,
    )

    axis.axvline(
        0.5,
        linewidth=1.2,
        linestyle="--",
    )

    axis.axvline(
        0.8,
        linewidth=1.2,
        linestyle=":",
    )

    axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    for position, (_, row) in enumerate(display_data.iterrows()):
        frequency = float(row["selection_frequency"])
        consistency = float(row["dominant_direction_fraction"])

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
            fontsize=ANNOTATION_SIZE,
        )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    return save_figure(figure)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Regenerate only the dissertation-final figure.

    No scientific analysis is rerun.
    """
    log("=" * 72)
    log("Final dissertation feature-stability figure regeneration")
    log("=" * 72)

    log()
    log("This script will NOT:")
    log("  - refit classifiers")
    log("  - rerun nested CV")
    log("  - rerun feature selection")
    log("  - recalculate stability statistics")
    log("  - overwrite original figures")
    log("  - modify completed TSV files")

    log()
    log("Checking completed source-data file...")
    validate_input_path()
    log("Completed source-data file found.")

    log()
    log("Reading completed source-data values...")
    plot_data = load_plot_data()

    log(
        f"Rows loaded for plotting: {len(plot_data)}"
    )

    log()
    log("Regenerating final figure...")
    png_path, svg_path = make_final_plot(plot_data)

    log(f"Written: {png_path}")
    log(f"Written: {svg_path}")

    log()
    log("=" * 72)
    log("Final figure regeneration completed")
    log("=" * 72)

    log()
    log("Scientific analysis unchanged:")
    log("  models refitted: False")
    log("  nested CV rerun: False")
    log("  feature selection rerun: False")
    log("  stability recalculated: False")
    log("  completed TSV results modified: False")
    log("  original figures overwritten: False")

    log()
    log("Done.")


if __name__ == "__main__":
    main()