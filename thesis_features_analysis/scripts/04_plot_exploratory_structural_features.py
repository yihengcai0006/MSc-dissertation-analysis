"""
Create a small set of exploratory structural figures for batch0100.

This script combines:

1. the exploratory statistics produced by script 02; and
2. the model-feature manifest produced by script 03.

Only features included in the original RF distance-feature list are
considered in the figures.

The script creates four scientific figures:

1. top residue-ligand effect sizes;
2. top residue-cofactor effect sizes;
3. top residue-residue effect sizes;
4. low/high distributions of one representative feature from each type.

Important
---------
The low/high labels originate from old RF-predicted barriers.

These figures describe structural associations with the behaviour of
the old model. They do not show measured or improved QM/MM barriers and
must not be interpreted as causal structural determinants.

The figures are exploratory and are not used to preselect features for
the formal nested-cross-validation classifier.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BATCH = "batch0100"

TOP_N_BY_TYPE = {
    "residue_ligand": 10,
    "residue_cofactor": 10,
    "residue_residue": 15,
}

RANDOM_SEED = 42


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TABLE_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "tables"
)

FIGURE_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "figures"
    / "exploratory_structural_features"
)

SOURCE_DATA_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "figure_source_data"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_FILE = (
    TABLE_DIR
    / "features_with_barriers.parquet"
)

ALL_FEATURE_RANKING_FILE = (
    TABLE_DIR
    / "low_high_feature_ranking_all.tsv"
)

MODEL_FEATURE_MANIFEST_FILE = (
    TABLE_DIR
    / "03_model_feature_manifest.tsv"
)

OUT_MODEL_USED_RANKING = (
    TABLE_DIR
    / "04_model_used_exploratory_ranking.tsv"
)

OUT_SELECTED_FEATURES = (
    TABLE_DIR
    / "04_representative_features.tsv"
)

OUT_FIGURE_MANIFEST = (
    TABLE_DIR
    / "04_figure_manifest.tsv"
)

OUT_RUN_SUMMARY = (
    TABLE_DIR
    / "04_plot_exploratory_structural_features_summary.tsv"
)


# ============================================================
# Input validation
# ============================================================

def validate_input_paths() -> None:
    """Confirm that all required upstream outputs exist."""
    required_files = [
        MASTER_FILE,
        ALL_FEATURE_RANKING_FILE,
        MODEL_FEATURE_MANIFEST_FILE,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required input files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing_files
            )
        )


def validate_input_tables(
    master: pd.DataFrame,
    ranking: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """Validate required columns and identifiers."""
    master_required = {
        "ligand_id",
        "barrier",
        "group",
    }

    ranking_required = {
        "feature",
        "feature_type",
        "quality_pass",
        "cohens_d",
        "abs_cohens_d",
        "difference_high_minus_low",
        "low_mean",
        "high_mean",
        "welch_p_value",
        "welch_fdr_bh",
    }

    manifest_required = {
        "feature",
        "feature_type",
        "available_in_batch0100",
        "included_in_formal_sets",
    }

    checks = [
        (
            "master table",
            master_required - set(master.columns),
        ),
        (
            "script-02 ranking",
            ranking_required - set(ranking.columns),
        ),
        (
            "script-03 manifest",
            manifest_required - set(manifest.columns),
        ),
    ]

    for table_name, missing_columns in checks:
        if missing_columns:
            raise ValueError(
                f"{table_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

    if master["ligand_id"].duplicated().any():
        raise ValueError(
            "Duplicated ligand IDs were found "
            "in the master table."
        )

    if ranking["feature"].duplicated().any():
        raise ValueError(
            "Duplicated feature names were found "
            "in the script-02 ranking."
        )

    if manifest["feature"].duplicated().any():
        raise ValueError(
            "Duplicated feature names were found "
            "in the script-03 manifest."
        )


# ============================================================
# Data preparation
# ============================================================

def build_model_used_ranking(
    ranking: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict the script-02 statistics to formal model-listed features.

    No statistics are recalculated here.
    """
    formal_manifest = manifest.loc[
        manifest[
            "included_in_formal_sets"
        ],
        [
            "feature",
            "feature_type",
        ],
    ].copy()

    formal_manifest = (
        formal_manifest.rename(
            columns={
                "feature_type": (
                    "manifest_feature_type"
                )
            }
        )
    )

    merged = formal_manifest.merge(
        ranking,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    missing_statistics = merged.loc[
        merged["cohens_d"].isna()
        & merged["quality_pass"].isna(),
        "feature",
    ].tolist()

    if missing_statistics:
        raise ValueError(
            "Model features were missing from "
            "the script-02 ranking:\n"
            + "\n".join(missing_statistics)
        )

    type_mismatch = merged[
        merged["manifest_feature_type"]
        != merged["feature_type"]
    ]

    if not type_mismatch.empty:
        raise ValueError(
            "Feature-type disagreement between "
            "scripts 02 and 03:\n"
            + type_mismatch[
                [
                    "feature",
                    "manifest_feature_type",
                    "feature_type",
                ]
            ].to_string(index=False)
        )

    merged = merged.sort_values(
        by=[
            "feature_type",
            "quality_pass",
            "abs_cohens_d",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        na_position="last",
    )

    return merged


def select_top_features(
    model_ranking: pd.DataFrame,
    feature_type: str,
    top_n: int,
) -> pd.DataFrame:
    """Select the top quality-passing features by absolute Cohen's d."""
    subset = model_ranking[
        (
            model_ranking["feature_type"]
            == feature_type
        )
        & (
            model_ranking["quality_pass"]
        )
    ].copy()

    subset = subset.dropna(
        subset=[
            "cohens_d",
            "abs_cohens_d",
        ]
    )

    subset = (
        subset
        .sort_values(
            "abs_cohens_d",
            ascending=False,
        )
        .head(top_n)
    )

    if subset.empty:
        raise ValueError(
            "No quality-passing features found for: "
            f"{feature_type}"
        )

    return subset


# ============================================================
# Figure helpers
# ============================================================

def save_figure(
    figure: plt.Figure,
    output_stem: str,
) -> tuple[Path, Path]:
    """Save one figure as publication-quality PNG and SVG."""
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

    return png_file, svg_file


def plot_effect_size_ranking(
    plot_data: pd.DataFrame,
    title: str,
    output_stem: str,
) -> tuple[Path, Path]:
    """
    Plot signed Cohen's d values for one structural feature type.

    Cohen's d is defined as mean(high) - mean(low).
    """
    display_data = plot_data.sort_values(
        "cohens_d",
        ascending=True,
    ).copy()

    figure_height = max(
        5.0,
        0.38 * len(display_data) + 1.7,
    )

    figure, axis = plt.subplots(
        figsize=(9, figure_height)
    )

    axis.barh(
        display_data["feature"],
        display_data["cohens_d"],
    )

    axis.axvline(
        0,
        linewidth=1,
    )

    axis.set_xlabel(
        "Cohen's d: mean(high) − mean(low)\n"
        "Positive = larger distance in high; "
        "negative = smaller distance in high"
    )

    axis.set_ylabel(
        "Structural distance feature"
    )

    axis.set_title(
        title
    )

    maximum_effect = max(
        display_data[
            "cohens_d"
        ].abs().max(),
        0.1,
    )

    axis.set_xlim(
        -1.30 * maximum_effect,
        1.30 * maximum_effect,
    )

    for position, effect_size in enumerate(
        display_data["cohens_d"]
    ):
        if effect_size >= 0:
            horizontal_alignment = "left"
            label_position = (
                effect_size
                + 0.025 * maximum_effect
            )
        else:
            horizontal_alignment = "right"
            label_position = (
                effect_size
                - 0.025 * maximum_effect
            )

        axis.text(
            label_position,
            position,
            f"{effect_size:.2f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
        )

    figure.tight_layout()

    output_files = save_figure(
        figure,
        output_stem,
    )

    plt.close(
        figure
    )

    return output_files


def make_jitter_positions(
    centre: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate reproducible horizontal jitter positions."""
    return rng.normal(
        loc=centre,
        scale=0.045,
        size=count,
    )


def plot_representative_distributions(
    master: pd.DataFrame,
    representative_features: pd.DataFrame,
) -> tuple[Path, Path, pd.DataFrame]:
    """
    Plot one representative low/high distribution for each feature type.

    The representative feature is the quality-passing model-listed
    feature with the largest absolute Cohen's d within that type.
    """
    low_high = master[
        master["group"].isin(
            ["low", "high"]
        )
    ].copy()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    n_panels = len(
        representative_features
    )

    figure, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            5.1 * n_panels,
            5.2,
        ),
        squeeze=False,
    )

    source_rows = []

    for panel_index, (_, row) in enumerate(
        representative_features.iterrows()
    ):
        axis = axes[
            0,
            panel_index,
        ]

        feature = row[
            "feature"
        ]

        feature_type = row[
            "feature_type"
        ]

        if feature not in low_high.columns:
            raise ValueError(
                "Representative feature missing "
                f"from master table: {feature}"
            )

        low_values = pd.to_numeric(
            low_high.loc[
                low_high["group"] == "low",
                feature,
            ],
            errors="coerce",
        ).dropna()

        high_values = pd.to_numeric(
            low_high.loc[
                low_high["group"] == "high",
                feature,
            ],
            errors="coerce",
        ).dropna()

        axis.boxplot(
            [
                low_values,
                high_values,
            ],
            tick_labels=[
                f"Low\nn={len(low_values)}",
                f"High\nn={len(high_values)}",
            ],
            showmeans=True,
            widths=0.5,
        )

        low_x = make_jitter_positions(
            centre=1.0,
            count=len(low_values),
            rng=rng,
        )

        high_x = make_jitter_positions(
            centre=2.0,
            count=len(high_values),
            rng=rng,
        )

        axis.scatter(
            low_x,
            low_values,
            alpha=0.70,
            s=26,
        )

        axis.scatter(
            high_x,
            high_values,
            alpha=0.70,
            s=26,
        )

        axis.set_title(
            f"{feature}\n"
            f"{feature_type.replace('_', '–')}\n"
            f"d = {row['cohens_d']:.2f}; "
            f"high − low = "
            f"{row['difference_high_minus_low']:.2f} Å"
        )

        axis.set_xlabel(
            "Old-model barrier group"
        )

        axis.set_ylabel(
            "Distance (Å)"
        )

        for group_name, values in [
            ("low", low_values),
            ("high", high_values),
        ]:
            for value in values:
                source_rows.append(
                    {
                        "feature": feature,
                        "feature_type": (
                            feature_type
                        ),
                        "group": group_name,
                        "distance": value,
                    }
                )

    figure.suptitle(
        "Representative model-listed structural distances\n"
        "Low versus high old RF-predicted barrier groups",
        y=1.03,
    )

    figure.tight_layout()

    output_files = save_figure(
        figure,
        "representative_model_feature_distributions",
    )

    plt.close(
        figure
    )

    source_data = pd.DataFrame(
        source_rows
    )

    return (
        output_files[0],
        output_files[1],
        source_data,
    )


# ============================================================
# Summaries
# ============================================================

def save_run_summary(
    model_ranking: pd.DataFrame,
    representative_features: pd.DataFrame,
) -> None:
    """Save key counts and interpretation information."""
    settings = {
        "batch": BATCH,
        "statistics_source": str(
            ALL_FEATURE_RANKING_FILE
        ),
        "model_manifest_source": str(
            MODEL_FEATURE_MANIFEST_FILE
        ),
        "master_data_source": str(
            MASTER_FILE
        ),
        "n_model_used_features": (
            len(model_ranking)
        ),
        "n_residue_ligand": int(
            (
                model_ranking[
                    "feature_type"
                ]
                == "residue_ligand"
            ).sum()
        ),
        "n_residue_cofactor": int(
            (
                model_ranking[
                    "feature_type"
                ]
                == "residue_cofactor"
            ).sum()
        ),
        "n_residue_residue": int(
            (
                model_ranking[
                    "feature_type"
                ]
                == "residue_residue"
            ).sum()
        ),
        "n_scientific_figures": 4,
        "n_ranking_figures": 3,
        "n_distribution_figures": 1,
        "representative_selection_rule": (
            "largest absolute Cohen's d "
            "among quality-passing model-listed "
            "features within each feature type"
        ),
        "statistics_recalculated": False,
        "used_for_classifier_preselection": False,
        "barrier_source": (
            "old RF-predicted barriers"
        ),
        "interpretation": (
            "exploratory structural associations "
            "with old-model behaviour"
        ),
        "representative_features": (
            "; ".join(
                representative_features[
                    "feature"
                ].astype(str)
            )
        ),
    }

    pd.DataFrame(
        [
            {
                "setting": key,
                "value": value,
            }
            for key, value
            in settings.items()
        ]
    ).to_csv(
        OUT_RUN_SUMMARY,
        sep="\t",
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Create the four exploratory structural figures."""
    print(
        f"Creating exploratory structural "
        f"figures for {BATCH}..."
    )

    validate_input_paths()

    master = pd.read_parquet(
        MASTER_FILE
    )

    ranking = pd.read_csv(
        ALL_FEATURE_RANKING_FILE,
        sep="\t",
    )

    manifest = pd.read_csv(
        MODEL_FEATURE_MANIFEST_FILE,
        sep="\t",
    )

    validate_input_tables(
        master=master,
        ranking=ranking,
        manifest=manifest,
    )

    model_ranking = (
        build_model_used_ranking(
            ranking=ranking,
            manifest=manifest,
        )
    )

    model_ranking.to_csv(
        OUT_MODEL_USED_RANKING,
        sep="\t",
        index=False,
    )

    figure_rows = []
    representative_rows = []

    figure_configuration = [
        {
            "feature_type": (
                "residue_ligand"
            ),
            "title": (
                "Top model-listed "
                "residue–ligand distances"
            ),
            "output_stem": (
                "top10_model_residue_ligand_"
                "effect_sizes"
            ),
        },
        {
            "feature_type": (
                "residue_cofactor"
            ),
            "title": (
                "Top model-listed "
                "residue–GTP distances"
            ),
            "output_stem": (
                "top10_model_residue_cofactor_"
                "effect_sizes"
            ),
        },
        {
            "feature_type": (
                "residue_residue"
            ),
            "title": (
                "Top model-listed "
                "residue–residue distances"
            ),
            "output_stem": (
                "top15_model_residue_residue_"
                "effect_sizes"
            ),
        },
    ]

    for configuration in (
        figure_configuration
    ):
        feature_type = configuration[
            "feature_type"
        ]

        top_n = TOP_N_BY_TYPE[
            feature_type
        ]

        plot_data = select_top_features(
            model_ranking=model_ranking,
            feature_type=feature_type,
            top_n=top_n,
        )

        source_file = (
            SOURCE_DATA_DIR
            / (
                f"04_{feature_type}_"
                f"effect_size_source.tsv"
            )
        )

        plot_data.to_csv(
            source_file,
            sep="\t",
            index=False,
        )

        png_file, svg_file = (
            plot_effect_size_ranking(
                plot_data=plot_data,
                title=configuration[
                    "title"
                ],
                output_stem=configuration[
                    "output_stem"
                ],
            )
        )

        figure_rows.append(
            {
                "figure_id": (
                    f"04_{feature_type}_ranking"
                ),
                "figure_role": (
                    "effect_size_ranking"
                ),
                "feature_type": (
                    feature_type
                ),
                "n_features_shown": (
                    len(plot_data)
                ),
                "selection_rule": (
                    "quality_pass and largest "
                    "absolute Cohen's d among "
                    "model-listed features"
                ),
                "source_data": (
                    str(source_file)
                ),
                "png_file": str(
                    png_file
                ),
                "svg_file": str(
                    svg_file
                ),
            }
        )

        representative_rows.append(
            plot_data.iloc[0]
        )

        print(
            f"Saved ranking figure for "
            f"{feature_type}: "
            f"{len(plot_data)} features"
        )

    representative_features = (
        pd.DataFrame(
            representative_rows
        )
        .reset_index(drop=True)
    )

    representative_features.to_csv(
        OUT_SELECTED_FEATURES,
        sep="\t",
        index=False,
    )

    (
        distribution_png,
        distribution_svg,
        distribution_source,
    ) = plot_representative_distributions(
        master=master,
        representative_features=(
            representative_features
        ),
    )

    distribution_source_file = (
        SOURCE_DATA_DIR
        / (
            "04_representative_feature_"
            "distribution_source.tsv"
        )
    )

    distribution_source.to_csv(
        distribution_source_file,
        sep="\t",
        index=False,
    )

    figure_rows.append(
        {
            "figure_id": (
                "04_representative_distributions"
            ),
            "figure_role": (
                "low_high_distribution"
            ),
            "feature_type": (
                "one representative per "
                "formal structural type"
            ),
            "n_features_shown": (
                len(representative_features)
            ),
            "selection_rule": (
                "top-ranked quality-passing "
                "model-listed feature within "
                "each feature type"
            ),
            "source_data": str(
                distribution_source_file
            ),
            "png_file": str(
                distribution_png
            ),
            "svg_file": str(
                distribution_svg
            ),
        }
    )

    figure_manifest = pd.DataFrame(
        figure_rows
    )

    figure_manifest.to_csv(
        OUT_FIGURE_MANIFEST,
        sep="\t",
        index=False,
    )

    save_run_summary(
        model_ranking=model_ranking,
        representative_features=(
            representative_features
        ),
    )

    print(
        "\nRepresentative features:"
    )

    print(
        representative_features[
            [
                "feature_type",
                "feature",
                "cohens_d",
                "difference_high_minus_low",
                "welch_p_value",
                "welch_fdr_bh",
            ]
        ].to_string(index=False)
    )

    print(
        "\nScientific figures created: 4"
    )

    print(
        "1. Residue-ligand effect-size ranking"
    )

    print(
        "2. Residue-cofactor effect-size ranking"
    )

    print(
        "3. Residue-residue effect-size ranking"
    )

    print(
        "4. Three-panel representative "
        "feature distribution figure"
    )

    print(
        "\nSaved figures to:"
    )

    print(FIGURE_DIR)

    print(
        "\nSaved figure source data to:"
    )

    print(SOURCE_DATA_DIR)

    print(
        "\nSaved figure manifest:"
    )

    print(OUT_FIGURE_MANIFEST)

    print("\nDone.")


if __name__ == "__main__":
    main()