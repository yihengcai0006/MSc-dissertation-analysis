"""
Perform an exploratory low-versus-high analysis of all batch0100 features.

This script evaluates every structural feature in the master table
created by 01_build_feature_barrier_table.py. The predefined intermediate
group is excluded, leaving the clear low- and high-barrier samples.

For each feature, the script calculates:

- available low/high sample counts and missingness;
- low/high means, medians and standard deviations;
- Cohen's d, defined as mean(high) - mean(low);
- Welch's unequal-variance t-test;
- two-sided Mann-Whitney U test;
- Pearson correlation with the continuous old RF-predicted barrier;
- Benjamini-Hochberg FDR-adjusted p-values;
- a label-independent data-quality flag.

Important
---------
The barriers and low/high labels come from the original RF workflow.
This analysis characterises the structural behaviour of that model and
does not test measured or improved QM/MM activation barriers.

This is an exploratory all-feature analysis. The formal classifier uses
the predefined model-used feature sets produced by script 03.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Configuration and paths
# ============================================================

BATCH = "batch0100"

MIN_LOW_COUNT = 20
MIN_HIGH_COUNT = 40
MAX_MISSING_FRACTION = 0.10
MIN_UNIQUE_VALUES = 5
NEAR_ZERO_STD_THRESHOLD = 1e-8

BASE_DIR = Path(__file__).resolve().parents[1]

IN_FILE = (
    BASE_DIR
    / "results"
    / BATCH
    / "tables"
    / "features_with_barriers.parquet"
)

TABLE_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_ALL_RANKED = (
    TABLE_DIR
    / "low_high_feature_ranking_all.tsv"
)

OUT_QUALITY_FILTERED = (
    TABLE_DIR
    / "low_high_feature_ranking_quality_filtered.tsv"
)

OUT_RESIDUE_LIGAND = (
    TABLE_DIR
    / "low_high_feature_ranking_residue_ligand.tsv"
)

OUT_RESIDUE_COFACTOR = (
    TABLE_DIR
    / "low_high_feature_ranking_residue_cofactor.tsv"
)

OUT_RESIDUE_RESIDUE = (
    TABLE_DIR
    / "low_high_feature_ranking_residue_residue.tsv"
)

OUT_COFACTOR_LIGAND = (
    TABLE_DIR
    / "low_high_feature_ranking_cofactor_ligand.tsv"
)

# Retained as a convenience table containing all feature names
# that explicitly involve LIG. It does not include PRO-GTP features.
OUT_LIGAND_RELATED = (
    TABLE_DIR
    / "low_high_feature_ranking_ligand_related.tsv"
)

OUT_TOP_HIGH = (
    TABLE_DIR
    / "top_features_higher_in_high.tsv"
)

OUT_TOP_LOW = (
    TABLE_DIR
    / "top_features_higher_in_low.tsv"
)

OUT_FEATURE_TYPE_SUMMARY = (
    TABLE_DIR
    / "all_feature_type_summary.tsv"
)

OUT_RUN_SUMMARY = (
    TABLE_DIR
    / "02_rank_all_low_high_features_summary.tsv"
)


# ============================================================
# Input validation
# ============================================================

def validate_input_table(
    table: pd.DataFrame,
) -> None:
    """Validate the master feature-barrier table."""
    required_columns = {
        "ligand_id",
        "barrier",
        "group",
    }

    missing_columns = (
        required_columns - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if table["ligand_id"].duplicated().any():
        raise ValueError(
            "Duplicated ligand IDs were found "
            "in the master table."
        )

    clear_groups = set(
        table.loc[
            table["group"].isin(
                ["low", "high"]
            ),
            "group",
        ]
    )

    if clear_groups != {"low", "high"}:
        raise ValueError(
            "Both low and high groups are required."
        )


# ============================================================
# Feature classification
# ============================================================

def classify_feature_type(
    feature_name: str,
) -> str:
    """
    Assign a structural feature type from the feature name.

    Examples
    --------
    PRO66-LIG:
        residue_ligand

    PRO66-GTP:
        residue_cofactor

    PRO66-PRO67:
        residue_residue

    GTP-LIG:
        cofactor_ligand
    """
    parts = str(
        feature_name
    ).upper().split("-")

    if len(parts) != 2:
        return "other"

    first, second = parts

    first_is_residue = first.startswith(
        "PRO"
    )

    second_is_residue = second.startswith(
        "PRO"
    )

    has_ligand = "LIG" in parts
    has_cofactor = "GTP" in parts

    if has_ligand and has_cofactor:
        return "cofactor_ligand"

    if has_ligand and (
        first_is_residue
        or second_is_residue
    ):
        return "residue_ligand"

    if first_is_residue and second_is_residue:
        return "residue_residue"

    if has_cofactor and (
        first_is_residue
        or second_is_residue
    ):
        return "residue_cofactor"

    if has_ligand:
        return "ligand_related"

    return "other"


# ============================================================
# Statistical helpers
# ============================================================

def cohens_d(
    high_values,
    low_values,
) -> float:
    """
    Calculate pooled-standard-deviation Cohen's d.

    The direction is:

        mean(high) - mean(low)

    Positive values indicate larger feature values in the high-barrier
    group. Negative values indicate larger values in the low-barrier
    group.
    """
    high_values = np.asarray(
        high_values,
        dtype=float,
    )

    low_values = np.asarray(
        low_values,
        dtype=float,
    )

    high_values = high_values[
        np.isfinite(high_values)
    ]

    low_values = low_values[
        np.isfinite(low_values)
    ]

    n_high = len(high_values)
    n_low = len(low_values)

    if n_high < 2 or n_low < 2:
        return np.nan

    variance_high = np.var(
        high_values,
        ddof=1,
    )

    variance_low = np.var(
        low_values,
        ddof=1,
    )

    pooled_variance = (
        (n_high - 1) * variance_high
        + (n_low - 1) * variance_low
    ) / (
        n_high + n_low - 2
    )

    if (
        not np.isfinite(
            pooled_variance
        )
        or pooled_variance <= 0
    ):
        return np.nan

    return (
        np.mean(high_values)
        - np.mean(low_values)
    ) / np.sqrt(pooled_variance)


def benjamini_hochberg(
    p_values,
) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_mask = np.isfinite(
        p_values
    )

    valid_values = p_values[
        valid_mask
    ]

    if len(valid_values) == 0:
        return adjusted

    order = np.argsort(
        valid_values
    )

    sorted_values = valid_values[
        order
    ]

    n_tests = len(
        sorted_values
    )

    ranks = np.arange(
        1,
        n_tests + 1,
    )

    adjusted_sorted = (
        sorted_values
        * n_tests
        / ranks
    )

    adjusted_sorted = (
        np.minimum.accumulate(
            adjusted_sorted[::-1]
        )[::-1]
    )

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    restored = np.empty(
        n_tests,
        dtype=float,
    )

    restored[order] = (
        adjusted_sorted
    )

    adjusted[valid_mask] = (
        restored
    )

    return adjusted


def safe_welch_ttest(
    high_values: pd.Series,
    low_values: pd.Series,
) -> tuple[float, float]:
    """Run Welch's t-test after removing missing values."""
    high_clean = high_values.dropna()
    low_clean = low_values.dropna()

    if (
        len(high_clean) < 2
        or len(low_clean) < 2
    ):
        return np.nan, np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            statistic, p_value = (
                stats.ttest_ind(
                    high_clean,
                    low_clean,
                    equal_var=False,
                    nan_policy="omit",
                )
            )

        return statistic, p_value

    except Exception:
        return np.nan, np.nan


def safe_mannwhitney(
    high_values: pd.Series,
    low_values: pd.Series,
) -> tuple[float, float]:
    """Run a two-sided Mann-Whitney U test."""
    high_clean = high_values.dropna()
    low_clean = low_values.dropna()

    if (
        len(high_clean) == 0
        or len(low_clean) == 0
    ):
        return np.nan, np.nan

    try:
        statistic, p_value = (
            stats.mannwhitneyu(
                high_clean,
                low_clean,
                alternative="two-sided",
                method="auto",
            )
        )

        return statistic, p_value

    except Exception:
        return np.nan, np.nan


def safe_pearson(
    feature_values: pd.Series,
    barrier_values: pd.Series,
) -> tuple[float, float, int]:
    """Calculate Pearson correlation using pairwise-complete rows."""
    paired = pd.DataFrame(
        {
            "feature": pd.to_numeric(
                feature_values,
                errors="coerce",
            ),
            "barrier": pd.to_numeric(
                barrier_values,
                errors="coerce",
            ),
        }
    ).dropna()

    n_pairs = len(paired)

    if n_pairs < 3:
        return np.nan, np.nan, n_pairs

    if paired["feature"].nunique() < 2:
        return np.nan, np.nan, n_pairs

    if paired["barrier"].nunique() < 2:
        return np.nan, np.nan, n_pairs

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            correlation, p_value = (
                stats.pearsonr(
                    paired["feature"],
                    paired["barrier"],
                )
            )

        return (
            correlation,
            p_value,
            n_pairs,
        )

    except Exception:
        return np.nan, np.nan, n_pairs


# ============================================================
# Output helpers
# ============================================================

def save_feature_type_subset(
    result: pd.DataFrame,
    feature_type: str,
    output_file: Path,
) -> None:
    """Save one feature-type-specific ranking table."""
    subset = result[
        result["feature_type"]
        == feature_type
    ].copy()

    subset.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print(
        f"Saved {feature_type}: "
        f"{len(subset)} features -> "
        f"{output_file}"
    )


def build_feature_type_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise feature counts and exploratory significance by type."""
    rows = []

    for feature_type, subset in (
        result.groupby(
            "feature_type",
            dropna=False,
        )
    ):
        quality_subset = subset[
            subset["quality_pass"]
        ]

        rows.append(
            {
                "feature_type": (
                    feature_type
                ),
                "n_features": len(
                    subset
                ),
                "n_quality_pass": int(
                    subset[
                        "quality_pass"
                    ].sum()
                ),
                "n_welch_fdr_below_0_05": int(
                    (
                        subset[
                            "welch_fdr_bh"
                        ] < 0.05
                    ).sum()
                ),
                "n_mannwhitney_fdr_below_0_05": int(
                    (
                        subset[
                            "mannwhitney_fdr_bh"
                        ] < 0.05
                    ).sum()
                ),
                "n_pearson_fdr_below_0_05": int(
                    (
                        subset[
                            "pearson_fdr_bh"
                        ] < 0.05
                    ).sum()
                ),
                "median_abs_cohens_d": (
                    quality_subset[
                        "abs_cohens_d"
                    ].median()
                ),
                "maximum_abs_cohens_d": (
                    quality_subset[
                        "abs_cohens_d"
                    ].max()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "n_features",
            ascending=False,
        )
    )


def save_run_summary(
    low_count: int,
    high_count: int,
    total_feature_count: int,
    result: pd.DataFrame,
) -> None:
    """Save analysis settings and key output counts."""
    settings = {
        "batch": BATCH,
        "input_file": str(IN_FILE),
        "barrier_source": (
            "old RF-predicted barriers"
        ),
        "n_low": low_count,
        "n_high": high_count,
        "n_low_high_total": (
            low_count + high_count
        ),
        "n_feature_columns": (
            total_feature_count
        ),
        "n_quality_pass": int(
            result[
                "quality_pass"
            ].sum()
        ),
        "cohens_d_definition": (
            "mean(high) - mean(low)"
        ),
        "minimum_low_count": (
            MIN_LOW_COUNT
        ),
        "minimum_high_count": (
            MIN_HIGH_COUNT
        ),
        "maximum_missing_fraction": (
            MAX_MISSING_FRACTION
        ),
        "minimum_unique_values": (
            MIN_UNIQUE_VALUES
        ),
        "near_zero_std_threshold": (
            NEAR_ZERO_STD_THRESHOLD
        ),
        "fdr_scope": (
            "all analysed feature columns"
        ),
        "analysis_role": (
            "exploratory all-feature "
            "low/high comparison"
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
# Main analysis
# ============================================================

def main() -> None:
    """Run the all-feature low-versus-high exploratory analysis."""
    if not IN_FILE.exists():
        raise FileNotFoundError(
            "Master feature-barrier table "
            "not found:\n"
            f"{IN_FILE}"
        )

    print(
        "Reading merged feature table..."
    )

    table = pd.read_parquet(
        IN_FILE
    )

    validate_input_table(
        table
    )

    print(
        "Table shape:",
        table.shape,
    )

    print("\nGroup counts:")

    print(
        table["group"]
        .value_counts()
        .to_string()
    )

    clear_table = table[
        table["group"].isin(
            ["low", "high"]
        )
    ].copy()

    low_table = clear_table[
        clear_table["group"] == "low"
    ]

    high_table = clear_table[
        clear_table["group"] == "high"
    ]

    print(
        "\nUsing clear low/high groups:"
    )

    print(
        "low count:",
        len(low_table),
    )

    print(
        "high count:",
        len(high_table),
    )

    metadata_columns = {
        "ligand_id",
        "barrier",
        "group",
    }

    feature_columns = [
        column
        for column in clear_table.columns
        if column not in metadata_columns
    ]

    print(
        "\nNumber of feature columns:",
        len(feature_columns),
    )

    rows = []

    for feature_index, feature in enumerate(
        feature_columns,
        start=1,
    ):
        low_values = pd.to_numeric(
            low_table[feature],
            errors="coerce",
        )

        high_values = pd.to_numeric(
            high_table[feature],
            errors="coerce",
        )

        all_values = pd.to_numeric(
            clear_table[feature],
            errors="coerce",
        )

        low_clean = low_values.dropna()
        high_clean = high_values.dropna()
        all_clean = all_values.dropna()

        n_low = len(low_clean)
        n_high = len(high_clean)
        n_total = len(all_clean)

        missing_count = (
            len(clear_table) - n_total
        )

        missing_fraction = (
            missing_count
            / len(clear_table)
        )

        low_mean = low_clean.mean()
        high_mean = high_clean.mean()

        low_median = low_clean.median()
        high_median = high_clean.median()

        low_std = low_clean.std(
            ddof=1
        )

        high_std = high_clean.std(
            ddof=1
        )

        overall_std = all_clean.std(
            ddof=1
        )

        n_unique = all_clean.nunique()

        difference = (
            high_mean - low_mean
        )

        effect_size = cohens_d(
            high_clean,
            low_clean,
        )

        (
            welch_statistic,
            welch_p_value,
        ) = safe_welch_ttest(
            high_values,
            low_values,
        )

        (
            mannwhitney_statistic,
            mannwhitney_p_value,
        ) = safe_mannwhitney(
            high_values,
            low_values,
        )

        (
            pearson_correlation,
            pearson_p_value,
            pearson_n,
        ) = safe_pearson(
            all_values,
            clear_table["barrier"],
        )

        near_zero_variance = (
            pd.isna(overall_std)
            or overall_std
            <= NEAR_ZERO_STD_THRESHOLD
        )

        quality_pass = (
            n_low >= MIN_LOW_COUNT
            and n_high >= MIN_HIGH_COUNT
            and missing_fraction
            <= MAX_MISSING_FRACTION
            and n_unique
            >= MIN_UNIQUE_VALUES
            and not near_zero_variance
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": (
                    classify_feature_type(
                        feature
                    )
                ),
                "n_low": n_low,
                "n_high": n_high,
                "n_total": n_total,
                "missing_count": (
                    missing_count
                ),
                "missing_fraction": (
                    missing_fraction
                ),
                "n_unique": n_unique,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "difference_high_minus_low": (
                    difference
                ),
                "low_median": (
                    low_median
                ),
                "high_median": (
                    high_median
                ),
                "low_std": low_std,
                "high_std": high_std,
                "overall_std": (
                    overall_std
                ),
                "cohens_d": effect_size,
                "abs_cohens_d": (
                    abs(effect_size)
                    if pd.notna(
                        effect_size
                    )
                    else np.nan
                ),
                "welch_t_stat": (
                    welch_statistic
                ),
                "welch_p_value": (
                    welch_p_value
                ),
                "mannwhitney_u": (
                    mannwhitney_statistic
                ),
                "mannwhitney_p_value": (
                    mannwhitney_p_value
                ),
                "pearson_n": (
                    pearson_n
                ),
                "pearson_corr_with_barrier": (
                    pearson_correlation
                ),
                "abs_pearson_corr": (
                    abs(
                        pearson_correlation
                    )
                    if pd.notna(
                        pearson_correlation
                    )
                    else np.nan
                ),
                "pearson_p_value": (
                    pearson_p_value
                ),
                "near_zero_variance": (
                    near_zero_variance
                ),
                "quality_pass": (
                    quality_pass
                ),
            }
        )

        if (
            feature_index % 5000 == 0
            or feature_index
            == len(feature_columns)
        ):
            print(
                f"Processed "
                f"{feature_index}/"
                f"{len(feature_columns)} "
                "features"
            )

    result = pd.DataFrame(
        rows
    )

    print(
        "\nApplying all-feature "
        "multiple-testing correction..."
    )

    result["welch_fdr_bh"] = (
        benjamini_hochberg(
            result[
                "welch_p_value"
            ]
        )
    )

    result[
        "mannwhitney_fdr_bh"
    ] = benjamini_hochberg(
        result[
            "mannwhitney_p_value"
        ]
    )

    result["pearson_fdr_bh"] = (
        benjamini_hochberg(
            result[
                "pearson_p_value"
            ]
        )
    )

    result = result.sort_values(
        by=[
            "quality_pass",
            "abs_cohens_d",
            "welch_fdr_bh",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last",
    )

    result.to_csv(
        OUT_ALL_RANKED,
        sep="\t",
        index=False,
    )

    quality_result = result[
        result["quality_pass"]
    ].copy()

    quality_result.to_csv(
        OUT_QUALITY_FILTERED,
        sep="\t",
        index=False,
    )

    save_feature_type_subset(
        result,
        "residue_ligand",
        OUT_RESIDUE_LIGAND,
    )

    save_feature_type_subset(
        result,
        "residue_cofactor",
        OUT_RESIDUE_COFACTOR,
    )

    save_feature_type_subset(
        result,
        "residue_residue",
        OUT_RESIDUE_RESIDUE,
    )

    save_feature_type_subset(
        result,
        "cofactor_ligand",
        OUT_COFACTOR_LIGAND,
    )

    ligand_related = result[
        result["feature_type"].isin(
            {
                "residue_ligand",
                "cofactor_ligand",
                "ligand_related",
            }
        )
    ].copy()

    ligand_related.to_csv(
        OUT_LIGAND_RELATED,
        sep="\t",
        index=False,
    )

    top_high = quality_result[
        quality_result[
            "difference_high_minus_low"
        ] > 0
    ].head(30)

    top_low = quality_result[
        quality_result[
            "difference_high_minus_low"
        ] < 0
    ].head(30)

    top_high.to_csv(
        OUT_TOP_HIGH,
        sep="\t",
        index=False,
    )

    top_low.to_csv(
        OUT_TOP_LOW,
        sep="\t",
        index=False,
    )

    feature_type_summary = (
        build_feature_type_summary(
            result
        )
    )

    feature_type_summary.to_csv(
        OUT_FEATURE_TYPE_SUMMARY,
        sep="\t",
        index=False,
    )

    save_run_summary(
        low_count=len(low_table),
        high_count=len(high_table),
        total_feature_count=len(
            feature_columns
        ),
        result=result,
    )

    print(
        "\nFeature-type summary:"
    )

    print(
        feature_type_summary
        .to_string(index=False)
    )

    columns_to_show = [
        "feature",
        "feature_type",
        "n_low",
        "n_high",
        "missing_fraction",
        "low_mean",
        "high_mean",
        "difference_high_minus_low",
        "cohens_d",
        "welch_p_value",
        "welch_fdr_bh",
        "mannwhitney_p_value",
        "pearson_corr_with_barrier",
    ]

    print(
        "\nTop 20 quality-passing "
        "features:"
    )

    print(
        quality_result[
            columns_to_show
        ].head(20).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(OUT_ALL_RANKED)
    print(OUT_QUALITY_FILTERED)
    print(OUT_RESIDUE_LIGAND)
    print(OUT_RESIDUE_COFACTOR)
    print(OUT_RESIDUE_RESIDUE)
    print(OUT_COFACTOR_LIGAND)
    print(OUT_LIGAND_RELATED)
    print(OUT_TOP_HIGH)
    print(OUT_TOP_LOW)
    print(OUT_FEATURE_TYPE_SUMMARY)
    print(OUT_RUN_SUMMARY)

    print("\nDone.")


if __name__ == "__main__":
    main()
