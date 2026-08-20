#!/usr/bin/env python3
"""
09_analyse_stable_feature_distributions.py

Descriptive/post-hoc interpretation of the three pre-defined stable
residue-residue features identified by the formal batch0100 nested-CV
feature-stability analysis.

IMPORTANT PROVENANCE RULE
-------------------------
This script uses ONLY the original February 2026 batch0100 classifier data.

It does NOT:
- use August electrostatic-rerun coordinates or features;
- re-run feature selection;
- select features using low/high group differences;
- re-train the classifier;
- modify the nested-CV results.

The three features analysed here were fixed in advance from Script 06
feature-selection stability:

    PRO62-PRO1279
    PRO1275-PRO1276
    PRO90-PRO92

The statistical comparisons in this script are descriptive/post-hoc
interpretation of already-selected features and should not be treated as
independent confirmatory hypothesis tests.
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

MASTER_TABLE = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "tables"
    / "features_with_barriers.parquet"
)

STABILITY_TABLE = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "stability"
    / "tables"
    / "candidate_feature_stability.tsv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

for directory in (TABLE_DIR, FIGURE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Fixed features
# =============================================================================

STABLE_FEATURES = [
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
]

STRUCTURAL_LABELS = {
    "PRO62-PRO1279": "GLU62–SER1279",
    "PRO1275-PRO1276": "PHE1275–ARG1276",
    "PRO90-PRO92": "PHE90–ASP92",
}

# These residue identities were mapped using the retained topology/rerun
# structure and original feature-generation code. Exact February atom pairs
# cannot be recovered because the original February final PDBs were cleaned up.

GROUP_ORDER = ["low", "high"]

RANDOM_SEED = 20260809


# =============================================================================
# Helper functions
# =============================================================================

def find_column(df, candidates, required=True):
    """Return the first matching column name from candidates."""
    for name in candidates:
        if name in df.columns:
            return name

    if required:
        raise KeyError(
            "Could not find any of the expected columns: "
            + ", ".join(candidates)
        )

    return None


def cohen_d(x, y):
    """
    Cohen's d for two independent groups.

    Here x = low and y = high, so positive d means the low-group
    mean is larger than the high-group mean.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n1 = len(x)
    n2 = len(y)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(x, ddof=1)
    var2 = np.var(y, ddof=1)

    pooled_variance = (
        ((n1 - 1) * var1 + (n2 - 1) * var2)
        / (n1 + n2 - 2)
    )

    if pooled_variance <= 0:
        return np.nan

    pooled_sd = math.sqrt(pooled_variance)

    return (np.mean(x) - np.mean(y)) / pooled_sd


def iqr(values):
    """Interquartile range."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    q25, q75 = np.percentile(values, [25, 75])
    return q75 - q25


def bh_fdr(p_values):
    """
    Benjamini-Hochberg correction.

    Included only as a descriptive multiple-comparison summary over the
    three already-fixed features.
    """
    p_values = np.asarray(p_values, dtype=float)

    result = np.full_like(p_values, np.nan, dtype=float)

    valid = np.isfinite(p_values)
    p = p_values[valid]

    if len(p) == 0:
        return result

    order = np.argsort(p)
    ranked = p[order]

    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)

    # Enforce monotonicity from largest rank backwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted

    result[valid] = restored

    return result


def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())

    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan, np.nan, int(mask.sum())

    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def safe_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())

    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


# =============================================================================
# Load and validate February master table
# =============================================================================

if not MASTER_TABLE.exists():
    raise FileNotFoundError(
        f"February master table not found:\n{MASTER_TABLE}"
    )

df = pd.read_parquet(MASTER_TABLE)

print("=" * 80)
print("SCRIPT 09: STABLE FEATURE DISTRIBUTION ANALYSIS")
print("=" * 80)
print(f"Input table: {MASTER_TABLE}")
print(f"Input shape: {df.shape}")
print()


# Detect key metadata columns robustly.
group_col = find_column(
    df,
    [
        "group",
        "Group",
        "barrier_group",
        "low_high_group",
    ],
)

molecule_col = find_column(
    df,
    [
        "Molecule",
        "molecule",
        "molid",
        "ligand_id",
        "Ligand",
        "ligand",
    ],
    required=False,
)

barrier_col = find_column(
    df,
    [
        "barrier",
        "Barrier",
        "predicted_barrier",
        "old_rf_barrier",
        "barriers",
    ],
    required=False,
)

print(f"Group column:    {group_col}")
print(f"Molecule column: {molecule_col}")
print(f"Barrier column:  {barrier_col}")
print()


# Validate target features.
missing_features = [
    feature
    for feature in STABLE_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise KeyError(
        "Stable features missing from February master table:\n"
        + "\n".join(missing_features)
    )

print("Stable features found:")
for feature in STABLE_FEATURES:
    print(f"  FOUND: {feature}")

print()


# =============================================================================
# Group QC
# =============================================================================

group_series = df[group_col].astype(str).str.lower().str.strip()

group_counts_all = group_series.value_counts(dropna=False)

print("All group counts:")
print(group_counts_all.to_string())
print()

analysis_mask = group_series.isin(GROUP_ORDER)

analysis_df = df.loc[analysis_mask].copy()
analysis_df["_analysis_group"] = group_series.loc[analysis_mask]

group_counts = (
    analysis_df["_analysis_group"]
    .value_counts()
    .reindex(GROUP_ORDER)
)

print("Classifier-analysis group counts:")
print(group_counts.to_string())
print()

expected_low = 26
expected_high = 57

actual_low = int(group_counts.get("low", 0))
actual_high = int(group_counts.get("high", 0))

if actual_low != expected_low or actual_high != expected_high:
    raise RuntimeError(
        "Unexpected February low/high counts. "
        f"Expected low={expected_low}, high={expected_high}; "
        f"found low={actual_low}, high={actual_high}."
    )

print("QC PASSED: low=26, high=57, total=83.")
print()


# =============================================================================
# Per-molecule long-form table
# =============================================================================

value_records = []

for feature in STABLE_FEATURES:
    for idx, row in analysis_df.iterrows():
        record = {
            "feature": feature,
            "structural_label": STRUCTURAL_LABELS[feature],
            "group": row["_analysis_group"],
            "distance_A": row[feature],
        }

        if molecule_col is not None:
            record["molecule"] = row[molecule_col]
        else:
            record["molecule"] = idx

        if barrier_col is not None:
            record["old_rf_barrier"] = row[barrier_col]

        value_records.append(record)

values_long = pd.DataFrame(value_records)

values_out = TABLE_DIR / "09_stable_feature_values.tsv"
values_long.to_csv(values_out, sep="\t", index=False)


# =============================================================================
# Descriptive group statistics
# =============================================================================

summary_records = []

for feature in STABLE_FEATURES:

    low = pd.to_numeric(
        analysis_df.loc[
            analysis_df["_analysis_group"] == "low",
            feature,
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    high = pd.to_numeric(
        analysis_df.loc[
            analysis_df["_analysis_group"] == "high",
            feature,
        ],
        errors="coerce",
    ).to_numpy(dtype=float)

    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]

    if len(low) == 0 or len(high) == 0:
        raise RuntimeError(
            f"No usable low/high values for {feature}"
        )

    # Welch independent-samples t test.
    welch = stats.ttest_ind(
        low,
        high,
        equal_var=False,
        nan_policy="omit",
    )

    # Two-sided Mann-Whitney U.
    mwu = stats.mannwhitneyu(
        low,
        high,
        alternative="two-sided",
    )

    low_mean = float(np.mean(low))
    high_mean = float(np.mean(high))

    # Define both directions explicitly to avoid ambiguity.
    high_minus_low = high_mean - low_mean
    low_minus_high = low_mean - high_mean

    record = {
        "feature": feature,
        "structural_label": STRUCTURAL_LABELS[feature],

        "n_low": len(low),
        "n_high": len(high),

        "low_mean_A": low_mean,
        "high_mean_A": high_mean,

        "low_median_A": float(np.median(low)),
        "high_median_A": float(np.median(high)),

        "low_std_A": float(np.std(low, ddof=1)),
        "high_std_A": float(np.std(high, ddof=1)),

        "low_iqr_A": float(iqr(low)),
        "high_iqr_A": float(iqr(high)),

        "high_minus_low_mean_A": high_minus_low,
        "low_minus_high_mean_A": low_minus_high,

        # Positive Cohen's d = larger distance in low group.
        "cohens_d_low_minus_high": float(cohen_d(low, high)),

        "welch_t_statistic": float(welch.statistic),
        "welch_p_raw": float(welch.pvalue),

        "mannwhitney_u": float(mwu.statistic),
        "mannwhitney_p_raw": float(mwu.pvalue),
    }

    # Optional descriptive correlation with the old-RF-derived barrier.
    if barrier_col is not None:
        feature_values = pd.to_numeric(
            analysis_df[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

        barrier_values = pd.to_numeric(
            analysis_df[barrier_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        pearson_r, pearson_p, pearson_n = safe_pearson(
            feature_values,
            barrier_values,
        )

        spearman_rho, spearman_p, spearman_n = safe_spearman(
            feature_values,
            barrier_values,
        )

        record.update({
            "pearson_r_vs_old_rf_barrier": pearson_r,
            "pearson_p_raw": pearson_p,
            "pearson_n": pearson_n,

            "spearman_rho_vs_old_rf_barrier": spearman_rho,
            "spearman_p_raw": spearman_p,
            "spearman_n": spearman_n,
        })

    summary_records.append(record)


summary_df = pd.DataFrame(summary_records)

# Descriptive BH correction across the three already-fixed features.
summary_df["welch_q_bh_3features"] = bh_fdr(
    summary_df["welch_p_raw"].to_numpy()
)

summary_df["mannwhitney_q_bh_3features"] = bh_fdr(
    summary_df["mannwhitney_p_raw"].to_numpy()
)

if "pearson_p_raw" in summary_df.columns:
    summary_df["pearson_q_bh_3features"] = bh_fdr(
        summary_df["pearson_p_raw"].to_numpy()
    )

if "spearman_p_raw" in summary_df.columns:
    summary_df["spearman_q_bh_3features"] = bh_fdr(
        summary_df["spearman_p_raw"].to_numpy()
    )


# Add a plain-language direction column.
directions = []

for _, row in summary_df.iterrows():
    delta = row["high_minus_low_mean_A"]

    if np.isclose(delta, 0.0):
        direction = "approximately_equal"
    elif delta > 0:
        direction = "higher_in_high"
    else:
        direction = "higher_in_low"

    directions.append(direction)

summary_df["raw_distance_direction"] = directions


summary_out = TABLE_DIR / "09_stable_feature_group_statistics.tsv"
summary_df.to_csv(summary_out, sep="\t", index=False)


# =============================================================================
# Stability evidence: extract only the same pre-defined features
# =============================================================================

stability_out = (
    TABLE_DIR
    / "09_stable_feature_stability_evidence.tsv"
)

if STABILITY_TABLE.exists():

    stability_df = pd.read_csv(
        STABILITY_TABLE,
        sep="\t",
    )

    required_stability_cols = {
        "feature_set",
        "feature",
        "selected_outer_folds",
        "total_outer_folds",
        "selection_frequency",
        "dominant_direction",
        "dominant_direction_fraction",
        "stability_rank_within_feature_set",
    }

    missing = (
        required_stability_cols
        - set(stability_df.columns)
    )

    if missing:
        raise KeyError(
            "Missing expected columns from stability table: "
            + ", ".join(sorted(missing))
        )

    stable_evidence = stability_df[
        stability_df["feature"].isin(STABLE_FEATURES)
        & stability_df["feature_set"].isin(
            ["residue_residue", "combined_all"]
        )
    ].copy()

    stable_evidence["structural_label"] = (
        stable_evidence["feature"]
        .map(STRUCTURAL_LABELS)
    )

    stable_evidence.to_csv(
        stability_out,
        sep="\t",
        index=False,
    )

else:
    print(
        "WARNING: stability table not found; "
        "stability evidence output skipped."
    )


# =============================================================================
# Figure: raw February low/high distributions
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)

fig, axes = plt.subplots(
    1,
    len(STABLE_FEATURES),
    figsize=(12, 4.6),
    sharey=False,
)

if len(STABLE_FEATURES) == 1:
    axes = [axes]

for ax, feature in zip(axes, STABLE_FEATURES):

    low = values_long.loc[
        (values_long["feature"] == feature)
        & (values_long["group"] == "low"),
        "distance_A",
    ].astype(float).to_numpy()

    high = values_long.loc[
        (values_long["feature"] == feature)
        & (values_long["group"] == "high"),
        "distance_A",
    ].astype(float).to_numpy()

    datasets = [low, high]

    # Boxplots show median/IQR without hiding individual observations.
    ax.boxplot(
        datasets,
        positions=[1, 2],
        widths=0.5,
        showfliers=False,
    )

    # Overlay all individual February observations.
    for x_position, values in zip(
        [1, 2],
        datasets,
    ):
        jitter = rng.normal(
            loc=0.0,
            scale=0.045,
            size=len(values),
        )

        ax.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            s=24,
            alpha=0.7,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Low", "High"])

    ax.set_title(
        f"{feature}\n{STRUCTURAL_LABELS[feature]}"
    )

    ax.set_ylabel(
        "Minimum inter-residue distance (Å)"
    )

    # Small descriptive annotation.
    stats_row = summary_df.loc[
        summary_df["feature"] == feature
    ].iloc[0]

    delta = stats_row["high_minus_low_mean_A"]
    d = stats_row["cohens_d_low_minus_high"]

    annotation = (
        f"High − low mean = {delta:+.3f} Å\n"
        f"Cohen's d (low − high) = {d:+.2f}"
    )

    ax.text(
        0.5,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )


fig.suptitle(
    "batch0100: distributions of three stable "
    "residue–residue features",
    fontsize=12,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

png_out = (
    FIGURE_DIR
    / "09_stable_feature_low_high_distributions.png"
)

svg_out = (
    FIGURE_DIR
    / "09_stable_feature_low_high_distributions.svg"
)

fig.savefig(
    png_out,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    svg_out,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Run summary
# =============================================================================

log_out = LOG_DIR / "09_run_summary.txt"

with open(log_out, "w") as handle:

    handle.write(
        "Script 09: stable feature distribution analysis\n"
    )
    handle.write("=" * 60 + "\n\n")

    handle.write(
        "DATA PROVENANCE\n"
        "---------------\n"
        "Only the original February 2026 batch0100 "
        "classifier data were analysed.\n"
        "No August electrostatic-rerun coordinates or "
        "features were used.\n\n"
    )

    handle.write(
        f"Input master table:\n{MASTER_TABLE}\n\n"
    )

    handle.write(
        f"Input shape: {df.shape}\n"
    )

    handle.write(
        f"Analysis samples: {len(analysis_df)}\n"
    )

    handle.write(
        f"Low: {actual_low}\n"
    )

    handle.write(
        f"High: {actual_high}\n\n"
    )

    handle.write(
        "PRE-DEFINED FEATURES\n"
        "--------------------\n"
    )

    for feature in STABLE_FEATURES:
        handle.write(
            f"{feature}\t{STRUCTURAL_LABELS[feature]}\n"
        )

    handle.write("\n")

    handle.write(
        "IMPORTANT INTERPRETATION NOTE\n"
        "-----------------------------\n"
        "These group comparisons are descriptive/post-hoc "
        "interpretations of features already identified by "
        "nested-CV stability. Their p-values are not independent "
        "confirmatory tests of newly specified hypotheses.\n\n"
    )

    handle.write(
        "SUMMARY\n"
        "-------\n"
    )

    handle.write(
        summary_df.to_string(index=False)
    )

    handle.write("\n")


# =============================================================================
# Console output
# =============================================================================

print("=" * 80)
print("RESULTS")
print("=" * 80)

display_columns = [
    "feature",
    "structural_label",
    "n_low",
    "n_high",
    "low_mean_A",
    "high_mean_A",
    "high_minus_low_mean_A",
    "cohens_d_low_minus_high",
    "mannwhitney_p_raw",
    "raw_distance_direction",
]

print(
    summary_df[display_columns]
    .to_string(index=False)
)

print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)
print(summary_out)
print(values_out)

if STABILITY_TABLE.exists():
    print(stability_out)

print(png_out)
print(svg_out)
print(log_out)

print()
print("SCRIPT 09 COMPLETE.")


