#!/usr/bin/env python3
"""
09b_build_stable_feature_summary.py

Build a thesis-ready summary table for the three pre-defined stable
batch0100 residue-residue features.

This script does NOT:
- perform feature selection;
- train or evaluate models;
- calculate new low/high group comparisons;
- use August electrostatic-rerun distances.

It only combines:
1. Script 06 feature-selection stability evidence;
2. Script 09 February low/high descriptive statistics;
3. previously established residue/context annotations.

All quantitative classifier-related values remain February 2026 batch0100 data.

No dependency on the optional 'tabulate' package is required.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATS_TABLE = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "tables"
    / "09_stable_feature_group_statistics.tsv"
)

STABILITY_TABLE = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "tables"
    / "09_stable_feature_stability_evidence.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "tables"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TSV_OUT = OUTPUT_DIR / "09b_stable_feature_thesis_summary.tsv"
MD_OUT = OUTPUT_DIR / "09b_stable_feature_thesis_summary.md"


# =============================================================================
# Fixed feature definitions
# =============================================================================

STABLE_FEATURES = [
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
]


STRUCTURAL_CONTEXT = {
    "PRO62-PRO1279": (
        "cross-chain A-B residue pair"
    ),
    "PRO1275-PRO1276": (
        "consecutive-residue local backbone geometry"
    ),
    "PRO90-PRO92": (
        "local intrachain residue pair"
    ),
}


INTERPRETATION_NOTE = {
    "PRO62-PRO1279": (
        "Highest-stability feature; largest raw low/high mean shift; "
        "priority candidate for RAS-NF1 structural-context interpretation."
    ),

    "PRO1275-PRO1276": (
        "Large standardized effect but extremely small absolute distance shift; "
        "an August representative structure suggests a constrained peptide-"
        "backbone C-N geometry, so avoid describing this as a long-range "
        "interface contact."
    ),

    "PRO90-PRO92": (
        "Stable intrachain feature with a moderate absolute low/high distance "
        "shift; suitable as a secondary structural-context feature."
    ),
}


PROVENANCE_NOTE = {
    feature: (
        "All quantitative values are from the original February 2026 "
        "batch0100 classifier dataset. Residue identities were mapped using "
        "the original feature-generation logic and a retained August rerun "
        "structure; exact February atom pairs cannot be recovered because "
        "the original final PDBs were cleaned up."
    )
    for feature in STABLE_FEATURES
}


# =============================================================================
# Helper functions
# =============================================================================

def validate_inputs():
    """Check required Script 09 and stability tables exist."""

    if not STATS_TABLE.exists():
        raise FileNotFoundError(
            f"Missing Script 09 statistics table:\n{STATS_TABLE}"
        )

    if not STABILITY_TABLE.exists():
        raise FileNotFoundError(
            f"Missing stability evidence table:\n{STABILITY_TABLE}"
        )


def get_stability_row(df, feature, feature_set):
    """
    Retrieve exactly one stability row for a feature/feature-set combination.
    """

    rows = df[
        (df["feature"] == feature)
        & (df["feature_set"] == feature_set)
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one stability row for "
            f"{feature} / {feature_set}, found {len(rows)}"
        )

    return rows.iloc[0]


def escape_markdown(value):
    """
    Escape characters that could break a Markdown table.
    """

    text = str(value)

    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")

    return text


def dataframe_to_markdown_table(df):
    """
    Convert a DataFrame into a simple Markdown table without pandas.to_markdown()
    and therefore without requiring the optional 'tabulate' package.
    """

    columns = list(df.columns)

    lines = []

    # Header
    header = "| " + " | ".join(
        escape_markdown(col)
        for col in columns
    ) + " |"

    separator = "| " + " | ".join(
        "---"
        for _ in columns
    ) + " |"

    lines.append(header)
    lines.append(separator)

    # Rows
    for _, row in df.iterrows():
        values = [
            escape_markdown(row[col])
            for col in columns
        ]

        line = "| " + " | ".join(values) + " |"
        lines.append(line)

    return "\n".join(lines)


# =============================================================================
# Main analysis
# =============================================================================

def main():

    validate_inputs()

    stats_df = pd.read_csv(
        STATS_TABLE,
        sep="\t",
    )

    stability_df = pd.read_csv(
        STABILITY_TABLE,
        sep="\t",
    )

    print("=" * 80)
    print("09b THESIS-READY STABLE FEATURE SUMMARY")
    print("=" * 80)

    print(f"Script 09 statistics table:")
    print(STATS_TABLE)

    print()

    print(f"Stability evidence table:")
    print(STABILITY_TABLE)

    print()

    # -------------------------------------------------------------------------
    # Basic feature QC
    # -------------------------------------------------------------------------

    missing_stats = (
        set(STABLE_FEATURES)
        - set(stats_df["feature"])
    )

    if missing_stats:
        raise RuntimeError(
            "Missing stable features from Script 09 statistics: "
            + ", ".join(sorted(missing_stats))
        )

    missing_stability = (
        set(STABLE_FEATURES)
        - set(stability_df["feature"])
    )

    if missing_stability:
        raise RuntimeError(
            "Missing stable features from stability evidence: "
            + ", ".join(sorted(missing_stability))
        )

    print("Stable-feature QC:")

    for feature in STABLE_FEATURES:
        print(f"  FOUND: {feature}")

    print()

    # -------------------------------------------------------------------------
    # Combine Script 09 and Script 06 evidence
    # -------------------------------------------------------------------------

    records = []

    for feature in STABLE_FEATURES:

        stats_rows = stats_df[
            stats_df["feature"] == feature
        ]

        if len(stats_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one Script 09 statistics row "
                f"for {feature}, found {len(stats_rows)}"
            )

        stats_row = stats_rows.iloc[0]

        rr = get_stability_row(
            stability_df,
            feature,
            "residue_residue",
        )

        combined = get_stability_row(
            stability_df,
            feature,
            "combined_all",
        )

        record = {
            "feature":
                feature,

            "structural_label":
                stats_row["structural_label"],

            "structural_context":
                STRUCTURAL_CONTEXT[feature],

            "rr_selected_folds":
                int(rr["selected_outer_folds"]),

            "rr_total_folds":
                int(rr["total_outer_folds"]),

            "rr_selection_frequency":
                float(rr["selection_frequency"]),

            "combined_selected_folds":
                int(combined["selected_outer_folds"]),

            "combined_total_folds":
                int(combined["total_outer_folds"]),

            "combined_selection_frequency":
                float(combined["selection_frequency"]),

            "low_mean_A":
                float(stats_row["low_mean_A"]),

            "high_mean_A":
                float(stats_row["high_mean_A"]),

            "high_minus_low_mean_A":
                float(
                    stats_row["high_minus_low_mean_A"]
                ),

            "absolute_mean_shift_A":
                abs(
                    float(
                        stats_row["high_minus_low_mean_A"]
                    )
                ),

            "cohens_d_low_minus_high":
                float(
                    stats_row[
                        "cohens_d_low_minus_high"
                    ]
                ),

            "mannwhitney_p_raw":
                float(
                    stats_row[
                        "mannwhitney_p_raw"
                    ]
                ),

            "raw_distance_direction":
                stats_row[
                    "raw_distance_direction"
                ],

            "dominant_direction_rr":
                rr["dominant_direction"],

            "dominant_direction_fraction_rr":
                float(
                    rr[
                        "dominant_direction_fraction"
                    ]
                ),

            "stability_rank_rr":
                int(
                    rr[
                        "stability_rank_within_feature_set"
                    ]
                ),

            "interpretation_note":
                INTERPRETATION_NOTE[feature],

            "provenance_note":
                PROVENANCE_NOTE[feature],
        }

        records.append(record)

    out = pd.DataFrame(records)

    # Order by residue-residue stability rank.
    out = (
        out
        .sort_values("stability_rank_rr")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Cross-analysis consistency checks
    # -------------------------------------------------------------------------

    if not (
        out["rr_total_folds"] == 25
    ).all():
        raise RuntimeError(
            "Unexpected residue-residue total outer-fold count."
        )

    if not (
        out["combined_total_folds"] == 25
    ).all():
        raise RuntimeError(
            "Unexpected combined total outer-fold count."
        )

    # All three Script 09 features were found to have larger raw distances
    # in the low group. Check that this remains true in the summary.
    if not (
        out["raw_distance_direction"]
        == "higher_in_low"
    ).all():
        raise RuntimeError(
            "Unexpected raw-distance direction: "
            "not all three stable features are higher in the low group."
        )

    # Script 06 coefficients were recorded as predominantly towards low.
    if not (
        out["dominant_direction_rr"]
        == "towards_low"
    ).all():
        raise RuntimeError(
            "Unexpected coefficient direction in residue-residue stability data."
        )

    print(
        "QC PASSED: stability and February raw-distance "
        "directions are mutually consistent."
    )

    print()

    # -------------------------------------------------------------------------
    # Write full TSV
    # -------------------------------------------------------------------------

    out.to_csv(
        TSV_OUT,
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Create concise thesis-facing display table
    # -------------------------------------------------------------------------

    display = out[
        [
            "feature",
            "structural_label",
            "structural_context",
            "rr_selection_frequency",
            "combined_selection_frequency",
            "low_mean_A",
            "high_mean_A",
            "high_minus_low_mean_A",
            "cohens_d_low_minus_high",
            "dominant_direction_fraction_rr",
            "interpretation_note",
        ]
    ].copy()

    display = display.rename(
        columns={
            "feature":
                "Feature",

            "structural_label":
                "Residue pair",

            "structural_context":
                "Structural context",

            "rr_selection_frequency":
                "RR stability",

            "combined_selection_frequency":
                "Combined stability",

            "low_mean_A":
                "Low mean (A)",

            "high_mean_A":
                "High mean (A)",

            "high_minus_low_mean_A":
                "High-low mean (A)",

            "cohens_d_low_minus_high":
                "Cohen's d (low-high)",

            "dominant_direction_fraction_rr":
                "RR direction consistency",

            "interpretation_note":
                "Interpretation note",
        }
    )

    # Format values for human-readable Markdown output.
    display["RR stability"] = display[
        "RR stability"
    ].map(
        lambda x: f"{x:.2f}"
    )

    display["Combined stability"] = display[
        "Combined stability"
    ].map(
        lambda x: f"{x:.2f}"
    )

    display["Low mean (A)"] = display[
        "Low mean (A)"
    ].map(
        lambda x: f"{x:.3f}"
    )

    display["High mean (A)"] = display[
        "High mean (A)"
    ].map(
        lambda x: f"{x:.3f}"
    )

    display["High-low mean (A)"] = display[
        "High-low mean (A)"
    ].map(
        lambda x: f"{x:+.3f}"
    )

    display["Cohen's d (low-high)"] = display[
        "Cohen's d (low-high)"
    ].map(
        lambda x: f"{x:+.2f}"
    )

    display["RR direction consistency"] = display[
        "RR direction consistency"
    ].map(
        lambda x: f"{x:.3f}"
    )

    # -------------------------------------------------------------------------
    # Write Markdown manually - no tabulate dependency
    # -------------------------------------------------------------------------

    markdown_table = dataframe_to_markdown_table(
        display
    )

    with open(
        MD_OUT,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "# Stable residue-residue features: thesis-ready summary\n\n"
        )

        handle.write(
            "All quantitative classifier-related values in this table "
            "come from the original February 2026 batch0100 dataset. "
            "The three features were fixed from nested-CV feature-selection "
            "stability before this descriptive summary was constructed.\n\n"
        )

        handle.write(markdown_table)

        handle.write("\n\n")

        handle.write(
            "## Direction conventions\n\n"
        )

        handle.write(
            "- `High-low mean (A)` is the high-group mean minus the "
            "low-group mean. Negative values therefore mean that the raw "
            "distance is larger in the low group.\n"
        )

        handle.write(
            "- Positive `Cohen's d (low-high)` means that the raw distance "
            "is larger in the low group.\n"
        )

        handle.write(
            "- `RR direction consistency` is the dominant logistic-regression "
            "coefficient-direction fraction among outer folds in which the "
            "feature was selected.\n\n"
        )

        handle.write(
            "## Interpretation boundary\n\n"
        )

        handle.write(
            "The low/high comparisons are descriptive/post-hoc "
            "interpretations of features already identified by nested-CV "
            "stability. They should not be treated as independent confirmatory "
            "hypothesis tests. The primary evidence for classifier signal "
            "remains nested cross-validation, the dummy baseline, and the "
            "complete label-permutation test.\n\n"
        )

        handle.write(
            "These features characterize structural associations with "
            "old-RF-derived low/high labels and are not established causal "
            "determinants of catalytic barriers or improved-QM/MM barriers.\n\n"
        )

        handle.write(
            "Residue identities were established using the original "
            "feature-generation logic together with a retained August rerun "
            "structure. The original February final PDB structures were "
            "cleaned up, so exact February atom-pair identities cannot be "
            "recovered. August distances must therefore not be substituted "
            "for February classifier distances.\n"
        )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    console_columns = [
        "feature",
        "structural_label",
        "rr_selected_folds",
        "rr_selection_frequency",
        "combined_selected_folds",
        "combined_selection_frequency",
        "low_mean_A",
        "high_mean_A",
        "high_minus_low_mean_A",
        "absolute_mean_shift_A",
        "cohens_d_low_minus_high",
        "dominant_direction_fraction_rr",
        "stability_rank_rr",
    ]

    print("=" * 80)
    print("THESIS SUMMARY")
    print("=" * 80)

    print(
        out[
            console_columns
        ].to_string(index=False)
    )

    print()

    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print(TSV_OUT)
    print(MD_OUT)

    print()
    print("SCRIPT 09b COMPLETE.")


if __name__ == "__main__":
    main()
