#!/usr/bin/env python3

"""
Select representative February batch0100 molecules for PyMOL figures.

Representatives are selected objectively as the molecules whose feature
values are closest to the median value of their low/high group.

No classifier fitting or feature selection is performed here.
The features were fixed previously by nested-CV stability analysis.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "tables"
    / "09_stable_feature_values.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "batch0100"
    / "structural_interpretation"
    / "tables"
)

OUTPUT = OUTPUT_DIR / "10_pymol_representative_structures.tsv"

FEATURES = [
    "PRO62-PRO1279",
    "PRO90-PRO92",
]

STRUCTURAL_LABELS = {
    "PRO62-PRO1279": "GLU62–SER1279",
    "PRO90-PRO92": "PHE90–ASP92",
}


def select_median_representative(df, feature, group):
    subset = df[
        (df["feature"] == feature)
        & (df["group"] == group)
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"No values found for {feature}, group={group}"
        )

    subset["distance_A"] = pd.to_numeric(
        subset["distance_A"],
        errors="raise",
    )

    median = subset["distance_A"].median()

    subset["absolute_distance_from_group_median_A"] = (
        subset["distance_A"] - median
    ).abs()

    # Stable deterministic tie-breaking.
    subset = subset.sort_values(
        [
            "absolute_distance_from_group_median_A",
            "molecule",
        ]
    )

    selected = subset.iloc[0]

    return {
        "feature": feature,
        "structural_label": STRUCTURAL_LABELS[feature],
        "group": group,
        "group_median_A": median,
        "selected_molecule": selected["molecule"],
        "selected_feature_value_A": selected["distance_A"],
        "absolute_distance_from_group_median_A":
            selected["absolute_distance_from_group_median_A"],
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, sep="\t")

    required = {
        "molecule",
        "feature",
        "group",
        "distance_A",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    records = []

    for feature in FEATURES:
        for group in ["low", "high"]:
            records.append(
                select_median_representative(
                    df,
                    feature,
                    group,
                )
            )

    out = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
    )

    print("=" * 80)
    print("PYMOL REPRESENTATIVE STRUCTURES")
    print("=" * 80)
    print(out.to_string(index=False))
    print()
    print("Output:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
