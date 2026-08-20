#!/usr/bin/env python3
"""
02_extract_prefixed_features.py

Extract the three pre-fixed residue-residue features from the corrected
47-complete-structure feature matrix and merge them with the improved-QM/MM
barriers.

IMPORTANT:
- The three features were fixed in advance from the independent February
  batch0100 nested-CV stability analysis.
- No feature screening or ranking is performed on the 47-structure dataset.
- No model is trained.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "existing_features"
    / "feature_matrix_47.parquet"
)

BARRIER_FILE = (
    PROJECT_ROOT
    / "qmmm_validation"
    / "barriers.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "02_qmmm47_prefixed_feature_values.tsv"
)

LOG_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "02_extract_prefixed_features.log"
)


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


ID_CANDIDATES = [
    "ligand_id",
    "molecule",
    "Molecule",
    "molid",
    "ligand",
]


def find_id_column(df):
    """Identify the ligand-ID column in the feature matrix."""
    for column in ID_CANDIDATES:
        if column in df.columns:
            return column

    raise RuntimeError(
        "Could not identify ligand-ID column in feature matrix."
    )


def load_inputs():
    """Load corrected feature matrix and improved-QM/MM barriers."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    if not BARRIER_FILE.exists():
        raise FileNotFoundError(BARRIER_FILE)

    features = pd.read_parquet(FEATURE_FILE)
    barriers = pd.read_csv(BARRIER_FILE)

    return features, barriers


def validate_prefixed_features(features):
    """Verify that all three pre-fixed features are available."""
    missing = [
        feature
        for feature in PREFIXED_FEATURES
        if feature not in features.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing pre-fixed features: "
            + ", ".join(missing)
        )


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    features, barriers = load_inputs()
    validate_prefixed_features(features)

    feature_id_col = find_id_column(features)

    feature_subset = features[
        [feature_id_col] + PREFIXED_FEATURES
    ].copy()

    feature_subset = feature_subset.rename(
        columns={feature_id_col: "ligand_id"}
    )

    feature_subset["ligand_id"] = (
        feature_subset["ligand_id"]
        .astype(str)
        .str.strip()
    )

    barriers = barriers.copy()

    barriers["ligand_id"] = (
        barriers["ligand_id"]
        .astype(str)
        .str.strip()
    )

    # Keep useful provenance metadata from the supplied barrier table.
    barrier_columns = [
        "ligand_id",
        "barrier_kcal_per_mol",
        "n_sp_windows",
        "converged_scfs",
        "source_batch",
    ]

    missing_barrier_columns = [
        column
        for column in barrier_columns
        if column not in barriers.columns
    ]

    if missing_barrier_columns:
        raise RuntimeError(
            "Missing expected barrier columns: "
            + ", ".join(missing_barrier_columns)
        )

    merged = barriers[barrier_columns].merge(
        feature_subset,
        on="ligand_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 47:
        raise RuntimeError(
            f"Expected 47 merged rows, found {len(merged)}"
        )

    if merged["barrier_kcal_per_mol"].isna().any():
        raise RuntimeError(
            "Missing improved-QM/MM barrier after merge."
        )

    for feature in PREFIXED_FEATURES:
        if merged[feature].isna().any():
            raise RuntimeError(
                f"Missing values found for {feature}"
            )

    # Sort deterministically for reproducible output.
    merged = merged.sort_values(
        "ligand_id"
    ).reset_index(drop=True)

    merged.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    print("=" * 80)
    print("QMMM47 PRE-FIXED FEATURE EXTRACTION")
    print("=" * 80)

    print(f"Merged rows: {len(merged)}")
    print()

    print("Feature variation:")
    print()

    variation_records = []

    for feature in PREFIXED_FEATURES:
        values = merged[feature]

        n_unique = int(values.nunique(dropna=True))
        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())
        std = float(values.std(ddof=1))

        variation_records.append(
            {
                "feature": feature,
                "structural_label": STRUCTURAL_LABELS[feature],
                "n_unique": n_unique,
                "min_A": minimum,
                "max_A": maximum,
                "range_A": maximum - minimum,
                "mean_A": mean,
                "std_A": std,
            }
        )

        print(
            f"{feature:20s} "
            f"{STRUCTURAL_LABELS[feature]:20s} "
            f"n_unique={n_unique:2d}  "
            f"min={minimum:.6f}  "
            f"max={maximum:.6f}  "
            f"range={maximum - minimum:.6f}  "
            f"std={std:.6f}"
        )

    print()
    print("Improved-QM/MM barrier:")
    print(
        f"min={merged['barrier_kcal_per_mol'].min():.3f}  "
        f"max={merged['barrier_kcal_per_mol'].max():.3f}  "
        f"mean={merged['barrier_kcal_per_mol'].mean():.3f}  "
        f"std={merged['barrier_kcal_per_mol'].std(ddof=1):.3f}"
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    with open(LOG_FILE, "w", encoding="utf-8") as handle:
        handle.write(
            "QMMM47 pre-fixed feature extraction\n"
        )
        handle.write("=" * 60 + "\n\n")

        handle.write(f"Merged rows: {len(merged)}\n\n")

        handle.write("Feature variation:\n")
        handle.write(
            pd.DataFrame(
                variation_records
            ).to_string(index=False)
        )

        handle.write("\n\n")
        handle.write(
            "The three features were fixed independently from the "
            "February batch0100 nested-CV stability analysis. "
            "No screening of the 47-structure dataset was performed.\n"
        )

    print()
    print(LOG_FILE)
    print()
    print("SCRIPT 02 COMPLETE.")


if __name__ == "__main__":
    main()
