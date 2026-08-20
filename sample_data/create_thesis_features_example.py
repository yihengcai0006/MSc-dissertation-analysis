#!/usr/bin/env python3

"""
Create a compact GitHub sample from the completed February 2026
100-ligand structural-feature dataset.

IMPORTANT
---------
This script is for repository packaging only.

It:
- reads existing parquet files;
- extracts only three already-defined stable feature columns;
- writes a new compact TSV file under sample_data/.

It does NOT:
- modify any original parquet file;
- modify the barrier file;
- modify any result table;
- rerun nested cross-validation;
- rerun feature selection;
- rerun label permutation;
- recalculate dissertation results.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = (
    ROOT
    / "thesis_features_analysis"
    / "data"
    / "raw"
    / "batch0100"
    / "features"
)

OUTPUT_DIR = (
    ROOT
    / "sample_data"
    / "thesis_features_example"
)

OUTPUT_FILE = OUTPUT_DIR / "example_feature_values.tsv"


# ============================================================
# Configuration
# ============================================================

# A small deterministic sample is enough to demonstrate
# the input format without duplicating the full ~4 GB dataset.
N_EXAMPLE_LIGANDS = 6

STABLE_FEATURES = [
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
]


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 72)
    print("Creating compact thesis_features_analysis GitHub sample")
    print("=" * 72)

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found:\n{FEATURE_DIR}"
        )

    parquet_files = sorted(FEATURE_DIR.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise FileNotFoundError(
            f"No parquet files found in:\n{FEATURE_DIR}"
        )

    print(f"Found {len(parquet_files)} original parquet files.")

    sample_files = parquet_files[:N_EXAMPLE_LIGANDS]

    print()
    print("Example files selected:")
    for path in sample_files:
        print(f"  {path.name}")

    rows = []

    for parquet_path in sample_files:
        ligand_id = parquet_path.stem

        # Inspect schema without loading the full ~38 MB file.
        schema = pq.read_schema(parquet_path)
        available_columns = set(schema.names)

        missing = [
            feature
            for feature in STABLE_FEATURES
            if feature not in available_columns
        ]

        if missing:
            raise KeyError(
                f"{parquet_path.name} is missing expected columns: "
                f"{missing}"
            )

        # Read ONLY the three target columns.
        # The remaining ~83,000 columns are never loaded.
        feature_df = pd.read_parquet(
            parquet_path,
            columns=STABLE_FEATURES,
        )

        if len(feature_df) != 1:
            raise ValueError(
                f"Expected one row in {parquet_path.name}, "
                f"found {len(feature_df)}."
            )

        row = {
            "ligand_id": ligand_id,
        }

        for feature in STABLE_FEATURES:
            row[feature] = feature_df.iloc[0][feature]

        rows.append(row)

    output_df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    print()
    print("Written:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("Sample shape:")
    print(f"  rows: {len(output_df)}")
    print(f"  columns: {len(output_df.columns)}")

    print()
    print("Columns:")
    for column in output_df.columns:
        print(f"  {column}")

    print()
    print("=" * 72)
    print("Original scientific analysis unchanged")
    print("=" * 72)
    print("Original parquet files modified: False")
    print("Nested CV rerun: False")
    print("Feature selection rerun: False")
    print("Permutation test rerun: False")
    print("Completed result tables modified: False")
    print()
    print("Done.")


if __name__ == "__main__":
    main()