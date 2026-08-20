#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from original_feature_functions import (
    compute_pp_dists_from_pdb,
    clean_protein_labels,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--complex", required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--required-features",
        default=(
            "provenance/"
            "dist_features_below_10_ang_list_new.npy"
        ),
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Computing original distance features...")

    features = compute_pp_dists_from_pdb(
        pdb_path=args.complex,
        ligands=("GTP", "LIG"),
        radius=25.0,
        molid=args.molecule,
    )

    features = clean_protein_labels(features)

    # Check for duplicate names after removal of chain labels.
    duplicate_columns = features.columns[
        features.columns.duplicated()
    ].tolist()

    print("Raw generated columns:", len(features.columns))
    print(
        "Duplicate columns after cleaning:",
        len(duplicate_columns),
    )

    if duplicate_columns:
        print("First duplicate names:")
        print("\n".join(duplicate_columns[:20]))

        raise RuntimeError(
            "Duplicate feature names were created after "
            "cleaning protein labels."
        )

    required = [
        str(value)
        for value in np.load(
            args.required_features,
            allow_pickle=True,
        )
    ]

    missing = [
        name
        for name in required
        if name not in features.columns
    ]

    print("Required model features:", len(required))
    print("Missing required features:", len(missing))

    if missing:
        missing_path = output_path.with_suffix(
            ".missing_features.txt"
        )

        missing_path.write_text(
            "\n".join(missing) + "\n"
        )

        print("Missing-feature list:", missing_path)
        print("\nFirst 30 missing:")
        print("\n".join(missing[:30]))

        raise RuntimeError(
            f"{len(missing)} model features are missing."
        )

    # Retain molecule ID plus exact training feature order.
    ordered = features.loc[
        :,
        ["Molecule"] + required,
    ].copy()

    numeric = ordered.loc[:, required].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric.isna().any(axis=None):
        raise RuntimeError(
            "NA/non-numeric values found in required features."
        )

    ordered.loc[:, required] = numeric

    ordered.to_parquet(
        output_path,
        index=False,
    )

    print("Written:", output_path)
    print("Rows:", len(ordered))
    print("Feature columns:", len(required))
    print(
        "Minimum distance:",
        float(numeric.min(axis=1).iat[0]),
    )
    print(
        "Maximum distance:",
        float(numeric.max(axis=1).iat[0]),
    )


if __name__ == "__main__":
    main()
