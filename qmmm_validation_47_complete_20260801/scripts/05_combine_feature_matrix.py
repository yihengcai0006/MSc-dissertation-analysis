#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features" / "per_ligand"
BARRIERS_FILE = ROOT / "qmmm_validation" / "barriers.csv"
REQUIRED_FILE = (
    ROOT / "model"
    / "dist_features_below_10_ang_list_new.npy"
)

OUT_PARQUET = ROOT / "features" / "feature_matrix_47.parquet"
OUT_PREVIEW = ROOT / "features" / "feature_matrix_47_preview.tsv"
OUT_SUMMARY = (
    ROOT / "results" / "statistics"
    / "feature_matrix_summary.txt"
)


def main() -> None:
    required = [
        str(value)
        for value in np.load(
            REQUIRED_FILE,
            allow_pickle=True,
        )
    ]

    barriers = pd.read_csv(BARRIERS_FILE)
    ligand_ids = (
        barriers["ligand_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    frames = []

    for ligand_id in ligand_ids:
        feature_file = FEATURE_DIR / f"{ligand_id}.parquet"

        if not feature_file.is_file():
            raise FileNotFoundError(feature_file)

        frame = pd.read_parquet(feature_file)

        if len(frame) != 1:
            raise ValueError(
                f"{feature_file.name}: expected one row, "
                f"found {len(frame)}"
            )

        missing = [
            name
            for name in required
            if name not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{feature_file.name}: "
                f"{len(missing)} required features missing"
            )

        frames.append(
            frame.loc[:, ["Molecule"] + required]
        )

    matrix = pd.concat(frames, ignore_index=True)
    numeric = matrix.loc[:, required].astype(float)

    if matrix["Molecule"].duplicated().any():
        raise ValueError("Duplicate molecule IDs detected")

    if matrix["Molecule"].tolist() != ligand_ids:
        raise ValueError(
            "Feature matrix order differs from barriers.csv"
        )

    if numeric.isna().any(axis=None):
        raise ValueError("NaN values detected")

    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite values detected")

    matrix.to_parquet(OUT_PARQUET, index=False)

    preview_columns = ["Molecule"] + required[:15]
    matrix.loc[:, preview_columns].to_csv(
        OUT_PREVIEW,
        sep="\t",
        index=False,
    )

    lines = [
        "Combined feature matrix summary",
        "===============================",
        f"Ligands: {len(matrix)}",
        f"Required feature columns: {len(required)}",
        f"Total matrix columns: {len(matrix.columns)}",
        (
            "Duplicate molecule IDs: "
            f"{int(matrix['Molecule'].duplicated().sum())}"
        ),
        f"NaN values: {int(numeric.isna().sum().sum())}",
        (
            "Infinite values: "
            f"{int(np.isinf(numeric.to_numpy()).sum())}"
        ),
        (
            "Overall minimum distance: "
            f"{numeric.min().min():.3f}"
        ),
        (
            "Overall maximum distance: "
            f"{numeric.max().max():.3f}"
        ),
        "Feature order matches model list: YES",
        "Ligand order matches barriers.csv: YES",
    ]

    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nWritten:", OUT_PARQUET)
    print("Written:", OUT_PREVIEW)
    print("Written:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
