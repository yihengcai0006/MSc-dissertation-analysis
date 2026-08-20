#!/usr/bin/env python3
"""
01_audit_qmmm47_dataset.py

Audit the 47-complete-structure improved-QM/MM dataset used for the
pre-fixed stable-feature exploratory check.

The script verifies that:
1. exactly 47 improved-QM/MM barrier rows are present;
2. exactly 47 complete PDB structures are present;
3. exactly 47 feature-matrix rows are present;
4. ligand IDs match exactly across barriers, PDBs and features;
5. the three pre-fixed batch0100 stable features are present;
6. no target-feature values are missing.

No feature selection, model fitting or correlation analysis is performed here.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BARRIER_FILE = (
    PROJECT_ROOT
    / "qmmm_validation"
    / "barriers.csv"
)

STRUCTURE_DIR = (
    PROJECT_ROOT
    / "qmmm_validation"
    / "complete_structures"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "existing_features"
    / "feature_matrix_47.parquet"
)

OUTPUT_TABLE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "01_qmmm47_dataset_audit.tsv"
)

OUTPUT_LOG = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "01_qmmm47_dataset_audit.log"
)


PREFIXED_FEATURES = [
    "PRO62-PRO1279",
    "PRO1275-PRO1276",
    "PRO90-PRO92",
]


ID_CANDIDATES = [
    "ligand_id",
    "molecule",
    "Molecule",
    "molid",
    "ligand",
]


def find_id_column(df):
    """Return the first recognized ligand-ID column."""
    for column in ID_CANDIDATES:
        if column in df.columns:
            return column

    raise RuntimeError(
        "Could not identify ligand-ID column in feature matrix. "
        f"First columns: {df.columns[:20].tolist()}"
    )


def load_inputs():
    """Load the barrier and feature tables."""
    if not BARRIER_FILE.exists():
        raise FileNotFoundError(BARRIER_FILE)

    if not STRUCTURE_DIR.exists():
        raise FileNotFoundError(STRUCTURE_DIR)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    barriers = pd.read_csv(BARRIER_FILE)
    features = pd.read_parquet(FEATURE_FILE)

    return barriers, features


def get_structure_ids():
    """Return ligand IDs inferred from complete PDB filenames."""
    pdb_files = sorted(STRUCTURE_DIR.glob("*.pdb"))

    structure_ids = {
        path.stem
        for path in pdb_files
    }

    return pdb_files, structure_ids


def main():
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)

    barriers, features = load_inputs()

    feature_id_col = find_id_column(features)

    pdb_files, structure_ids = get_structure_ids()

    barrier_ids = set(
        barriers["ligand_id"]
        .astype(str)
        .str.strip()
    )

    feature_ids = set(
        features[feature_id_col]
        .astype(str)
        .str.strip()
    )

    print("=" * 80)
    print("QMMM47 DATASET AUDIT")
    print("=" * 80)

    print(f"Barrier file:   {BARRIER_FILE}")
    print(f"Structure dir:  {STRUCTURE_DIR}")
    print(f"Feature file:   {FEATURE_FILE}")
    print()

    print("Dataset sizes:")
    print(f"  barrier rows:        {len(barriers)}")
    print(f"  unique barrier IDs:  {len(barrier_ids)}")
    print(f"  PDB files:           {len(pdb_files)}")
    print(f"  unique PDB IDs:      {len(structure_ids)}")
    print(f"  feature rows:        {len(features)}")
    print(f"  unique feature IDs:  {len(feature_ids)}")
    print()

    if len(barriers) != 47:
        raise RuntimeError(
            f"Expected 47 barrier rows, found {len(barriers)}"
        )

    if len(pdb_files) != 47:
        raise RuntimeError(
            f"Expected 47 PDB files, found {len(pdb_files)}"
        )

    if len(features) != 47:
        raise RuntimeError(
            f"Expected 47 feature rows, found {len(features)}"
        )

    if barriers["ligand_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate ligand IDs found in barriers.csv"
        )

    if features[feature_id_col].duplicated().any():
        raise RuntimeError(
            "Duplicate ligand IDs found in feature matrix"
        )

    missing_barriers = barriers["barrier_kcal_per_mol"].isna().sum()

    if missing_barriers != 0:
        raise RuntimeError(
            f"Missing improved-QM/MM barriers: {missing_barriers}"
        )

    print("ID-set differences:")
    print(
        "  features not in barriers:",
        sorted(feature_ids - barrier_ids),
    )
    print(
        "  barriers not in features:",
        sorted(barrier_ids - feature_ids),
    )
    print(
        "  structures not in barriers:",
        sorted(structure_ids - barrier_ids),
    )
    print(
        "  barriers not in structures:",
        sorted(barrier_ids - structure_ids),
    )
    print()

    if not (
        feature_ids == barrier_ids
        and barrier_ids == structure_ids
        and len(feature_ids) == 47
    ):
        raise RuntimeError(
            "Ligand IDs do not match exactly across the three data sources."
        )

    print(
        "QC PASSED: exact 47/47 ID match across "
        "features, barriers and complete PDBs."
    )
    print()

    print("Pre-fixed features:")

    audit_records = []

    for feature in PREFIXED_FEATURES:
        if feature not in features.columns:
            raise RuntimeError(
                f"Pre-fixed feature missing from feature matrix: {feature}"
            )

        missing = int(features[feature].isna().sum())
        n_unique = int(features[feature].nunique(dropna=True))

        print(
            f"  {feature:20s} "
            f"FOUND  missing={missing}  n_unique={n_unique}"
        )

        if missing != 0:
            raise RuntimeError(
                f"{feature} contains {missing} missing values."
            )

        audit_records.append(
            {
                "feature": feature,
                "n_rows": len(features),
                "n_missing": missing,
                "n_unique": n_unique,
                "feature_present": True,
            }
        )

    audit_df = pd.DataFrame(audit_records)

    audit_df.to_csv(
        OUTPUT_TABLE,
        sep="\t",
        index=False,
    )

    with open(OUTPUT_LOG, "w", encoding="utf-8") as handle:
        handle.write(
            "QMMM47 pre-fixed feature dataset audit\n"
        )
        handle.write("=" * 60 + "\n\n")

        handle.write(f"Barrier rows: {len(barriers)}\n")
        handle.write(f"PDB files: {len(pdb_files)}\n")
        handle.write(f"Feature rows: {len(features)}\n\n")

        handle.write(
            "Exact 47/47 ID match across barriers, "
            "features and PDBs: PASS\n\n"
        )

        handle.write(
            audit_df.to_string(index=False)
        )

        handle.write("\n")

    print()
    print("Outputs:")
    print(OUTPUT_TABLE)
    print(OUTPUT_LOG)
    print()
    print("SCRIPT 01 COMPLETE.")


if __name__ == "__main__":
    main()
