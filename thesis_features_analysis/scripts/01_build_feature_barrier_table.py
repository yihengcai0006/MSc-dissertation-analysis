"""
Build the batch0100 feature–barrier master table.

The script reads one structural-feature parquet file per ligand, parses
the old RF-predicted barriers, merges the two sources by ligand ID, and
assigns the predefined low, middle and high barrier groups.

Important
---------
The barrier values are predictions from the original RF workflow. They
are used to characterise the behaviour of that model and are not
measured or improved QM/MM activation barriers.

Group definitions
-----------------
low:
    barrier < 28 kcal/mol

middle:
    28 <= barrier < 29 kcal/mol

high:
    barrier >= 29 kcal/mol

Different ligand parquet files may contain different feature columns.
Pandas concatenates their union, and missing ligand–feature combinations
are represented as NaN. Missingness is assessed later by scripts 02 and
03 rather than during this assembly step.
"""

import re
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BATCH = "batch0100"
EXPECTED_FEATURE_FILE_COUNT = 100

LOW_THRESHOLD = 28.0
HIGH_THRESHOLD = 29.0


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / BATCH
    / "features"
)

SUMMARY_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / BATCH
    / "barriers"
    / "ligand_summaries_20260223_2245.txt"
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

OUT_BARRIERS = (
    TABLE_DIR
    / "matched_barriers.tsv"
)

OUT_FEATURES = (
    TABLE_DIR
    / "feature_table.parquet"
)

OUT_MERGED = (
    TABLE_DIR
    / "features_with_barriers.parquet"
)

OUT_PREVIEW = (
    TABLE_DIR
    / "features_with_barriers_preview.tsv"
)

OUT_FEATURE_ONLY_IDS = (
    TABLE_DIR
    / "feature_ids_without_barriers.txt"
)

OUT_BARRIER_ONLY_IDS = (
    TABLE_DIR
    / "barrier_ids_without_features.txt"
)

OUT_RUN_SUMMARY = (
    TABLE_DIR
    / "01_build_feature_barrier_table_summary.tsv"
)


# ============================================================
# Input handling
# ============================================================

def validate_input_paths() -> None:
    """Confirm that required input files and directories exist."""
    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            "Feature directory not found:\n"
            f"{FEATURE_DIR}"
        )

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            "Barrier summary file not found:\n"
            f"{SUMMARY_FILE}"
        )


def parse_summary(
    summary_file: Path,
) -> pd.DataFrame:
    """
    Parse ligand IDs and predicted barriers.

    Expected lines resemble:

        PV-008757770384: 26.218 kcal/mol
    """
    rows = []

    pattern = re.compile(
        r"(PV-\d+):\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)\s*"
        r"kcal/mol"
    )

    with summary_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            match = pattern.search(line)

            if match is None:
                continue

            rows.append(
                {
                    "ligand_id": match.group(1),
                    "barrier": float(
                        match.group(2)
                    ),
                }
            )

    barriers = pd.DataFrame(rows)

    if barriers.empty:
        raise ValueError(
            "No barriers were parsed from:\n"
            f"{summary_file}"
        )

    duplicated_ids = barriers.loc[
        barriers["ligand_id"].duplicated(
            keep=False
        ),
        "ligand_id",
    ]

    if not duplicated_ids.empty:
        raise ValueError(
            "Duplicated ligand IDs in barrier summary:\n"
            + "\n".join(
                duplicated_ids.astype(str)
            )
        )

    return barriers


def read_feature_files(
    feature_dir: Path,
) -> pd.DataFrame:
    """
    Read one-row parquet files and concatenate their column union.

    The filename stem is used as the authoritative ligand ID.
    """
    files = sorted(
        feature_dir.glob("*.parquet")
    )

    print(
        f"Found {len(files)} parquet "
        "feature files"
    )

    if len(files) != EXPECTED_FEATURE_FILE_COUNT:
        raise ValueError(
            "Unexpected parquet count: "
            f"expected {EXPECTED_FEATURE_FILE_COUNT}, "
            f"found {len(files)}"
        )

    filename_ids = [
        feature_file.stem
        for feature_file in files
    ]

    if len(filename_ids) != len(
        set(filename_ids)
    ):
        raise ValueError(
            "Duplicate ligand IDs were derived "
            "from parquet filenames."
        )

    feature_frames = []

    for file_number, feature_file in enumerate(
        files,
        start=1,
    ):
        ligand_id = feature_file.stem

        feature_row = pd.read_parquet(
            feature_file
        )

        if feature_row.shape[0] != 1:
            raise ValueError(
                f"{feature_file.name} contains "
                f"{feature_row.shape[0]} rows; "
                "one row was expected."
            )

        if "Molecule" in feature_row.columns:
            feature_row = feature_row.rename(
                columns={
                    "Molecule": "ligand_id",
                }
            )

        elif "ligand_id" not in feature_row.columns:
            feature_row.insert(
                0,
                "ligand_id",
                ligand_id,
            )

        # Use the filename as the authoritative identifier.
        feature_row["ligand_id"] = ligand_id

        feature_frames.append(
            feature_row
        )

        if (
            file_number % 10 == 0
            or file_number == len(files)
        ):
            print(
                f"Read {file_number}/"
                f"{len(files)} files"
            )

    print(
        "\nConcatenating all feature rows "
        "using the union of columns..."
    )

    feature_table = pd.concat(
        feature_frames,
        ignore_index=True,
        sort=False,
    )

    if feature_table[
        "ligand_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate ligand IDs in assembled "
            "feature table."
        )

    return feature_table


# ============================================================
# Dataset construction
# ============================================================

def assign_group(
    barrier: float,
) -> str:
    """Assign a predefined old-model barrier group."""
    if barrier < LOW_THRESHOLD:
        return "low"

    if barrier < HIGH_THRESHOLD:
        return "middle"

    return "high"


def save_id_list(
    identifiers: list[str],
    output_file: Path,
) -> None:
    """Write one identifier per line."""
    output_file.write_text(
        (
            "\n".join(identifiers)
            + ("\n" if identifiers else "")
        ),
        encoding="utf-8",
    )


def save_run_summary(
    features: pd.DataFrame,
    barriers: pd.DataFrame,
    merged: pd.DataFrame,
    feature_only_ids: list[str],
    barrier_only_ids: list[str],
) -> None:
    """Save key counts and definitions for provenance."""
    group_counts = (
        merged["group"]
        .value_counts()
        .to_dict()
    )

    settings = {
        "batch": BATCH,
        "n_feature_files": len(features),
        "n_parsed_barriers": len(barriers),
        "n_matched_ligands": len(merged),
        "n_feature_columns": (
            features.shape[1] - 1
        ),
        "n_low": int(
            group_counts.get("low", 0)
        ),
        "n_middle": int(
            group_counts.get("middle", 0)
        ),
        "n_high": int(
            group_counts.get("high", 0)
        ),
        "low_definition": (
            f"barrier < {LOW_THRESHOLD:g}"
        ),
        "middle_definition": (
            f"{LOW_THRESHOLD:g} <= barrier "
            f"< {HIGH_THRESHOLD:g}"
        ),
        "high_definition": (
            f"barrier >= {HIGH_THRESHOLD:g}"
        ),
        "n_feature_ids_without_barriers": (
            len(feature_only_ids)
        ),
        "n_barrier_ids_without_features": (
            len(barrier_only_ids)
        ),
        "barrier_source": (
            "old RF-predicted barriers"
        ),
        "feature_schema_handling": (
            "union of parquet columns; "
            "missing values represented by NaN"
        ),
    }

    summary = pd.DataFrame(
        [
            {
                "setting": key,
                "value": value,
            }
            for key, value in settings.items()
        ]
    )

    summary.to_csv(
        OUT_RUN_SUMMARY,
        sep="\t",
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Build and save the batch0100 master analysis table."""
    print(
        f"Building feature–barrier table "
        f"for {BATCH}..."
    )

    validate_input_paths()

    print("\nParsing predicted barriers...")

    barriers = parse_summary(
        SUMMARY_FILE
    )

    print(
        "Parsed barriers:",
        len(barriers),
    )

    print(
        barriers["barrier"]
        .describe()
        .to_string()
    )

    barriers.to_csv(
        OUT_BARRIERS,
        sep="\t",
        index=False,
    )

    print("\nReading feature parquet files...")

    features = read_feature_files(
        FEATURE_DIR
    )

    print(
        "Feature table shape:",
        features.shape,
    )

    features.to_parquet(
        OUT_FEATURES,
        index=False,
    )

    feature_ids = set(
        features["ligand_id"]
    )

    barrier_ids = set(
        barriers["ligand_id"]
    )

    feature_only_ids = sorted(
        feature_ids - barrier_ids
    )

    barrier_only_ids = sorted(
        barrier_ids - feature_ids
    )

    save_id_list(
        feature_only_ids,
        OUT_FEATURE_ONLY_IDS,
    )

    save_id_list(
        barrier_only_ids,
        OUT_BARRIER_ONLY_IDS,
    )

    print(
        "\nFeature IDs without barriers:",
        len(feature_only_ids),
    )

    print(
        "Barrier IDs without features:",
        len(barrier_only_ids),
    )

    print("\nMerging features with barriers...")

    merged = features.merge(
        barriers,
        on="ligand_id",
        how="inner",
        validate="one_to_one",
    )

    merged["group"] = (
        merged["barrier"]
        .apply(assign_group)
    )

    first_columns = [
        "ligand_id",
        "barrier",
        "group",
    ]

    other_columns = [
        column
        for column in merged.columns
        if column not in first_columns
    ]

    merged = merged[
        first_columns + other_columns
    ]

    print(
        "Merged table shape:",
        merged.shape,
    )

    print("\nGroup counts:")

    print(
        merged["group"]
        .value_counts()
        .to_string()
    )

    print("\nBarrier summary by group:")

    print(
        merged
        .groupby("group")["barrier"]
        .describe()
        .to_string()
    )

    merged.to_parquet(
        OUT_MERGED,
        index=False,
    )

    merged.iloc[
        :,
        :min(30, merged.shape[1]),
    ].to_csv(
        OUT_PREVIEW,
        sep="\t",
        index=False,
    )

    save_run_summary(
        features=features,
        barriers=barriers,
        merged=merged,
        feature_only_ids=feature_only_ids,
        barrier_only_ids=barrier_only_ids,
    )

    print("\nSaved:")
    print(OUT_BARRIERS)
    print(OUT_FEATURES)
    print(OUT_MERGED)
    print(OUT_PREVIEW)
    print(OUT_FEATURE_ONLY_IDS)
    print(OUT_BARRIER_ONLY_IDS)
    print(OUT_RUN_SUMMARY)

    print("\nDone.")


if __name__ == "__main__":
    main()
