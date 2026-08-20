"""
Build the predefined batch0100 distance-feature sets for nested CV.

This script matches the distance-feature list used by the original RF
workflow against the batch0100 master feature table produced by script 01.

The available model-listed features are classified into three structural
categories:

- residue_ligand:   PRO*-LIG
- residue_cofactor: PRO*-GTP
- residue_residue:  PRO*-PRO*

A fourth set, combined_all, is constructed as the union of those three
categories.

Important
---------
This script does not use the low/high labels, effect sizes, p-values or
the all-feature ranking produced by script 02 to select features.

Its purpose is only to define the candidate feature sets used by the
formal nested-cross-validation analysis. Imputation, variance filtering,
scaling and SelectKBest must be fitted inside the cross-validation
pipeline rather than applied here to the complete dataset.

The batch0100 labels originate from old RF-predicted barriers. The later
classifier therefore characterises the behaviour of that model rather
than measured or improved QM/MM activation barriers.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BATCH = "batch0100"

FORMAL_FEATURE_TYPES = (
    "residue_ligand",
    "residue_cofactor",
    "residue_residue",
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MERGED_FILE = (
    BASE_DIR
    / "results"
    / BATCH
    / "tables"
    / "features_with_barriers.parquet"
)

MODEL_FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "shared"
    / "model_features"
    / "dist_features_below_10_ang_list_new.npy"
)

TABLE_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "tables"
)

MANIFEST_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "feature_sets"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Outputs
# ============================================================

OUT_RESIDUE_LIGAND = (
    MANIFEST_DIR
    / "residue_ligand_features.txt"
)

OUT_RESIDUE_COFACTOR = (
    MANIFEST_DIR
    / "residue_cofactor_features.txt"
)

OUT_RESIDUE_RESIDUE = (
    MANIFEST_DIR
    / "residue_residue_features.txt"
)

OUT_COMBINED_ALL = (
    MANIFEST_DIR
    / "combined_all_features.txt"
)

OUT_COFACTOR_LIGAND = (
    MANIFEST_DIR
    / "cofactor_ligand_features_descriptive_only.txt"
)

OUT_OTHER = (
    MANIFEST_DIR
    / "other_model_features.txt"
)

OUT_AVAILABLE = (
    MANIFEST_DIR
    / "model_features_available_in_batch0100.txt"
)

OUT_MISSING = (
    MANIFEST_DIR
    / "model_features_missing_from_batch0100.txt"
)

OUT_DUPLICATES = (
    MANIFEST_DIR
    / "duplicate_model_feature_names.txt"
)

OUT_FEATURE_MANIFEST = (
    TABLE_DIR
    / "03_model_feature_manifest.tsv"
)

OUT_FEATURE_SET_SUMMARY = (
    TABLE_DIR
    / "03_feature_set_summary.tsv"
)

OUT_RUN_SUMMARY = (
    TABLE_DIR
    / "03_build_model_feature_sets_summary.tsv"
)


# ============================================================
# Input validation
# ============================================================

def validate_input_paths() -> None:
    """Confirm that the required input files exist."""
    if not MERGED_FILE.exists():
        raise FileNotFoundError(
            "Master feature table not found:\n"
            f"{MERGED_FILE}"
        )

    if not MODEL_FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Original RF distance-feature list not found:\n"
            f"{MODEL_FEATURE_FILE}"
        )


def validate_master_table(
    table: pd.DataFrame,
) -> None:
    """Validate the basic structure of the master table."""
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
            "Master table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if table["ligand_id"].duplicated().any():
        duplicated_ids = (
            table.loc[
                table["ligand_id"].duplicated(
                    keep=False
                ),
                "ligand_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicated ligand IDs were found:\n"
            + "\n".join(duplicated_ids)
        )


# ============================================================
# Feature handling
# ============================================================

def load_model_features(
    feature_file: Path,
) -> tuple[list[str], list[str]]:
    """
    Load the old RF distance-feature list.

    Returns
    -------
    unique_features:
        Feature names with duplicates removed while preserving order.

    duplicate_features:
        Feature names that occurred more than once in the original file.
    """
    raw_features = np.load(
        feature_file,
        allow_pickle=True,
    ).tolist()

    feature_names = [
        str(feature)
        for feature in raw_features
    ]

    seen = set()
    unique_features = []
    duplicate_features = []

    for feature in feature_names:
        if feature in seen:
            duplicate_features.append(
                feature
            )
            continue

        seen.add(feature)
        unique_features.append(
            feature
        )

    duplicate_features = list(
        dict.fromkeys(
            duplicate_features
        )
    )

    return (
        unique_features,
        duplicate_features,
    )


def classify_feature_type(
    feature_name: str,
) -> str:
    """
    Classify a distance feature from its name.

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

    if has_cofactor and (
        first_is_residue
        or second_is_residue
    ):
        return "residue_cofactor"

    if first_is_residue and second_is_residue:
        return "residue_residue"

    return "other"


def write_feature_list(
    features: list[str],
    output_file: Path,
) -> None:
    """Write one feature name per line."""
    output_file.write_text(
        (
            "\n".join(features)
            + ("\n" if features else "")
        ),
        encoding="utf-8",
    )


def build_feature_manifest(
    model_features: list[str],
    master_columns: set[str],
) -> pd.DataFrame:
    """Build a row-level manifest for every unique model feature."""
    rows = []

    for original_index, feature in enumerate(
        model_features,
        start=1,
    ):
        feature_type = classify_feature_type(
            feature
        )

        available = (
            feature in master_columns
        )

        included_in_formal_sets = (
            available
            and feature_type
            in FORMAL_FEATURE_TYPES
        )

        rows.append(
            {
                "original_model_order": (
                    original_index
                ),
                "feature": feature,
                "feature_type": (
                    feature_type
                ),
                "available_in_batch0100": (
                    available
                ),
                "included_in_formal_sets": (
                    included_in_formal_sets
                ),
                "included_in_combined_all": (
                    included_in_formal_sets
                ),
                "analysis_role": (
                    "formal_classifier_candidate"
                    if included_in_formal_sets
                    else (
                        "descriptive_only"
                        if (
                            available
                            and feature_type
                            == "cofactor_ligand"
                        )
                        else "excluded"
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_formal_sets(
    feature_sets: dict[str, list[str]],
) -> None:
    """Check uniqueness, overlap and combined-set integrity."""
    for set_name, features in (
        feature_sets.items()
    ):
        if len(features) != len(
            set(features)
        ):
            raise ValueError(
                f"Duplicated features found "
                f"in set: {set_name}"
            )

    individual_sets = [
        set(
            feature_sets[
                "residue_ligand"
            ]
        ),
        set(
            feature_sets[
                "residue_cofactor"
            ]
        ),
        set(
            feature_sets[
                "residue_residue"
            ]
        ),
    ]

    for first_index in range(
        len(individual_sets)
    ):
        for second_index in range(
            first_index + 1,
            len(individual_sets),
        ):
            overlap = (
                individual_sets[first_index]
                & individual_sets[second_index]
            )

            if overlap:
                raise ValueError(
                    "Formal structural feature "
                    "sets overlap unexpectedly:\n"
                    + "\n".join(
                        sorted(overlap)
                    )
                )

    expected_combined = set().union(
        *individual_sets
    )

    observed_combined = set(
        feature_sets[
            "combined_all"
        ]
    )

    if expected_combined != observed_combined:
        raise ValueError(
            "combined_all does not exactly equal "
            "the union of the three structural sets."
        )


# ============================================================
# Summaries
# ============================================================

def build_feature_set_summary(
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarise formal feature-set sizes."""
    rows = []

    for set_name, features in (
        feature_sets.items()
    ):
        rows.append(
            {
                "feature_set": (
                    set_name
                ),
                "n_features": (
                    len(features)
                ),
                "definition": {
                    "residue_ligand": (
                        "available model-listed "
                        "PRO*-LIG distances"
                    ),
                    "residue_cofactor": (
                        "available model-listed "
                        "PRO*-GTP distances"
                    ),
                    "residue_residue": (
                        "available model-listed "
                        "PRO*-PRO* distances"
                    ),
                    "combined_all": (
                        "union of residue_ligand, "
                        "residue_cofactor and "
                        "residue_residue"
                    ),
                }[set_name],
                "formal_nested_cv_set": True,
            }
        )

    return pd.DataFrame(
        rows
    )


def save_run_summary(
    table: pd.DataFrame,
    model_features: list[str],
    duplicate_features: list[str],
    manifest: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    """Save key provenance and QC information."""
    settings = {
        "batch": BATCH,
        "master_table": str(
            MERGED_FILE
        ),
        "model_feature_file": str(
            MODEL_FEATURE_FILE
        ),
        "n_master_rows": len(
            table
        ),
        "n_master_feature_columns": (
            table.shape[1] - 3
        ),
        "n_unique_model_features": (
            len(model_features)
        ),
        "n_duplicate_model_feature_names": (
            len(duplicate_features)
        ),
        "n_model_features_available": int(
            manifest[
                "available_in_batch0100"
            ].sum()
        ),
        "n_model_features_missing": int(
            (
                ~manifest[
                    "available_in_batch0100"
                ]
            ).sum()
        ),
        "n_residue_ligand": len(
            feature_sets[
                "residue_ligand"
            ]
        ),
        "n_residue_cofactor": len(
            feature_sets[
                "residue_cofactor"
            ]
        ),
        "n_residue_residue": len(
            feature_sets[
                "residue_residue"
            ]
        ),
        "n_combined_all": len(
            feature_sets[
                "combined_all"
            ]
        ),
        "selection_uses_labels": False,
        "selection_uses_script_02_ranking": False,
        "global_preprocessing_applied": False,
        "formal_feature_sets": (
            "residue_ligand; "
            "residue_cofactor; "
            "residue_residue; "
            "combined_all"
        ),
        "cofactor_ligand_role": (
            "descriptive only; excluded "
            "from formal four-set comparison"
        ),
        "intended_use": (
            "candidate feature manifests "
            "for formal nested CV"
        ),
        "barrier_source": (
            "old RF-predicted barriers; "
            "not used by this script "
            "for feature selection"
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
# Main
# ============================================================

def main() -> None:
    """Build and save the four formal distance-feature sets."""
    print(
        f"Building predefined feature sets "
        f"for {BATCH}..."
    )

    validate_input_paths()

    print(
        "\nReading master feature table..."
    )

    table = pd.read_parquet(
        MERGED_FILE
    )

    validate_master_table(
        table
    )

    print(
        "Master table shape:",
        table.shape,
    )

    print(
        "\nReading original RF "
        "distance-feature list..."
    )

    (
        model_features,
        duplicate_features,
    ) = load_model_features(
        MODEL_FEATURE_FILE
    )

    print(
        "Unique model-listed features:",
        len(model_features),
    )

    print(
        "Duplicated names in source list:",
        len(duplicate_features),
    )

    master_columns = set(
        table.columns
    )

    manifest = build_feature_manifest(
        model_features=model_features,
        master_columns=master_columns,
    )

    manifest.to_csv(
        OUT_FEATURE_MANIFEST,
        sep="\t",
        index=False,
    )

    available_features = (
        manifest.loc[
            manifest[
                "available_in_batch0100"
            ],
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    missing_features = (
        manifest.loc[
            ~manifest[
                "available_in_batch0100"
            ],
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    residue_ligand = (
        manifest.loc[
            (
                manifest[
                    "available_in_batch0100"
                ]
            )
            & (
                manifest["feature_type"]
                == "residue_ligand"
            ),
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    residue_cofactor = (
        manifest.loc[
            (
                manifest[
                    "available_in_batch0100"
                ]
            )
            & (
                manifest["feature_type"]
                == "residue_cofactor"
            ),
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    residue_residue = (
        manifest.loc[
            (
                manifest[
                    "available_in_batch0100"
                ]
            )
            & (
                manifest["feature_type"]
                == "residue_residue"
            ),
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    cofactor_ligand = (
        manifest.loc[
            (
                manifest[
                    "available_in_batch0100"
                ]
            )
            & (
                manifest["feature_type"]
                == "cofactor_ligand"
            ),
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    other_features = (
        manifest.loc[
            (
                manifest[
                    "available_in_batch0100"
                ]
            )
            & (
                manifest["feature_type"]
                == "other"
            ),
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    combined_all = (
        residue_ligand
        + residue_cofactor
        + residue_residue
    )

    feature_sets = {
        "residue_ligand": (
            residue_ligand
        ),
        "residue_cofactor": (
            residue_cofactor
        ),
        "residue_residue": (
            residue_residue
        ),
        "combined_all": (
            combined_all
        ),
    }

    validate_formal_sets(
        feature_sets
    )

    write_feature_list(
        residue_ligand,
        OUT_RESIDUE_LIGAND,
    )

    write_feature_list(
        residue_cofactor,
        OUT_RESIDUE_COFACTOR,
    )

    write_feature_list(
        residue_residue,
        OUT_RESIDUE_RESIDUE,
    )

    write_feature_list(
        combined_all,
        OUT_COMBINED_ALL,
    )

    write_feature_list(
        cofactor_ligand,
        OUT_COFACTOR_LIGAND,
    )

    write_feature_list(
        other_features,
        OUT_OTHER,
    )

    write_feature_list(
        available_features,
        OUT_AVAILABLE,
    )

    write_feature_list(
        missing_features,
        OUT_MISSING,
    )

    write_feature_list(
        duplicate_features,
        OUT_DUPLICATES,
    )

    feature_set_summary = (
        build_feature_set_summary(
            feature_sets
        )
    )

    feature_set_summary.to_csv(
        OUT_FEATURE_SET_SUMMARY,
        sep="\t",
        index=False,
    )

    save_run_summary(
        table=table,
        model_features=model_features,
        duplicate_features=duplicate_features,
        manifest=manifest,
        feature_sets=feature_sets,
    )

    print(
        "\nFormal feature-set summary:"
    )

    print(
        feature_set_summary
        .to_string(index=False)
    )

    print(
        "\nAdditional categories:"
    )

    print(
        "cofactor_ligand "
        "(descriptive only):",
        len(cofactor_ligand),
    )

    print(
        "other available features:",
        len(other_features),
    )

    print(
        "missing model-listed features:",
        len(missing_features),
    )

    print(
        "\nSaved manifests to:"
    )

    print(MANIFEST_DIR)

    print(
        "\nSaved tabular summaries:"
    )

    print(OUT_FEATURE_MANIFEST)
    print(OUT_FEATURE_SET_SUMMARY)
    print(OUT_RUN_SUMMARY)

    print("\nDone.")


if __name__ == "__main__":
    main()