#!/usr/bin/env python3

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    ROOT / "model"
    / "rf_model_dist_below_10_new.pkl"
)

FEATURE_LIST_FILE = (
    ROOT / "model"
    / "dist_features_below_10_ang_list_new.npy"
)

FEATURE_MATRIX_FILE = (
    ROOT / "features"
    / "feature_matrix_47.parquet"
)

OUT_ALL = (
    ROOT / "results"
    / "tables"
    / "feature_importance_variance_all.tsv"
)

OUT_TOP = (
    ROOT / "results"
    / "tables"
    / "top_100_rf_features_variance.tsv"
)

OUT_TYPES = (
    ROOT / "results"
    / "tables"
    / "rf_importance_by_feature_type.tsv"
)

OUT_SUMMARY = (
    ROOT / "results"
    / "statistics"
    / "feature_importance_variance_summary.txt"
)


def classify_feature(name: str) -> str:
    parts = name.split("-")

    contains_lig = "LIG" in parts
    contains_gtp = "GTP" in parts
    protein_count = sum(
        part.startswith("PRO")
        for part in parts
    )

    if protein_count == 2:
        return "protein-protein"

    if contains_lig and protein_count == 1:
        return "protein-LIG"

    if contains_gtp and protein_count == 1:
        return "protein-GTP"

    if contains_lig and contains_gtp:
        return "GTP-LIG"

    return "other"


def main() -> None:
    feature_names = [
        str(value)
        for value in np.load(
            FEATURE_LIST_FILE,
            allow_pickle=True,
        )
    ]

    matrix = pd.read_parquet(FEATURE_MATRIX_FILE)
    X = matrix.loc[:, feature_names].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            InconsistentVersionWarning,
        )
        model = joblib.load(MODEL_FILE)

    importances = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature importance length does not match "
            "feature-list length"
        )

    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    minima = X.min(axis=0)
    maxima = X.max(axis=0)
    ranges = maxima - minima
    unique_counts = X.nunique(axis=0)

    result = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_type": [
                classify_feature(name)
                for name in feature_names
            ],
            "rf_importance": importances,
            "mean_distance_angstrom": [
                means[name]
                for name in feature_names
            ],
            "sd_distance_angstrom": [
                stds[name]
                for name in feature_names
            ],
            "min_distance_angstrom": [
                minima[name]
                for name in feature_names
            ],
            "max_distance_angstrom": [
                maxima[name]
                for name in feature_names
            ],
            "range_distance_angstrom": [
                ranges[name]
                for name in feature_names
            ],
            "n_unique": [
                unique_counts[name]
                for name in feature_names
            ],
        }
    )

    # Features were rounded to 0.001 Å during generation.
    result["is_zero_variance"] = (
        result["sd_distance_angstrom"] == 0
    )

    result["is_effectively_fixed_0_001A"] = (
        result["range_distance_angstrom"] <= 0.001
    )

    result["is_near_zero_variance_0_01A"] = (
        result["sd_distance_angstrom"] < 0.01
    )

    result = result.sort_values(
        "rf_importance",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "importance_rank",
        np.arange(1, len(result) + 1),
    )

    OUT_ALL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_ALL,
        sep="\t",
        index=False,
        float_format="%.10f",
    )

    result.head(100).to_csv(
        OUT_TOP,
        sep="\t",
        index=False,
        float_format="%.10f",
    )

    by_type = (
        result.groupby(
            "feature_type",
            as_index=False,
        )
        .agg(
            feature_count=("feature_name", "size"),
            total_rf_importance=("rf_importance", "sum"),
            zero_variance_feature_count=(
                "is_zero_variance",
                "sum",
            ),
            near_zero_feature_count=(
                "is_near_zero_variance_0_01A",
                "sum",
            ),
        )
    )

    by_type["importance_percentage"] = (
        by_type["total_rf_importance"] * 100
    )

    by_type = by_type.sort_values(
        "total_rf_importance",
        ascending=False,
    )

    by_type.to_csv(
        OUT_TYPES,
        sep="\t",
        index=False,
        float_format="%.10f",
    )

    zero_importance = result.loc[
        result["is_zero_variance"],
        "rf_importance",
    ].sum()

    effective_fixed_importance = result.loc[
        result["is_effectively_fixed_0_001A"],
        "rf_importance",
    ].sum()

    near_zero_importance = result.loc[
        result["is_near_zero_variance_0_01A"],
        "rf_importance",
    ].sum()

    target_name = "PRO1281-PRO1391"
    target = result.loc[
        result["feature_name"] == target_name
    ]

    lines = [
        "RF feature importance and variability summary",
        "=============================================",
        f"Total model features: {len(result)}",
        f"Total RF importance: {result['rf_importance'].sum():.10f}",
        "",
        "Importance by feature type:",
    ]

    for row in by_type.itertuples(index=False):
        lines.append(
            f"  {row.feature_type}: "
            f"{row.total_rf_importance:.6f} "
            f"({row.importance_percentage:.3f}%), "
            f"n={row.feature_count}"
        )

    lines.extend(
        [
            "",
            "Feature variability:",
            (
                "  Exactly zero-variance features: "
                f"{int(result['is_zero_variance'].sum())}"
            ),
            (
                "  RF importance on exactly zero-variance features: "
                f"{zero_importance:.6f} "
                f"({zero_importance * 100:.3f}%)"
            ),
            (
                "  Effectively fixed features (range <= 0.001 Å): "
                f"{int(result['is_effectively_fixed_0_001A'].sum())}"
            ),
            (
                "  RF importance on effectively fixed features: "
                f"{effective_fixed_importance:.6f} "
                f"({effective_fixed_importance * 100:.3f}%)"
            ),
            (
                "  Near-zero-variance features (SD < 0.01 Å): "
                f"{int(result['is_near_zero_variance_0_01A'].sum())}"
            ),
            (
                "  RF importance on near-zero-variance features: "
                f"{near_zero_importance:.6f} "
                f"({near_zero_importance * 100:.3f}%)"
            ),
            "",
            "Most important feature:",
            (
                f"  Name: "
                f"{result.iloc[0]['feature_name']}"
            ),
            (
                f"  RF importance: "
                f"{result.iloc[0]['rf_importance']:.10f}"
            ),
            (
                f"  SD: "
                f"{result.iloc[0]['sd_distance_angstrom']:.10f} Å"
            ),
            (
                f"  Range: "
                f"{result.iloc[0]['range_distance_angstrom']:.10f} Å"
            ),
            "",
            f"Requested diagnostic feature: {target_name}",
        ]
    )

    if target.empty:
        lines.append("  Feature not found.")
    else:
        row = target.iloc[0]
        lines.extend(
            [
                f"  Importance rank: {int(row['importance_rank'])}",
                f"  RF importance: {row['rf_importance']:.10f}",
                f"  Importance percentage: {row['rf_importance'] * 100:.3f}%",
                f"  Mean distance: {row['mean_distance_angstrom']:.10f} Å",
                f"  SD: {row['sd_distance_angstrom']:.10f} Å",
                f"  Minimum: {row['min_distance_angstrom']:.10f} Å",
                f"  Maximum: {row['max_distance_angstrom']:.10f} Å",
                f"  Range: {row['range_distance_angstrom']:.10f} Å",
                f"  Unique values: {int(row['n_unique'])}",
            ]
        )

    OUT_SUMMARY.write_text(
        "\n".join(lines) + "\n"
    )

    print("\n".join(lines))
    print("\nWritten:", OUT_ALL)
    print("Written:", OUT_TOP)
    print("Written:", OUT_TYPES)
    print("Written:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
