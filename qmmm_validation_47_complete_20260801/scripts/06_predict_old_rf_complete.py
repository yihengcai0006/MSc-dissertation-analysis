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

BARRIERS_FILE = (
    ROOT / "qmmm_validation"
    / "barriers.csv"
)

OUT_PREDICTIONS = (
    ROOT / "predictions"
    / "old_rf_predictions_47_complete.tsv"
)

OUT_COMPARISON = (
    ROOT / "predictions"
    / "old_rf_vs_qmmm_47_complete.csv"
)

OUT_SUMMARY = (
    ROOT / "results"
    / "statistics"
    / "prediction_summary_47_complete.txt"
)


def main() -> None:
    required = [
        str(value)
        for value in np.load(
            FEATURE_LIST_FILE,
            allow_pickle=True,
        )
    ]

    matrix = pd.read_parquet(FEATURE_MATRIX_FILE)
    barriers = pd.read_csv(BARRIERS_FILE)

    if len(matrix) != 47:
        raise ValueError(
            f"Expected 47 feature rows, found {len(matrix)}"
        )

    missing = [
        name
        for name in required
        if name not in matrix.columns
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} required features are missing"
        )

    X = (
        matrix.loc[:, required]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )

    if np.isnan(X).any():
        raise ValueError("NaN values found in model input")

    if np.isinf(X).any():
        raise ValueError("Infinite values found in model input")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            InconsistentVersionWarning,
        )
        model = joblib.load(MODEL_FILE)

    version_warnings = [
        str(item.message)
        for item in caught
        if issubclass(
            item.category,
            InconsistentVersionWarning,
        )
    ]

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_feature_count != len(required):
        raise ValueError(
            "Model feature count does not match feature list: "
            f"{model_feature_count} != {len(required)}"
        )

    predictions = model.predict(X)

    prediction_df = pd.DataFrame(
        {
            "ligand_id": matrix["Molecule"].astype(str),
            "old_rf_prediction_kcal_per_mol": predictions,
        }
    )

    if prediction_df["ligand_id"].duplicated().any():
        raise ValueError("Duplicate ligand IDs in predictions")

    comparison = barriers.merge(
        prediction_df,
        on="ligand_id",
        how="left",
        validate="one_to_one",
    )

    if comparison[
        "old_rf_prediction_kcal_per_mol"
    ].isna().any():
        raise ValueError(
            "Some barriers lack an RF prediction"
        )

    comparison["residual_pred_minus_reference"] = (
        comparison["old_rf_prediction_kcal_per_mol"]
        - comparison["barrier_kcal_per_mol"]
    )

    OUT_PREDICTIONS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df.to_csv(
        OUT_PREDICTIONS,
        sep="\t",
        index=False,
        float_format="%.10f",
    )

    comparison.to_csv(
        OUT_COMPARISON,
        index=False,
        float_format="%.10f",
    )

    pred = comparison[
        "old_rf_prediction_kcal_per_mol"
    ].astype(float)

    ref = comparison[
        "barrier_kcal_per_mol"
    ].astype(float)

    summary_lines = [
        "Old RF prediction summary — 47 complete structures",
        "==================================================",
        f"Ligands: {len(comparison)}",
        f"Required features: {len(required)}",
        f"Model input shape: {X.shape}",
        f"Model estimators: {len(model.estimators_)}",
        f"Version warnings captured: {len(version_warnings)}",
        "",
        "Reference QM/MM barriers:",
        f"  Mean: {ref.mean():.6f}",
        f"  SD: {ref.std(ddof=1):.6f}",
        f"  Minimum: {ref.min():.6f}",
        f"  Maximum: {ref.max():.6f}",
        f"  Range: {(ref.max() - ref.min()):.6f}",
        "",
        "Old RF predictions:",
        f"  Mean: {pred.mean():.6f}",
        f"  SD: {pred.std(ddof=1):.6f}",
        f"  Minimum: {pred.min():.6f}",
        f"  Maximum: {pred.max():.6f}",
        f"  Range: {(pred.max() - pred.min()):.6f}",
        f"  Unique values (full precision): {pred.nunique()}",
        f"  Unique values (3 d.p.): {pred.round(3).nunique()}",
        f"  Unique values (2 d.p.): {pred.round(2).nunique()}",
        "",
        "Prediction compression:",
        (
            "  Prediction SD / reference SD: "
            f"{pred.std(ddof=1) / ref.std(ddof=1):.6f}"
        ),
        (
            "  Prediction range / reference range: "
            f"{(pred.max() - pred.min()) / (ref.max() - ref.min()):.6f}"
        ),
        "",
        "Model loading warnings:",
        *(
            [f"  {message}" for message in version_warnings]
            if version_warnings
            else ["  None"]
        ),
    ]

    OUT_SUMMARY.write_text(
        "\n".join(summary_lines) + "\n"
    )

    print("\n".join(summary_lines))
    print("\nWritten:", OUT_PREDICTIONS)
    print("Written:", OUT_COMPARISON)
    print("Written:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
