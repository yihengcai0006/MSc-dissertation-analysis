#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "predictions"
    / "old_rf_vs_qmmm_47_complete.csv"
)

OUT_SUMMARY = (
    ROOT / "results"
    / "statistics"
    / "statistics_summary_47_complete.txt"
)

OUT_BOOTSTRAP = (
    ROOT / "results"
    / "tables"
    / "bootstrap_correlations_47_complete.tsv"
)

OUT_RESIDUALS = (
    ROOT / "results"
    / "tables"
    / "prediction_residuals_47_complete.tsv"
)

BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 20260801


def bootstrap_correlations(
    reference: np.ndarray,
    prediction: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(reference)

    rows = []

    for iteration in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

        ref_sample = reference[indices]
        pred_sample = prediction[indices]

        # Some bootstrap samples can become constant.
        if (
            np.std(ref_sample) == 0
            or np.std(pred_sample) == 0
        ):
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = stats.pearsonr(
                ref_sample,
                pred_sample,
            ).statistic

            spearman = stats.spearmanr(
                ref_sample,
                pred_sample,
            ).statistic

        rows.append(
            {
                "bootstrap_iteration": iteration + 1,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )

    return pd.DataFrame(rows)


def percentile_ci(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)

    return (
        float(np.percentile(clean, 2.5)),
        float(np.percentile(clean, 97.5)),
    )


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "ligand_id",
        "barrier_kcal_per_mol",
        "old_rf_prediction_kcal_per_mol",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    reference = df[
        "barrier_kcal_per_mol"
    ].to_numpy(dtype=float)

    prediction = df[
        "old_rf_prediction_kcal_per_mol"
    ].to_numpy(dtype=float)

    if len(df) != 47:
        raise ValueError(
            f"Expected 47 rows, found {len(df)}"
        )

    pearson_result = stats.pearsonr(
        reference,
        prediction,
    )

    spearman_result = stats.spearmanr(
        reference,
        prediction,
    )

    kendall_result = stats.kendalltau(
        reference,
        prediction,
    )

    slope, intercept, regression_r, regression_p, slope_stderr = (
        stats.linregress(reference, prediction)
    )

    residual = prediction - reference
    absolute_error = np.abs(residual)

    mae = mean_absolute_error(reference, prediction)
    rmse = mean_squared_error(
        reference,
        prediction,
    ) ** 0.5

    mean_offset = float(np.mean(residual))
    median_offset = float(np.median(residual))

    bootstrap = bootstrap_correlations(
        reference,
        prediction,
        BOOTSTRAP_SAMPLES,
        RANDOM_SEED,
    )

    pearson_ci = percentile_ci(
        bootstrap["pearson_r"]
    )

    spearman_ci = percentile_ci(
        bootstrap["spearman_rho"]
    )

    OUT_BOOTSTRAP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bootstrap.to_csv(
        OUT_BOOTSTRAP,
        sep="\t",
        index=False,
        float_format="%.10f",
    )

    residual_df = df.copy()
    residual_df["residual_pred_minus_reference"] = residual
    residual_df["absolute_error"] = absolute_error

    residual_df.to_csv(
        OUT_RESIDUALS,
        sep="\t",
        index=False,
        float_format="%.10f",
    )

    lines = [
        "Old RF versus improved QM/MM barriers",
        "47 complete structures — statistical summary",
        "================================================",
        f"n: {len(df)}",
        "",
        "Correlation:",
        f"  Pearson r: {pearson_result.statistic:.6f}",
        f"  Pearson p-value: {pearson_result.pvalue:.6g}",
        (
            "  Pearson bootstrap 95% CI: "
            f"[{pearson_ci[0]:.6f}, {pearson_ci[1]:.6f}]"
        ),
        f"  Spearman rho: {spearman_result.statistic:.6f}",
        f"  Spearman p-value: {spearman_result.pvalue:.6g}",
        (
            "  Spearman bootstrap 95% CI: "
            f"[{spearman_ci[0]:.6f}, {spearman_ci[1]:.6f}]"
        ),
        f"  Kendall tau: {kendall_result.statistic:.6f}",
        f"  Kendall p-value: {kendall_result.pvalue:.6g}",
        "",
        "Regression of RF prediction on QM/MM reference:",
        f"  Slope: {slope:.6f}",
        f"  Intercept: {intercept:.6f}",
        f"  Regression R-squared: {regression_r ** 2:.6f}",
        f"  Regression p-value: {regression_p:.6g}",
        f"  Slope standard error: {slope_stderr:.6f}",
        "",
        "Prediction error:",
        f"  MAE: {mae:.6f} kcal/mol",
        f"  RMSE: {rmse:.6f} kcal/mol",
        (
            "  Mean offset (prediction-reference): "
            f"{mean_offset:.6f} kcal/mol"
        ),
        (
            "  Median offset (prediction-reference): "
            f"{median_offset:.6f} kcal/mol"
        ),
        f"  Mean absolute residual: {absolute_error.mean():.6f}",
        f"  Maximum absolute residual: {absolute_error.max():.6f}",
        "",
        "Distributions:",
        f"  Reference mean: {reference.mean():.6f}",
        f"  Reference SD: {reference.std(ddof=1):.6f}",
        f"  Reference range: {reference.max() - reference.min():.6f}",
        f"  Prediction mean: {prediction.mean():.6f}",
        f"  Prediction SD: {prediction.std(ddof=1):.6f}",
        f"  Prediction range: {prediction.max() - prediction.min():.6f}",
        (
            "  Prediction/reference SD ratio: "
            f"{prediction.std(ddof=1) / reference.std(ddof=1):.6f}"
        ),
        (
            "  Prediction/reference range ratio: "
            f"{(prediction.max() - prediction.min()) / (reference.max() - reference.min()):.6f}"
        ),
        "",
        "Bootstrap:",
        f"  Resamples: {BOOTSTRAP_SAMPLES}",
        f"  Random seed: {RANDOM_SEED}",
        (
            "  Valid Pearson bootstrap samples: "
            f"{bootstrap['pearson_r'].notna().sum()}"
        ),
        (
            "  Valid Spearman bootstrap samples: "
            f"{bootstrap['spearman_rho'].notna().sum()}"
        ),
    ]

    OUT_SUMMARY.write_text(
        "\n".join(lines) + "\n"
    )

    print("\n".join(lines))
    print("\nWritten:", OUT_SUMMARY)
    print("Written:", OUT_BOOTSTRAP)
    print("Written:", OUT_RESIDUALS)


if __name__ == "__main__":
    main()
