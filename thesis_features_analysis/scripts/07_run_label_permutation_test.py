"""
Run label-permutation tests for the final batch0100 nested-CV classifier.

Scientific question
-------------------
The observed nested-CV analysis showed classification performance above
the prior dummy baseline. This script asks whether similarly high
performance could arise when the low/high labels are unrelated to the
structural features.

For every permutation:

1. The 83 clear low/high labels are randomly permuted.
2. A complete nested cross-validation analysis is rerun.
3. Outer stratified folds are generated from the permuted labels.
4. Within every outer-training fold:
       - median imputation,
       - variance filtering,
       - SelectKBest feature selection,
       - selection of k,
       - standard scaling,
       - selection of logistic-regression C,
       - class-balanced logistic regression
   are fitted using training data only.
5. Mean outer-fold balanced accuracy is recorded.

The observed scores are read from the completed script-05 results.
They are not refitted in this script.

Primary permutation statistic
-----------------------------
Mean outer-fold balanced accuracy.

Secondary descriptive statistics
--------------------------------
Mean outer-fold ROC AUC for the low class and mean MCC are also saved,
but balanced accuracy is the pre-specified primary statistic.

Important
---------
The batch0100 labels originate from old RF-predicted barriers.
The permutation test therefore evaluates whether the structural
classification of the old model's behaviour exceeds that expected
under randomly assigned low/high labels. It does not establish
prediction of measured or improved QM/MM activation barriers.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# General configuration
# ============================================================

BATCH = "batch0100"

POSITIVE_CLASS = "low"
CLASS_LABELS = ("low", "high")

RANDOM_STATE = 42

EXPECTED_TOTAL_SAMPLES = 100
EXPECTED_LOW_COUNT = 26
EXPECTED_MIDDLE_COUNT = 17
EXPECTED_HIGH_COUNT = 57
EXPECTED_CLEAR_COUNT = 83

FORMAL_FEATURE_SETS = (
    "residue_ligand",
    "residue_cofactor",
    "residue_residue",
    "combined_all",
)

C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

K_VALUES_BY_SET: dict[str, list[int | str]] = {
    "residue_ligand": [
        4,
        8,
        12,
        "all",
    ],
    "residue_cofactor": [
        5,
        10,
        20,
        40,
        "all",
    ],
    "residue_residue": [
        5,
        10,
        20,
        50,
    ],
    "combined_all": [
        5,
        10,
        20,
        50,
    ],
}


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    BASE_DIR
    / "results"
    / BATCH
    / "tables"
    / "features_with_barriers.parquet"
)

FEATURE_SET_DIR = (
    BASE_DIR
    / "results"
    / BATCH
    / "feature_sets"
)

FEATURE_SET_FILES = {
    "residue_ligand": (
        FEATURE_SET_DIR
        / "residue_ligand_features.txt"
    ),
    "residue_cofactor": (
        FEATURE_SET_DIR
        / "residue_cofactor_features.txt"
    ),
    "residue_residue": (
        FEATURE_SET_DIR
        / "residue_residue_features.txt"
    ),
    "combined_all": (
        FEATURE_SET_DIR
        / "combined_all_features.txt"
    ),
}

OBSERVED_METRICS_FILE = (
    BASE_DIR
    / "results"
    / BATCH
    / "nested_cv"
    / "final"
    / "nested_cv_fold_metrics.tsv"
)

PERMUTATION_ROOT = (
    BASE_DIR
    / "results"
    / BATCH
    / "permutation"
)


# ============================================================
# Console helper
# ============================================================

def log(
    message: str = "",
) -> None:
    """Print a console message immediately."""
    print(
        message,
        flush=True,
    )


# ============================================================
# Command-line configuration
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse run mode and computational options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run label-permutation tests for "
            "the batch0100 nested-CV classifier."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "debug",
            "pilot",
            "final",
        ],
        default="debug",
        help=(
            "debug: fast code check using outer 5x1 / inner 3; "
            "pilot: formal CV structure using outer 5x5 / inner 5; "
            "final: formal CV structure using outer 5x5 / inner 5."
        ),
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=None,
        help=(
            "Number of permutations. Defaults: "
            "debug=2, pilot=3, final=40."
        ),
    )

    parser.add_argument(
        "--start-permutation",
        type=int,
        default=1,
        help=(
            "First permutation ID. Useful for running "
            "final permutations in separate batches."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel jobs used inside GridSearchCV. "
            "Use 1 for debug; 2 is recommended for final."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow an existing output directory for this exact "
            "run range to be overwritten."
        ),
    )

    return parser.parse_args()


def get_mode_configuration(
    mode: str,
    requested_n_permutations: int | None,
) -> dict[str, int]:
    """Return CV settings and default permutation counts."""
    if mode == "debug":
        default_permutations = 2

        configuration = {
            "outer_splits": 5,
            "outer_repeats": 1,
            "inner_splits": 3,
        }

    elif mode == "pilot":
        default_permutations = 3

        configuration = {
            "outer_splits": 5,
            "outer_repeats": 5,
            "inner_splits": 5,
        }

    elif mode == "final":
        default_permutations = 40

        configuration = {
            "outer_splits": 5,
            "outer_repeats": 5,
            "inner_splits": 5,
        }

    else:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    n_permutations = (
        default_permutations
        if requested_n_permutations is None
        else requested_n_permutations
    )

    if n_permutations < 1:
        raise ValueError(
            "--n-permutations must be at least 1."
        )

    configuration[
        "n_permutations"
    ] = int(
        n_permutations
    )

    return configuration


# ============================================================
# Input validation
# ============================================================

def validate_input_paths() -> None:
    """Confirm that all required inputs exist."""
    required_files = [
        MASTER_FILE,
        OBSERVED_METRICS_FILE,
        *FEATURE_SET_FILES.values(),
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required input files are missing:\n"
            + "\n".join(
                str(file_path)
                for file_path in missing_files
            )
        )


def read_feature_list(
    feature_file: Path,
) -> list[str]:
    """Read one unique feature name per line."""
    features = [
        line.strip()
        for line in feature_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not features:
        raise ValueError(
            "Empty feature manifest:\n"
            f"{feature_file}"
        )

    if len(features) != len(
        set(features)
    ):
        raise ValueError(
            "Duplicated feature names found in:\n"
            f"{feature_file}"
        )

    return features


def load_feature_sets() -> dict[str, list[str]]:
    """Load and validate the four formal feature manifests."""
    feature_sets = {
        feature_set: read_feature_list(
            FEATURE_SET_FILES[
                feature_set
            ]
        )
        for feature_set in FORMAL_FEATURE_SETS
    }

    atomic_union = set(
        feature_sets[
            "residue_ligand"
        ]
    )

    atomic_union.update(
        feature_sets[
            "residue_cofactor"
        ]
    )

    atomic_union.update(
        feature_sets[
            "residue_residue"
        ]
    )

    if atomic_union != set(
        feature_sets[
            "combined_all"
        ]
    ):
        raise ValueError(
            "combined_all is not the exact union "
            "of the three atomic feature sets."
        )

    return feature_sets


def load_analysis_dataset(
    feature_sets: dict[str, list[str]],
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Load the 83 clear low/high samples used by script 05."""
    table = pd.read_parquet(
        MASTER_FILE
    )

    required_columns = {
        "ligand_id",
        "barrier",
        "group",
    }

    missing_columns = (
        required_columns
        - set(table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Master table is missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    if table[
        "ligand_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicated ligand IDs found in master table."
        )

    if len(table) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError(
            "Unexpected total sample count: "
            f"{len(table)}"
        )

    counts = (
        table[
            "group"
        ]
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        "low": EXPECTED_LOW_COUNT,
        "middle": EXPECTED_MIDDLE_COUNT,
        "high": EXPECTED_HIGH_COUNT,
    }

    for group_name, expected_count in (
        expected_counts.items()
    ):
        if int(
            counts.get(
                group_name,
                0,
            )
        ) != expected_count:
            raise ValueError(
                f"Unexpected count for {group_name}."
            )

    required_features = set(
        feature_sets[
            "combined_all"
        ]
    )

    missing_features = (
        required_features
        - set(table.columns)
    )

    if missing_features:
        raise ValueError(
            "Formal model features are missing "
            "from the master table."
        )

    clear_table = table.loc[
        table[
            "group"
        ].isin(
            CLASS_LABELS
        )
    ].copy()

    if len(
        clear_table
    ) != EXPECTED_CLEAR_COUNT:
        raise ValueError(
            "Expected 83 clear low/high samples, "
            f"observed {len(clear_table)}."
        )

    clear_table = (
        clear_table
        .set_index(
            "ligand_id"
        )
        .sort_index()
    )

    y = clear_table[
        "group"
    ].copy()

    if int(
        (
            y
            == "low"
        ).sum()
    ) != EXPECTED_LOW_COUNT:
        raise ValueError(
            "Unexpected low count."
        )

    if int(
        (
            y
            == "high"
        ).sum()
    ) != EXPECTED_HIGH_COUNT:
        raise ValueError(
            "Unexpected high count."
        )

    return (
        clear_table,
        y,
    )


# ============================================================
# Observed script-05 results
# ============================================================

def load_observed_scores() -> pd.DataFrame:
    """
    Read observed performance from the completed final script-05 run.

    The observed analysis is not refitted here.
    """
    metrics = pd.read_csv(
        OBSERVED_METRICS_FILE,
        sep="\t",
    )

    required_columns = {
        "model_type",
        "feature_set",
        "balanced_accuracy",
        "roc_auc_low",
        "mcc",
    }

    missing_columns = (
        required_columns
        - set(metrics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Observed metric table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    metrics = metrics[
        metrics[
            "model_type"
        ]
        == "logistic_regression"
    ].copy()

    rows = []

    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        subset = metrics[
            metrics[
                "feature_set"
            ]
            == feature_set
        ]

        if len(
            subset
        ) != 25:
            raise ValueError(
                f"{feature_set}: expected 25 observed "
                f"outer-fold results, observed {len(subset)}."
            )

        rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "observed_mean_balanced_accuracy": (
                    subset[
                        "balanced_accuracy"
                    ].mean()
                ),
                "observed_mean_roc_auc_low": (
                    subset[
                        "roc_auc_low"
                    ].mean()
                ),
                "observed_mean_mcc": (
                    subset[
                        "mcc"
                    ].mean()
                ),
                "n_observed_outer_folds": (
                    len(subset)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Model pipeline
# ============================================================

def build_model_pipeline() -> Pipeline:
    """Build the same leakage-safe pipeline used in script 05."""
    try:
        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

    except TypeError as error:
        raise RuntimeError(
            "This script requires "
            "SimpleImputer(keep_empty_features=True). "
            f"Installed scikit-learn version: "
            f"{sklearn.__version__}"
        ) from error

    return Pipeline(
        steps=[
            (
                "imputer",
                imputer,
            ),
            (
                "variance",
                VarianceThreshold(
                    threshold=0.0
                ),
            ),
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k="all",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_parameter_grid(
    feature_set: str,
    n_input_features: int,
) -> dict[str, list[Any]]:
    """Return the same k and C grid used by script 05."""
    valid_k_values: list[int | str] = []

    for k_value in (
        K_VALUES_BY_SET[
            feature_set
        ]
    ):
        if k_value == "all":
            valid_k_values.append(
                k_value
            )

        elif isinstance(
            k_value,
            int,
        ) and (
            k_value
            <= n_input_features
        ):
            valid_k_values.append(
                k_value
            )

    if not valid_k_values:
        raise ValueError(
            f"No valid k values for {feature_set}."
        )

    return {
        "selector__k": (
            valid_k_values
        ),
        "classifier__C": (
            C_VALUES
        ),
    }


# ============================================================
# Label permutation
# ============================================================

def permute_labels(
    original_y: pd.Series,
    permutation_id: int,
) -> pd.Series:
    """
    Permute labels reproducibly while preserving ligand order.

    Each permutation ID maps to a deterministic independent random seed.
    """
    permutation_seed = (
        RANDOM_STATE
        + 100_000
        + permutation_id
    )

    generator = np.random.default_rng(
        permutation_seed
    )

    permuted_values = generator.permutation(
        original_y.to_numpy()
    )

    permuted_y = pd.Series(
        permuted_values,
        index=original_y.index,
        name=original_y.name,
    )

    if (
        permuted_y.value_counts().to_dict()
        != original_y.value_counts().to_dict()
    ):
        raise RuntimeError(
            "Permutation changed class counts."
        )

    return permuted_y


# ============================================================
# Metrics
# ============================================================

def get_low_probability(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Return probability assigned to the low class."""
    classifier = fitted_pipeline.named_steps[
        "classifier"
    ]

    classes = list(
        classifier.classes_
    )

    if POSITIVE_CLASS not in classes:
        raise ValueError(
            "Low class missing from fitted model."
        )

    positive_index = classes.index(
        POSITIVE_CLASS
    )

    return fitted_pipeline.predict_proba(
        X_test
    )[:, positive_index]


def calculate_outer_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probability_low: np.ndarray,
) -> dict[str, float]:
    """Calculate the permutation statistics for one outer test fold."""
    y_binary = (
        y_true
        == POSITIVE_CLASS
    ).astype(int)

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    try:
        roc_auc = roc_auc_score(
            y_binary,
            probability_low,
        )

    except ValueError:
        roc_auc = np.nan

    return {
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "roc_auc_low": (
            roc_auc
        ),
        "mcc": (
            mcc
        ),
    }


# ============================================================
# One feature-set permutation
# ============================================================

def evaluate_permuted_feature_set(
    clear_table: pd.DataFrame,
    permuted_y: pd.Series,
    feature_set: str,
    features: list[str],
    permutation_id: int,
    outer_splits: int,
    outer_repeats: int,
    inner_splits: int,
    n_jobs: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Run a complete nested CV for one feature set and one permutation.

    Outer folds are regenerated from the permuted labels so that each
    permutation follows the same stratified-CV procedure as the
    observed analysis.
    """
    X = clear_table[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    parameter_grid = get_parameter_grid(
        feature_set=feature_set,
        n_input_features=X.shape[1],
    )

    outer_random_state = (
        RANDOM_STATE
        + 200_000
        + permutation_id
    )

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=outer_random_state,
    )

    outer_split_indices = list(
        outer_cv.split(
            X,
            permuted_y,
        )
    )

    expected_outer_count = (
        outer_splits
        * outer_repeats
    )

    if len(
        outer_split_indices
    ) != expected_outer_count:
        raise RuntimeError(
            "Unexpected number of outer splits."
        )

    outer_metric_rows = []

    for split_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_split_indices,
        start=1,
    ):
        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = permuted_y.iloc[
            train_indices
        ]

        y_test = permuted_y.iloc[
            test_indices
        ]

        if y_train.nunique() != 2:
            raise RuntimeError(
                "Outer-training fold contains "
                "fewer than two classes."
            )

        if y_test.nunique() != 2:
            raise RuntimeError(
                "Outer-test fold contains "
                "fewer than two classes."
            )

        inner_random_state = (
            RANDOM_STATE
            + 300_000
            + permutation_id * 100
            + split_index
        )

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=(
                inner_random_state
            ),
        )

        search = GridSearchCV(
            estimator=(
                build_model_pipeline()
            ),
            param_grid=(
                parameter_grid
            ),
            scoring=(
                "balanced_accuracy"
            ),
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(
            X_train,
            y_train,
        )

        best_model = (
            search.best_estimator_
        )

        y_pred = best_model.predict(
            X_test
        )

        probability_low = (
            get_low_probability(
                fitted_pipeline=(
                    best_model
                ),
                X_test=X_test,
            )
        )

        metrics = (
            calculate_outer_metrics(
                y_true=y_test,
                y_pred=y_pred,
                probability_low=(
                    probability_low
                ),
            )
        )

        outer_metric_rows.append(
            {
                "permutation_id": (
                    permutation_id
                ),
                "feature_set": (
                    feature_set
                ),
                "outer_split": (
                    split_index
                ),
                "n_train": (
                    len(train_indices)
                ),
                "n_test": (
                    len(test_indices)
                ),
                "n_train_low": int(
                    (
                        y_train
                        == "low"
                    ).sum()
                ),
                "n_train_high": int(
                    (
                        y_train
                        == "high"
                    ).sum()
                ),
                "n_test_low": int(
                    (
                        y_test
                        == "low"
                    ).sum()
                ),
                "n_test_high": int(
                    (
                        y_test
                        == "high"
                    ).sum()
                ),
                "best_k": (
                    search.best_params_[
                        "selector__k"
                    ]
                ),
                "best_c": float(
                    search.best_params_[
                        "classifier__C"
                    ]
                ),
                "best_inner_balanced_accuracy": (
                    search.best_score_
                ),
                **metrics,
            }
        )

    fold_metrics = pd.DataFrame(
        outer_metric_rows
    )

    summary = {
        "permutation_id": (
            permutation_id
        ),
        "feature_set": (
            feature_set
        ),
        "n_outer_folds": (
            len(fold_metrics)
        ),
        "mean_balanced_accuracy": (
            fold_metrics[
                "balanced_accuracy"
            ].mean()
        ),
        "std_balanced_accuracy": (
            fold_metrics[
                "balanced_accuracy"
            ].std(
                ddof=1
            )
        ),
        "mean_roc_auc_low": (
            fold_metrics[
                "roc_auc_low"
            ].mean()
        ),
        "std_roc_auc_low": (
            fold_metrics[
                "roc_auc_low"
            ].std(
                ddof=1
            )
        ),
        "mean_mcc": (
            fold_metrics[
                "mcc"
            ].mean()
        ),
        "std_mcc": (
            fold_metrics[
                "mcc"
            ].std(
                ddof=1
            )
        ),
    }

    return (
        summary,
        outer_metric_rows,
    )


# ============================================================
# Output directory
# ============================================================

def build_run_name(
    mode: str,
    start_permutation: int,
    n_permutations: int,
) -> str:
    """Build a deterministic directory name for one permutation batch."""
    end_permutation = (
        start_permutation
        + n_permutations
        - 1
    )

    return (
        f"{mode}_"
        f"perm{start_permutation:04d}_"
        f"{end_permutation:04d}"
    )


def prepare_output_directory(
    run_name: str,
    overwrite: bool,
) -> Path:
    """Create one isolated output directory for this run batch."""
    output_directory = (
        PERMUTATION_ROOT
        / run_name
    )

    protected_files = [
        output_directory
        / "permutation_scores.tsv",
        output_directory
        / "permutation_outer_fold_metrics.tsv",
        output_directory
        / "observed_scores.tsv",
        output_directory
        / "permutation_test_summary.tsv",
        output_directory
        / "07_permutation_run_configuration.tsv",
    ]

    existing_files = [
        file_path
        for file_path in protected_files
        if file_path.exists()
    ]

    if (
        existing_files
        and not overwrite
    ):
        raise FileExistsError(
            "Output files already exist for this "
            "permutation batch. Use --overwrite only "
            "if replacement is intentional:\n"
            + "\n".join(
                str(file_path)
                for file_path in existing_files
            )
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


# ============================================================
# Permutation p-values and summaries
# ============================================================

def build_permutation_test_summary(
    permutation_scores: pd.DataFrame,
    observed_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare observed BA with the empirical null distribution.

    P = (number of null scores >= observed + 1) / (N + 1)
    """
    rows = []

    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        null_subset = (
            permutation_scores[
                permutation_scores[
                    "feature_set"
                ]
                == feature_set
            ]
        )

        observed_row = (
            observed_scores[
                observed_scores[
                    "feature_set"
                ]
                == feature_set
            ]
            .iloc[0]
        )

        observed_ba = float(
            observed_row[
                "observed_mean_balanced_accuracy"
            ]
        )

        null_ba = pd.to_numeric(
            null_subset[
                "mean_balanced_accuracy"
            ],
            errors="raise",
        )

        n_permutations = len(
            null_ba
        )

        n_null_ge_observed = int(
            (
                null_ba
                >= observed_ba
            ).sum()
        )

        permutation_p_value = (
            n_null_ge_observed
            + 1
        ) / (
            n_permutations
            + 1
        )

        rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "observed_mean_balanced_accuracy": (
                    observed_ba
                ),
                "null_mean_balanced_accuracy": (
                    null_ba.mean()
                ),
                "null_std_balanced_accuracy": (
                    null_ba.std(
                        ddof=1
                    )
                    if n_permutations > 1
                    else np.nan
                ),
                "null_median_balanced_accuracy": (
                    null_ba.median()
                ),
                "null_min_balanced_accuracy": (
                    null_ba.min()
                ),
                "null_max_balanced_accuracy": (
                    null_ba.max()
                ),
                "n_permutations": (
                    n_permutations
                ),
                "n_null_ge_observed": (
                    n_null_ge_observed
                ),
                "permutation_p_value": (
                    permutation_p_value
                ),
                "p_value_formula": (
                    "(n_null_ge_observed + 1) / "
                    "(n_permutations + 1)"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Configuration and validation
# ============================================================

def save_run_configuration(
    output_file: Path,
    arguments: argparse.Namespace,
    configuration: dict[str, int],
    feature_sets: dict[str, list[str]],
    total_elapsed_seconds: float,
) -> None:
    """Save complete provenance for this permutation batch."""
    settings: dict[str, Any] = {
        "batch": BATCH,
        "mode": arguments.mode,
        "start_permutation": (
            arguments.start_permutation
        ),
        "n_permutations": (
            configuration[
                "n_permutations"
            ]
        ),
        "end_permutation": (
            arguments.start_permutation
            + configuration[
                "n_permutations"
            ]
            - 1
        ),
        "outer_cv": (
            "RepeatedStratifiedKFold"
        ),
        "outer_splits": (
            configuration[
                "outer_splits"
            ]
        ),
        "outer_repeats": (
            configuration[
                "outer_repeats"
            ]
        ),
        "outer_total_folds_per_permutation": (
            configuration[
                "outer_splits"
            ]
            * configuration[
                "outer_repeats"
            ]
        ),
        "outer_folds_regenerated_after_each_label_permutation": (
            True
        ),
        "inner_cv": (
            "StratifiedKFold with shuffle"
        ),
        "inner_splits": (
            configuration[
                "inner_splits"
            ]
        ),
        "primary_permutation_statistic": (
            "mean outer-fold balanced accuracy"
        ),
        "secondary_statistics": (
            "mean ROC AUC low; mean MCC"
        ),
        "pipeline": (
            "median imputation; variance threshold; "
            "SelectKBest(f_classif); StandardScaler; "
            "class-balanced LogisticRegression"
        ),
        "logistic_regularisation": (
            "default L2"
        ),
        "logistic_solver": (
            "liblinear"
        ),
        "c_values": json.dumps(
            C_VALUES
        ),
        "k_values_by_set": json.dumps(
            K_VALUES_BY_SET
        ),
        "base_random_state": (
            RANDOM_STATE
        ),
        "n_jobs": (
            arguments.n_jobs
        ),
        "observed_scores_source": (
            str(
                OBSERVED_METRICS_FILE
            )
        ),
        "observed_models_refitted": (
            False
        ),
        "selection_uses_script_02_ranking": (
            False
        ),
        "preprocessing_outside_cv": (
            False
        ),
        "n_clear_samples": (
            EXPECTED_CLEAR_COUNT
        ),
        "n_low": (
            EXPECTED_LOW_COUNT
        ),
        "n_high": (
            EXPECTED_HIGH_COUNT
        ),
        "n_middle_excluded": (
            EXPECTED_MIDDLE_COUNT
        ),
        "n_residue_ligand_features": len(
            feature_sets[
                "residue_ligand"
            ]
        ),
        "n_residue_cofactor_features": len(
            feature_sets[
                "residue_cofactor"
            ]
        ),
        "n_residue_residue_features": len(
            feature_sets[
                "residue_residue"
            ]
        ),
        "n_combined_all_features": len(
            feature_sets[
                "combined_all"
            ]
        ),
        "barrier_source": (
            "old RF-predicted barriers"
        ),
        "total_elapsed_seconds": (
            total_elapsed_seconds
        ),
        "python_version": (
            sys.version.replace(
                "\n",
                " ",
            )
        ),
        "platform": (
            platform.platform()
        ),
        "numpy_version": (
            np.__version__
        ),
        "pandas_version": (
            pd.__version__
        ),
        "scipy_version": (
            scipy.__version__
        ),
        "scikit_learn_version": (
            sklearn.__version__
        ),
    }

    pd.DataFrame(
        [
            {
                "setting": key,
                "value": value,
            }
            for key, value in (
                settings.items()
            )
        ]
    ).to_csv(
        output_file,
        sep="\t",
        index=False,
    )


def validate_outputs(
    permutation_scores: pd.DataFrame,
    outer_fold_metrics: pd.DataFrame,
    configuration: dict[str, int],
) -> None:
    """Check expected output dimensions and score ranges."""
    n_permutations = (
        configuration[
            "n_permutations"
        ]
    )

    n_outer_folds = (
        configuration[
            "outer_splits"
        ]
        * configuration[
            "outer_repeats"
        ]
    )

    expected_score_rows = (
        n_permutations
        * len(
            FORMAL_FEATURE_SETS
        )
    )

    expected_fold_rows = (
        expected_score_rows
        * n_outer_folds
    )

    if len(
        permutation_scores
    ) != expected_score_rows:
        raise RuntimeError(
            "Unexpected permutation-score row count: "
            f"expected {expected_score_rows}, "
            f"observed {len(permutation_scores)}."
        )

    if len(
        outer_fold_metrics
    ) != expected_fold_rows:
        raise RuntimeError(
            "Unexpected permutation outer-fold row count: "
            f"expected {expected_fold_rows}, "
            f"observed {len(outer_fold_metrics)}."
        )

    duplicate_scores = (
        permutation_scores.duplicated(
            subset=[
                "permutation_id",
                "feature_set",
            ]
        )
    )

    if duplicate_scores.any():
        raise RuntimeError(
            "Duplicated permutation/feature-set "
            "summary rows detected."
        )

    if not permutation_scores[
        "mean_balanced_accuracy"
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise RuntimeError(
            "Balanced accuracy outside [0, 1]."
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run one complete permutation-test batch."""
    arguments = parse_arguments()

    configuration = (
        get_mode_configuration(
            mode=arguments.mode,
            requested_n_permutations=(
                arguments.n_permutations
            ),
        )
    )

    if (
        arguments.start_permutation
        < 1
    ):
        raise ValueError(
            "--start-permutation must be >= 1."
        )

    run_name = build_run_name(
        mode=arguments.mode,
        start_permutation=(
            arguments.start_permutation
        ),
        n_permutations=(
            configuration[
                "n_permutations"
            ]
        ),
    )

    output_directory = (
        prepare_output_directory(
            run_name=run_name,
            overwrite=arguments.overwrite,
        )
    )

    overall_start = (
        time.perf_counter()
    )

    validate_input_paths()

    feature_sets = (
        load_feature_sets()
    )

    (
        clear_table,
        original_y,
    ) = load_analysis_dataset(
        feature_sets
    )

    observed_scores = (
        load_observed_scores()
    )

    log("=" * 72)
    log(
        "Batch0100 label-permutation test"
    )
    log("=" * 72)

    log(
        f"Mode: {arguments.mode}"
    )

    log(
        "Samples: "
        f"{len(original_y)} "
        f"({EXPECTED_LOW_COUNT} low, "
        f"{EXPECTED_HIGH_COUNT} high)"
    )

    log(
        "Outer CV: "
        f"{configuration['outer_splits']} folds x "
        f"{configuration['outer_repeats']} repeats"
    )

    log(
        "Inner CV: "
        f"{configuration['inner_splits']} folds"
    )

    log(
        "Permutations in this run: "
        f"{configuration['n_permutations']}"
    )

    log(
        "Permutation IDs: "
        f"{arguments.start_permutation} to "
        f"{arguments.start_permutation + configuration['n_permutations'] - 1}"
    )

    log(
        f"GridSearchCV n_jobs: "
        f"{arguments.n_jobs}"
    )

    log()
    log(
        "Observed script-05 scores:"
    )

    log(
        observed_scores.to_string(
            index=False
        )
    )

    all_score_rows = []
    all_outer_fold_rows = []

    first_permutation = (
        arguments.start_permutation
    )

    last_permutation = (
        first_permutation
        + configuration[
            "n_permutations"
        ]
        - 1
    )

    for permutation_id in range(
        first_permutation,
        last_permutation + 1,
    ):
        permutation_start = (
            time.perf_counter()
        )

        permuted_y = permute_labels(
            original_y=original_y,
            permutation_id=(
                permutation_id
            ),
        )

        log()
        log("=" * 72)
        log(
            f"Permutation "
            f"{permutation_id} "
            f"({permutation_id - first_permutation + 1}/"
            f"{configuration['n_permutations']})"
        )
        log("=" * 72)

        for feature_set in (
            FORMAL_FEATURE_SETS
        ):
            feature_start = (
                time.perf_counter()
            )

            (
                score_row,
                outer_rows,
            ) = (
                evaluate_permuted_feature_set(
                    clear_table=(
                        clear_table
                    ),
                    permuted_y=(
                        permuted_y
                    ),
                    feature_set=(
                        feature_set
                    ),
                    features=(
                        feature_sets[
                            feature_set
                        ]
                    ),
                    permutation_id=(
                        permutation_id
                    ),
                    outer_splits=(
                        configuration[
                            "outer_splits"
                        ]
                    ),
                    outer_repeats=(
                        configuration[
                            "outer_repeats"
                        ]
                    ),
                    inner_splits=(
                        configuration[
                            "inner_splits"
                        ]
                    ),
                    n_jobs=(
                        arguments.n_jobs
                    ),
                )
            )

            feature_elapsed = (
                time.perf_counter()
                - feature_start
            )

            score_row[
                "elapsed_seconds"
            ] = feature_elapsed

            all_score_rows.append(
                score_row
            )

            all_outer_fold_rows.extend(
                outer_rows
            )

            log(
                f"  {feature_set:<18} "
                f"BA={score_row['mean_balanced_accuracy']:.3f} "
                f"AUC={score_row['mean_roc_auc_low']:.3f} "
                f"MCC={score_row['mean_mcc']:.3f} "
                f"time={feature_elapsed / 60:.2f} min"
            )

        permutation_elapsed = (
            time.perf_counter()
            - permutation_start
        )

        completed_permutations = (
            permutation_id
            - first_permutation
            + 1
        )

        average_seconds = (
            (
                time.perf_counter()
                - overall_start
            )
            / completed_permutations
        )

        remaining_permutations = (
            configuration[
                "n_permutations"
            ]
            - completed_permutations
        )

        estimated_remaining_minutes = (
            average_seconds
            * remaining_permutations
            / 60
        )

        log(
            f"Permutation {permutation_id} "
            f"completed in "
            f"{permutation_elapsed / 60:.2f} min"
        )

        log(
            "Estimated remaining time for "
            f"this batch: "
            f"{estimated_remaining_minutes:.1f} min"
        )

        # Save checkpoint after every completed permutation.
        checkpoint_scores = (
            pd.DataFrame(
                all_score_rows
            )
        )

        checkpoint_folds = (
            pd.DataFrame(
                all_outer_fold_rows
            )
        )

        checkpoint_scores.to_csv(
            output_directory
            / "permutation_scores.tsv",
            sep="\t",
            index=False,
        )

        checkpoint_folds.to_csv(
            output_directory
            / "permutation_outer_fold_metrics.tsv",
            sep="\t",
            index=False,
        )

    permutation_scores = (
        pd.DataFrame(
            all_score_rows
        )
    )

    outer_fold_metrics = (
        pd.DataFrame(
            all_outer_fold_rows
        )
    )

    validate_outputs(
        permutation_scores=(
            permutation_scores
        ),
        outer_fold_metrics=(
            outer_fold_metrics
        ),
        configuration=(
            configuration
        ),
    )

    permutation_summary = (
        build_permutation_test_summary(
            permutation_scores=(
                permutation_scores
            ),
            observed_scores=(
                observed_scores
            ),
        )
    )

    observed_scores.to_csv(
        output_directory
        / "observed_scores.tsv",
        sep="\t",
        index=False,
    )

    permutation_scores.to_csv(
        output_directory
        / "permutation_scores.tsv",
        sep="\t",
        index=False,
    )

    outer_fold_metrics.to_csv(
        output_directory
        / "permutation_outer_fold_metrics.tsv",
        sep="\t",
        index=False,
    )

    permutation_summary.to_csv(
        output_directory
        / "permutation_test_summary.tsv",
        sep="\t",
        index=False,
    )

    total_elapsed_seconds = (
        time.perf_counter()
        - overall_start
    )

    save_run_configuration(
        output_file=(
            output_directory
            / "07_permutation_run_configuration.tsv"
        ),
        arguments=arguments,
        configuration=configuration,
        feature_sets=feature_sets,
        total_elapsed_seconds=(
            total_elapsed_seconds
        ),
    )

    log()
    log("=" * 72)
    log(
        "Permutation-test summary"
    )
    log("=" * 72)

    log(
        permutation_summary.to_string(
            index=False
        )
    )

    log()
    log(
        "Output row counts:"
    )

    log(
        "  permutation scores: "
        f"{len(permutation_scores)}"
    )

    log(
        "  permutation outer-fold metrics: "
        f"{len(outer_fold_metrics)}"
    )

    log()
    log(
        "Saved outputs to:"
    )

    log(
        str(output_directory)
    )

    log()
    log(
        "Total elapsed time: "
        f"{total_elapsed_seconds / 60:.2f} minutes"
    )

    log()
    log(
        "Done."
    )


if __name__ == "__main__":
    main()