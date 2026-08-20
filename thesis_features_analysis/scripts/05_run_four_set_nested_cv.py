"""
Run the formal four-set nested cross-validation analysis for batch0100.

Four predefined distance-feature sets are evaluated:

1. residue_ligand
2. residue_cofactor
3. residue_residue
4. combined_all

The feature manifests are produced by script 03 without using the
low/high labels or the exploratory ranking from script 02.

Nested-CV design
----------------
Outer cross-validation:
    Used only for final model evaluation.

Inner cross-validation:
    Used to select:
    - the number of features retained by SelectKBest;
    - the logistic-regression regularisation parameter C.

All data-dependent preprocessing is fitted inside the CV pipeline:

    median imputation
        -> variance filtering
        -> SelectKBest
        -> standard scaling
        -> class-balanced logistic regression

A prior-based DummyClassifier is evaluated on exactly the same outer
train/test splits.

Important
---------
The batch0100 low/high labels originate from old RF-predicted barriers.
This classifier therefore characterises the structural behaviour of the
old model. It does not predict measured or improved QM/MM barriers.

The intermediate group is excluded before classification.

This script does not perform the final label-permutation test. It does,
however, save all outer-fold predictions, best parameters and selected
features required for later stability and permutation analyses.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
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
CLASS_LABELS = ["low", "high"]

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

NESTED_CV_ROOT = (
    BASE_DIR
    / "results"
    / BATCH
    / "nested_cv"
)


# ============================================================
# Console logging
# ============================================================

def log(
    message: str = "",
) -> None:
    """Print a message immediately, including when output is piped."""
    print(
        message,
        flush=True,
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Parse debug/final mode and parallel-processing settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Run four-set nested cross-validation "
            "for batch0100."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "debug",
            "final",
        ],
        default="debug",
        help=(
            "debug: outer 5x1 and inner 3-fold; "
            "final: outer 5x5 and inner 5-fold."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel jobs used by GridSearchCV. "
            "Use 1 for the safest run, or 2-4 "
            "if sufficient CPU and memory are available."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow existing output TSV files in the "
            "selected mode directory to be overwritten."
        ),
    )

    return parser.parse_args()


def get_mode_configuration(
    mode: str,
) -> dict[str, int]:
    """Return the cross-validation settings for one run mode."""
    if mode == "debug":
        return {
            "outer_splits": 5,
            "outer_repeats": 1,
            "inner_splits": 3,
        }

    if mode == "final":
        return {
            "outer_splits": 5,
            "outer_repeats": 5,
            "inner_splits": 5,
        }

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# ============================================================
# Input validation and loading
# ============================================================

def validate_input_paths() -> None:
    """Confirm that the master table and four manifests exist."""
    required_files = [
        MASTER_FILE,
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
    """Read one feature name per line and reject duplicates."""
    features = [
        line.strip()
        for line in feature_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not features:
        raise ValueError(
            "Feature manifest is empty:\n"
            f"{feature_file}"
        )

    if len(features) != len(
        set(features)
    ):
        duplicated = (
            pd.Series(features)
            .loc[
                pd.Series(features).duplicated(
                    keep=False
                )
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicated feature names found in:\n"
            f"{feature_file}\n"
            + "\n".join(
                str(feature)
                for feature in duplicated
            )
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

    atomic_sets = {
        feature_set: set(
            feature_sets[
                feature_set
            ]
        )
        for feature_set in (
            "residue_ligand",
            "residue_cofactor",
            "residue_residue",
        )
    }

    atomic_names = list(
        atomic_sets
    )

    for first_index in range(
        len(atomic_names)
    ):
        for second_index in range(
            first_index + 1,
            len(atomic_names),
        ):
            first_name = atomic_names[
                first_index
            ]

            second_name = atomic_names[
                second_index
            ]

            overlap = (
                atomic_sets[first_name]
                & atomic_sets[second_name]
            )

            if overlap:
                raise ValueError(
                    "Formal atomic feature sets "
                    "overlap unexpectedly:\n"
                    + "\n".join(
                        sorted(overlap)
                    )
                )

    expected_combined = set().union(
        *atomic_sets.values()
    )

    observed_combined = set(
        feature_sets[
            "combined_all"
        ]
    )

    if expected_combined != observed_combined:
        missing_from_combined = sorted(
            expected_combined
            - observed_combined
        )

        extra_in_combined = sorted(
            observed_combined
            - expected_combined
        )

        raise ValueError(
            "combined_all is not the exact union "
            "of the three atomic sets.\n"
            f"Missing: {missing_from_combined}\n"
            f"Extra: {extra_in_combined}"
        )

    return feature_sets


def validate_master_table(
    table: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    """Validate sample IDs, groups and feature availability."""
    required_columns = {
        "ligand_id",
        "barrier",
        "group",
    }

    missing_metadata = (
        required_columns
        - set(table.columns)
    )

    if missing_metadata:
        raise ValueError(
            "Master table is missing metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    if table["ligand_id"].duplicated().any():
        duplicated_ids = (
            table.loc[
                table[
                    "ligand_id"
                ].duplicated(
                    keep=False
                ),
                "ligand_id",
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicated ligand IDs found:\n"
            + "\n".join(
                duplicated_ids
            )
        )

    if len(table) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError(
            "Unexpected master-table row count: "
            f"expected {EXPECTED_TOTAL_SAMPLES}, "
            f"observed {len(table)}"
        )

    group_counts = (
        table["group"]
        .value_counts()
        .to_dict()
    )

    expected_group_counts = {
        "low": EXPECTED_LOW_COUNT,
        "middle": EXPECTED_MIDDLE_COUNT,
        "high": EXPECTED_HIGH_COUNT,
    }

    for group_name, expected_count in (
        expected_group_counts.items()
    ):
        observed_count = int(
            group_counts.get(
                group_name,
                0,
            )
        )

        if observed_count != expected_count:
            raise ValueError(
                f"Unexpected {group_name} count: "
                f"expected {expected_count}, "
                f"observed {observed_count}"
            )

    all_required_features = set(
        feature_sets[
            "combined_all"
        ]
    )

    missing_features = sorted(
        all_required_features
        - set(table.columns)
    )

    if missing_features:
        raise ValueError(
            "Manifest features are missing from "
            "the master table:\n"
            + "\n".join(
                missing_features
            )
        )


def load_analysis_dataset(
    feature_sets: dict[str, list[str]],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Load the master table and retain only clear low/high samples.

    Returns
    -------
    clear_table:
        Metadata and all required feature columns.

    y:
        Low/high class labels indexed by ligand ID.

    barriers:
        Continuous old RF-predicted barriers indexed by ligand ID.
    """
    table = pd.read_parquet(
        MASTER_FILE
    )

    validate_master_table(
        table=table,
        feature_sets=feature_sets,
    )

    clear_table = table.loc[
        table["group"].isin(
            CLASS_LABELS
        )
    ].copy()

    if len(clear_table) != EXPECTED_CLEAR_COUNT:
        raise ValueError(
            "Unexpected clear low/high sample count: "
            f"expected {EXPECTED_CLEAR_COUNT}, "
            f"observed {len(clear_table)}"
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

    barriers = pd.to_numeric(
        clear_table["barrier"],
        errors="raise",
    )

    observed_clear_counts = (
        y.value_counts()
        .to_dict()
    )

    if int(
        observed_clear_counts.get(
            "low",
            0,
        )
    ) != EXPECTED_LOW_COUNT:
        raise ValueError(
            "Unexpected low count after filtering."
        )

    if int(
        observed_clear_counts.get(
            "high",
            0,
        )
    ) != EXPECTED_HIGH_COUNT:
        raise ValueError(
            "Unexpected high count after filtering."
        )

    return (
        clear_table,
        y,
        barriers,
    )


# ============================================================
# Pipeline and hyperparameter grid
# ============================================================

def build_model_pipeline() -> Pipeline:
    """
    Build the complete leakage-safe modelling pipeline.

    SimpleImputer is configured to retain entirely empty columns so that
    feature-name positions remain traceable. VarianceThreshold then
    removes constant columns using training-fold data only.
    """
    try:
        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

    except TypeError as error:
        raise RuntimeError(
            "This script requires a scikit-learn "
            "version supporting "
            "SimpleImputer(keep_empty_features=True). "
            f"Installed version: {sklearn.__version__}"
        ) from error

    pipeline = Pipeline(
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

    return pipeline


def get_parameter_grid(
    feature_set: str,
    n_input_features: int,
) -> dict[str, list[Any]]:
    """Build the inner-CV grid and remove impossible integer k values."""
    raw_k_values = K_VALUES_BY_SET[
        feature_set
    ]

    valid_k_values: list[int | str] = []

    for k_value in raw_k_values:
        if k_value == "all":
            valid_k_values.append(
                k_value
            )

        elif isinstance(
            k_value,
            int,
        ) and k_value <= n_input_features:
            valid_k_values.append(
                k_value
            )

    if not valid_k_values:
        raise ValueError(
            f"No valid k values for {feature_set} "
            f"with {n_input_features} input features."
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
# Metrics
# ============================================================

def positive_class_probability(
    fitted_model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return the predicted probability for the low-barrier class."""
    classifier = fitted_model.named_steps[
        "classifier"
    ]

    classes = list(
        classifier.classes_
    )

    if POSITIVE_CLASS not in classes:
        raise ValueError(
            "Positive class was not found in "
            f"the fitted classifier classes: {classes}"
        )

    positive_index = classes.index(
        POSITIVE_CLASS
    )

    probabilities = (
        fitted_model.predict_proba(
            X
        )[:, positive_index]
    )

    return probabilities


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probability_low: np.ndarray,
) -> dict[str, float | int]:
    """Calculate fold-level classification metrics."""
    y_true_binary = (
        y_true == POSITIVE_CLASS
    ).astype(int)

    try:
        roc_auc = roc_auc_score(
            y_true_binary,
            probability_low,
        )

    except ValueError:
        roc_auc = np.nan

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
    )

    true_low_pred_low = int(
        matrix[0, 0]
    )

    true_low_pred_high = int(
        matrix[0, 1]
    )

    true_high_pred_low = int(
        matrix[1, 0]
    )

    true_high_pred_high = int(
        matrix[1, 1]
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc_low": roc_auc,
        "precision_low": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "recall_low": recall_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "f1_low": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(
            y_true,
            y_pred,
        ),
        "true_low_pred_low": (
            true_low_pred_low
        ),
        "true_low_pred_high": (
            true_low_pred_high
        ),
        "true_high_pred_low": (
            true_high_pred_low
        ),
        "true_high_pred_high": (
            true_high_pred_high
        ),
    }


# ============================================================
# Selected-feature extraction
# ============================================================

def extract_selected_features(
    fitted_pipeline: Pipeline,
    original_features: list[str],
    feature_set: str,
    repeat_number: int,
    fold_number: int,
    outer_split_number: int,
    best_k: int | str,
    best_c: float,
) -> list[dict[str, Any]]:
    """
    Recover the feature names retained by variance filtering and
    SelectKBest, together with training-fold scores and coefficients.
    """
    original_array = np.asarray(
        original_features,
        dtype=object,
    )

    variance_step = (
        fitted_pipeline.named_steps[
            "variance"
        ]
    )

    variance_support = (
        variance_step.get_support()
    )

    after_variance = original_array[
        variance_support
    ]

    selector = fitted_pipeline.named_steps[
        "selector"
    ]

    selector_support = (
        selector.get_support()
    )

    selected_names = after_variance[
        selector_support
    ]

    all_scores = np.asarray(
        selector.scores_,
        dtype=float,
    )

    all_p_values = np.asarray(
        selector.pvalues_,
        dtype=float,
    )

    selected_scores = all_scores[
        selector_support
    ]

    selected_p_values = all_p_values[
        selector_support
    ]

    classifier = (
        fitted_pipeline.named_steps[
            "classifier"
        ]
    )

    classifier_classes = list(
        classifier.classes_
    )

    if len(
        classifier_classes
    ) != 2:
        raise ValueError(
            "Binary logistic regression was "
            "expected, but classes were: "
            f"{classifier_classes}"
        )

    raw_coefficients = (
        classifier.coef_[0]
    )

    coefficient_target_class = (
        classifier_classes[1]
    )

    if coefficient_target_class == (
        POSITIVE_CLASS
    ):
        coefficients_for_low = (
            raw_coefficients
        )

    elif classifier_classes[0] == (
        POSITIVE_CLASS
    ):
        coefficients_for_low = (
            -raw_coefficients
        )

    else:
        raise ValueError(
            "The low class was not found in "
            f"classifier classes: {classifier_classes}"
        )

    if len(selected_names) != len(
        coefficients_for_low
    ):
        raise ValueError(
            "Selected-feature count does not match "
            "the logistic coefficient count."
        )

    safe_scores = np.nan_to_num(
        selected_scores,
        nan=-np.inf,
    )

    descending_order = np.argsort(
        -safe_scores
    )

    rank_by_position = np.empty(
        len(selected_names),
        dtype=int,
    )

    rank_by_position[
        descending_order
    ] = np.arange(
        1,
        len(selected_names) + 1,
    )

    rows = []

    for feature_position, feature_name in enumerate(
        selected_names
    ):
        coefficient = float(
            coefficients_for_low[
                feature_position
            ]
        )

        rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "outer_repeat": (
                    repeat_number
                ),
                "outer_fold": (
                    fold_number
                ),
                "outer_split": (
                    outer_split_number
                ),
                "best_k": best_k,
                "best_c": best_c,
                "n_input_features": (
                    len(original_features)
                ),
                "n_after_variance": (
                    len(after_variance)
                ),
                "n_selected_features": (
                    len(selected_names)
                ),
                "selected_rank_by_f_score": int(
                    rank_by_position[
                        feature_position
                    ]
                ),
                "feature": str(
                    feature_name
                ),
                "f_score_outer_training": (
                    selected_scores[
                        feature_position
                    ]
                ),
                "f_p_value_outer_training": (
                    selected_p_values[
                        feature_position
                    ]
                ),
                "coefficient_for_low_class": (
                    coefficient
                ),
                "absolute_coefficient": (
                    abs(coefficient)
                ),
                "coefficient_direction": (
                    "towards_low"
                    if coefficient > 0
                    else (
                        "towards_high"
                        if coefficient < 0
                        else "zero"
                    )
                ),
            }
        )

    return rows


# ============================================================
# Outer split preparation
# ============================================================

def build_outer_splits(
    y: pd.Series,
    outer_splits: int,
    outer_repeats: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate one reusable set of outer splits for every feature set."""
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=RANDOM_STATE,
    )

    placeholder_x = np.zeros(
        shape=(
            len(y),
            1,
        ),
        dtype=float,
    )

    splits = list(
        outer_cv.split(
            placeholder_x,
            y,
        )
    )

    expected_split_count = (
        outer_splits
        * outer_repeats
    )

    if len(splits) != expected_split_count:
        raise RuntimeError(
            "Unexpected outer split count: "
            f"expected {expected_split_count}, "
            f"observed {len(splits)}"
        )

    return splits


def split_identifiers(
    split_index: int,
    outer_splits: int,
) -> tuple[int, int]:
    """Convert a zero-based split index into repeat and fold numbers."""
    repeat_number = (
        split_index
        // outer_splits
        + 1
    )

    fold_number = (
        split_index
        % outer_splits
        + 1
    )

    return (
        repeat_number,
        fold_number,
    )


# ============================================================
# Main model evaluation
# ============================================================

def evaluate_feature_set(
    clear_table: pd.DataFrame,
    y: pd.Series,
    barriers: pd.Series,
    feature_set: str,
    features: list[str],
    outer_split_indices: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    outer_splits: int,
    inner_splits: int,
    n_jobs: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Run nested CV for one feature set.

    Returns
    -------
    metric_rows
    prediction_rows
    parameter_rows
    selected_feature_rows
    """
    X = clear_table[
        features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.shape[1] != len(features):
        raise RuntimeError(
            "Unexpected feature-matrix width."
        )

    parameter_grid = get_parameter_grid(
        feature_set=feature_set,
        n_input_features=X.shape[1],
    )

    metric_rows = []
    prediction_rows = []
    parameter_rows = []
    selected_feature_rows = []

    total_outer_splits = len(
        outer_split_indices
    )

    log()
    log("=" * 72)
    log(
        f"Feature set: {feature_set}"
    )
    log(
        f"Input features: {X.shape[1]}"
    )
    log(
        f"Outer models: {total_outer_splits}"
    )
    log(
        "k grid: "
        f"{parameter_grid['selector__k']}"
    )
    log(
        "C grid: "
        f"{parameter_grid['classifier__C']}"
    )
    log("=" * 72)

    for split_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_split_indices
    ):
        outer_split_number = (
            split_index + 1
        )

        (
            repeat_number,
            fold_number,
        ) = split_identifiers(
            split_index=split_index,
            outer_splits=outer_splits,
        )

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        barrier_test = barriers.iloc[
            test_indices
        ]

        inner_random_state = (
            RANDOM_STATE
            + outer_split_number
        )

        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=(
                inner_random_state
            ),
        )

        pipeline = (
            build_model_pipeline()
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score=np.nan,
        )

        log()
        log(
            f"[{feature_set}] "
            f"outer split "
            f"{outer_split_number}/"
            f"{total_outer_splits} "
            f"(repeat {repeat_number}, "
            f"fold {fold_number})"
        )

        log(
            "  train/test: "
            f"{len(train_indices)}/"
            f"{len(test_indices)}"
        )

        log(
            "  training counts: "
            f"{y_train.value_counts().to_dict()}"
        )

        fold_start_time = (
            time.perf_counter()
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )

            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
            )

            search.fit(
                X_train,
                y_train,
            )

        elapsed_seconds = (
            time.perf_counter()
            - fold_start_time
        )

        if not np.isfinite(
            search.best_score_
        ):
            raise RuntimeError(
                "No valid inner-CV model was found "
                f"for {feature_set}, outer split "
                f"{outer_split_number}."
            )

        best_model = (
            search.best_estimator_
        )

        best_k = search.best_params_[
            "selector__k"
        ]

        best_c = float(
            search.best_params_[
                "classifier__C"
            ]
        )

        y_pred = best_model.predict(
            X_test
        )

        probability_low = (
            positive_class_probability(
                fitted_model=best_model,
                X=X_test,
            )
        )

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            probability_low=(
                probability_low
            ),
        )

        variance_support = (
            best_model.named_steps[
                "variance"
            ].get_support()
        )

        n_after_variance = int(
            variance_support.sum()
        )

        selector_support = (
            best_model.named_steps[
                "selector"
            ].get_support()
        )

        n_selected = int(
            selector_support.sum()
        )

        metric_rows.append(
            {
                "model_type": (
                    "logistic_regression"
                ),
                "feature_set": (
                    feature_set
                ),
                "outer_repeat": (
                    repeat_number
                ),
                "outer_fold": (
                    fold_number
                ),
                "outer_split": (
                    outer_split_number
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
                "n_input_features": (
                    len(features)
                ),
                "n_after_variance": (
                    n_after_variance
                ),
                "n_selected_features": (
                    n_selected
                ),
                "best_k": best_k,
                "best_c": best_c,
                "best_inner_balanced_accuracy": (
                    search.best_score_
                ),
                "fit_seconds": (
                    elapsed_seconds
                ),
                **metrics,
            }
        )

        parameter_rows.append(
            {
                "feature_set": (
                    feature_set
                ),
                "outer_repeat": (
                    repeat_number
                ),
                "outer_fold": (
                    fold_number
                ),
                "outer_split": (
                    outer_split_number
                ),
                "n_input_features": (
                    len(features)
                ),
                "n_after_variance": (
                    n_after_variance
                ),
                "n_selected_features": (
                    n_selected
                ),
                "best_k": best_k,
                "best_c": best_c,
                "best_inner_balanced_accuracy": (
                    search.best_score_
                ),
                "inner_splits": (
                    inner_splits
                ),
                "inner_random_state": (
                    inner_random_state
                ),
                "fit_seconds": (
                    elapsed_seconds
                ),
            }
        )

        selected_feature_rows.extend(
            extract_selected_features(
                fitted_pipeline=best_model,
                original_features=features,
                feature_set=feature_set,
                repeat_number=(
                    repeat_number
                ),
                fold_number=(
                    fold_number
                ),
                outer_split_number=(
                    outer_split_number
                ),
                best_k=best_k,
                best_c=best_c,
            )
        )

        for test_position, ligand_id in enumerate(
            X_test.index
        ):
            prediction_rows.append(
                {
                    "model_type": (
                        "logistic_regression"
                    ),
                    "feature_set": (
                        feature_set
                    ),
                    "outer_repeat": (
                        repeat_number
                    ),
                    "outer_fold": (
                        fold_number
                    ),
                    "outer_split": (
                        outer_split_number
                    ),
                    "ligand_id": (
                        ligand_id
                    ),
                    "true_group": (
                        y_test.loc[
                            ligand_id
                        ]
                    ),
                    "predicted_group": (
                        y_pred[
                            test_position
                        ]
                    ),
                    "probability_low": (
                        probability_low[
                            test_position
                        ]
                    ),
                    "old_rf_predicted_barrier": (
                        barrier_test.loc[
                            ligand_id
                        ]
                    ),
                }
            )

        log(
            "  best k/C: "
            f"{best_k} / {best_c:g}"
        )

        log(
            "  inner BA: "
            f"{search.best_score_:.3f}"
        )

        log(
            "  outer BA/AUC/MCC: "
            f"{metrics['balanced_accuracy']:.3f} / "
            f"{metrics['roc_auc_low']:.3f} / "
            f"{metrics['mcc']:.3f}"
        )

        log(
            "  selected features: "
            f"{n_selected}"
        )

        log(
            "  elapsed: "
            f"{elapsed_seconds:.1f} s"
        )

    return (
        metric_rows,
        prediction_rows,
        parameter_rows,
        selected_feature_rows,
    )


# ============================================================
# Dummy baseline
# ============================================================

def evaluate_dummy_baseline(
    y: pd.Series,
    barriers: pd.Series,
    outer_split_indices: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    outer_splits: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate a prior-based dummy classifier on the same outer splits."""
    dummy_metric_rows = []
    dummy_prediction_rows = []

    placeholder_x = pd.DataFrame(
        {
            "dummy_input": np.zeros(
                len(y),
                dtype=float,
            )
        },
        index=y.index,
    )

    total_outer_splits = len(
        outer_split_indices
    )

    log()
    log("=" * 72)
    log(
        "Dummy baseline: prior strategy"
    )
    log("=" * 72)

    for split_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        outer_split_indices
    ):
        outer_split_number = (
            split_index + 1
        )

        (
            repeat_number,
            fold_number,
        ) = split_identifiers(
            split_index=split_index,
            outer_splits=outer_splits,
        )

        X_train = placeholder_x.iloc[
            train_indices
        ]

        X_test = placeholder_x.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        barrier_test = barriers.iloc[
            test_indices
        ]

        dummy = DummyClassifier(
            strategy="prior",
            random_state=RANDOM_STATE,
        )

        dummy.fit(
            X_train,
            y_train,
        )

        y_pred = dummy.predict(
            X_test
        )

        classes = list(
            dummy.classes_
        )

        positive_index = classes.index(
            POSITIVE_CLASS
        )

        probability_low = (
            dummy.predict_proba(
                X_test
            )[:, positive_index]
        )

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            probability_low=(
                probability_low
            ),
        )

        dummy_metric_rows.append(
            {
                "model_type": (
                    "dummy_prior"
                ),
                "feature_set": (
                    "dummy_prior"
                ),
                "outer_repeat": (
                    repeat_number
                ),
                "outer_fold": (
                    fold_number
                ),
                "outer_split": (
                    outer_split_number
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
                **metrics,
            }
        )

        for test_position, ligand_id in enumerate(
            X_test.index
        ):
            dummy_prediction_rows.append(
                {
                    "model_type": (
                        "dummy_prior"
                    ),
                    "feature_set": (
                        "dummy_prior"
                    ),
                    "outer_repeat": (
                        repeat_number
                    ),
                    "outer_fold": (
                        fold_number
                    ),
                    "outer_split": (
                        outer_split_number
                    ),
                    "ligand_id": (
                        ligand_id
                    ),
                    "true_group": (
                        y_test.loc[
                            ligand_id
                        ]
                    ),
                    "predicted_group": (
                        y_pred[
                            test_position
                        ]
                    ),
                    "probability_low": (
                        probability_low[
                            test_position
                        ]
                    ),
                    "old_rf_predicted_barrier": (
                        barrier_test.loc[
                            ligand_id
                        ]
                    ),
                }
            )

        log(
            f"Dummy split "
            f"{outer_split_number}/"
            f"{total_outer_splits}: "
            f"BA={metrics['balanced_accuracy']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f}"
        )

    return (
        dummy_metric_rows,
        dummy_prediction_rows,
    )


# ============================================================
# Output summaries and validation
# ============================================================

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc_low",
    "precision_low",
    "recall_low",
    "f1_low",
    "mcc",
]


def summarize_metrics(
    model_metrics: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise outer-fold performance using mean, SD and median."""
    combined_metrics = pd.concat(
        [
            model_metrics,
            dummy_metrics,
        ],
        ignore_index=True,
        sort=False,
    )

    grouped = (
        combined_metrics
        .groupby(
            [
                "model_type",
                "feature_set",
            ],
            dropna=False,
        )[METRIC_COLUMNS]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    grouped.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part)
        ).rstrip("_")
        if isinstance(
            column,
            tuple,
        )
        else column
        for column in grouped.columns
    ]

    return grouped


def build_feature_set_manifest_table(
    feature_sets: dict[
        str,
        list[str],
    ],
) -> pd.DataFrame:
    """Create a row-level record of the manifests used in this run."""
    rows = []

    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        features = feature_sets[
            feature_set
        ]

        for feature_order, feature in enumerate(
            features,
            start=1,
        ):
            rows.append(
                {
                    "feature_set": (
                        feature_set
                    ),
                    "feature_order": (
                        feature_order
                    ),
                    "feature": (
                        feature
                    ),
                    "n_features_in_set": (
                        len(features)
                    ),
                    "source_manifest": str(
                        FEATURE_SET_FILES[
                            feature_set
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def save_run_configuration(
    output_file: Path,
    mode: str,
    mode_configuration: dict[
        str,
        int,
    ],
    n_jobs: int,
    feature_sets: dict[
        str,
        list[str],
    ],
    elapsed_seconds: float,
) -> None:
    """Save software versions and all formal run settings."""
    settings: dict[str, Any] = {
        "batch": BATCH,
        "mode": mode,
        "master_file": str(
            MASTER_FILE
        ),
        "barrier_source": (
            "old RF-predicted barriers"
        ),
        "scientific_role": (
            "characterise old-model low/high "
            "structural behaviour"
        ),
        "n_total_samples": (
            EXPECTED_TOTAL_SAMPLES
        ),
        "n_low": (
            EXPECTED_LOW_COUNT
        ),
        "n_middle_excluded": (
            EXPECTED_MIDDLE_COUNT
        ),
        "n_high": (
            EXPECTED_HIGH_COUNT
        ),
        "n_clear_low_high": (
            EXPECTED_CLEAR_COUNT
        ),
        "positive_class": (
            POSITIVE_CLASS
        ),
        "outer_cv": (
            "RepeatedStratifiedKFold"
        ),
        "outer_splits": (
            mode_configuration[
                "outer_splits"
            ]
        ),
        "outer_repeats": (
            mode_configuration[
                "outer_repeats"
            ]
        ),
        "outer_total_splits": (
            mode_configuration[
                "outer_splits"
            ]
            * mode_configuration[
                "outer_repeats"
            ]
        ),
        "inner_cv": (
            "StratifiedKFold with shuffle"
        ),
        "inner_splits": (
            mode_configuration[
                "inner_splits"
            ]
        ),
        "inner_scoring": (
            "balanced_accuracy"
        ),
        "primary_outer_metric": (
            "balanced_accuracy"
        ),
        "secondary_outer_metrics": (
            "accuracy; roc_auc_low; "
            "precision_low; recall_low; "
            "f1_low; mcc"
        ),
        "pipeline_order": (
            "median_imputer; "
            "variance_threshold; "
            "select_k_best_f_classif; "
            "standard_scaler; "
            "balanced_logistic_regression"
        ),
        "logistic_solver": (
            "liblinear"
        ),
        "logistic_penalty": (
            "l2"
        ),
        "logistic_class_weight": (
            "balanced"
        ),
        "c_values": json.dumps(
            C_VALUES
        ),
        "k_values_by_set": json.dumps(
            K_VALUES_BY_SET
        ),
        "random_state": (
            RANDOM_STATE
        ),
        "n_jobs": n_jobs,
        "dummy_strategy": (
            "prior"
        ),
        "selection_uses_script_02_ranking": (
            False
        ),
        "preprocessing_outside_cv": (
            False
        ),
        "permutation_test_in_this_script": (
            False
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
        "elapsed_seconds": (
            elapsed_seconds
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


def validate_output_counts(
    mode_configuration: dict[
        str,
        int,
    ],
    model_metrics: pd.DataFrame,
    model_predictions: pd.DataFrame,
    best_parameters: pd.DataFrame,
    selected_features: pd.DataFrame,
    dummy_metrics: pd.DataFrame,
    dummy_predictions: pd.DataFrame,
) -> None:
    """Confirm the expected number and uniqueness of main output rows."""
    outer_total_splits = (
        mode_configuration[
            "outer_splits"
        ]
        * mode_configuration[
            "outer_repeats"
        ]
    )

    expected_model_metric_rows = (
        len(
            FORMAL_FEATURE_SETS
        )
        * outer_total_splits
    )

    expected_model_prediction_rows = (
        len(
            FORMAL_FEATURE_SETS
        )
        * mode_configuration[
            "outer_repeats"
        ]
        * EXPECTED_CLEAR_COUNT
    )

    expected_dummy_metric_rows = (
        outer_total_splits
    )

    expected_dummy_prediction_rows = (
        mode_configuration[
            "outer_repeats"
        ]
        * EXPECTED_CLEAR_COUNT
    )

    observed_counts = {
        "model metric rows": (
            len(model_metrics)
        ),
        "best-parameter rows": (
            len(best_parameters)
        ),
        "model prediction rows": (
            len(model_predictions)
        ),
        "dummy metric rows": (
            len(dummy_metrics)
        ),
        "dummy prediction rows": (
            len(dummy_predictions)
        ),
    }

    expected_counts = {
        "model metric rows": (
            expected_model_metric_rows
        ),
        "best-parameter rows": (
            expected_model_metric_rows
        ),
        "model prediction rows": (
            expected_model_prediction_rows
        ),
        "dummy metric rows": (
            expected_dummy_metric_rows
        ),
        "dummy prediction rows": (
            expected_dummy_prediction_rows
        ),
    }

    for count_name, expected_count in (
        expected_counts.items()
    ):
        observed_count = (
            observed_counts[
                count_name
            ]
        )

        if observed_count != expected_count:
            raise RuntimeError(
                f"Unexpected {count_name}: "
                f"expected {expected_count}, "
                f"observed {observed_count}"
            )

    if selected_features.empty:
        raise RuntimeError(
            "No selected features were saved."
        )

    model_prediction_duplicates = (
        model_predictions.duplicated(
            subset=[
                "feature_set",
                "outer_repeat",
                "ligand_id",
            ]
        )
    )

    if model_prediction_duplicates.any():
        raise RuntimeError(
            "A ligand appeared more than once "
            "within the same feature-set/repeat "
            "outer-test predictions."
        )

    dummy_prediction_duplicates = (
        dummy_predictions.duplicated(
            subset=[
                "outer_repeat",
                "ligand_id",
            ]
        )
    )

    if dummy_prediction_duplicates.any():
        raise RuntimeError(
            "A ligand appeared more than once "
            "within the same repeat of dummy "
            "outer-test predictions."
        )


def ensure_output_directory(
    output_directory: Path,
    overwrite: bool,
) -> None:
    """Create the output directory and protect existing formal results."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    protected_files = [
        output_directory
        / "nested_cv_fold_metrics.tsv",
        output_directory
        / "nested_cv_predictions.tsv",
        output_directory
        / "nested_cv_best_parameters.tsv",
        output_directory
        / "nested_cv_selected_features.tsv",
        output_directory
        / "nested_cv_dummy_metrics.tsv",
        output_directory
        / "nested_cv_dummy_predictions.tsv",
        output_directory
        / "nested_cv_summary.tsv",
        output_directory
        / "run_configuration.tsv",
        output_directory
        / "feature_set_manifest.tsv",
    ]

    existing_files = [
        file_path
        for file_path in protected_files
        if file_path.exists()
    ]

    if existing_files and not overwrite:
        raise FileExistsError(
            "Output files already exist. "
            "Use --overwrite only if rerunning "
            "this mode intentionally:\n"
            + "\n".join(
                str(file_path)
                for file_path in existing_files
            )
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run debug or final four-set nested cross-validation."""
    arguments = parse_arguments()

    mode_configuration = (
        get_mode_configuration(
            arguments.mode
        )
    )

    output_directory = (
        NESTED_CV_ROOT
        / arguments.mode
    )

    ensure_output_directory(
        output_directory=(
            output_directory
        ),
        overwrite=arguments.overwrite,
    )

    overall_start_time = (
        time.perf_counter()
    )

    log(
        f"Starting {arguments.mode} "
        f"nested CV for {BATCH}..."
    )

    log(
        "Output directory:"
    )

    log(
        str(output_directory)
    )

    log()
    log(
        "Cross-validation configuration:"
    )

    log(
        "  outer folds: "
        f"{mode_configuration['outer_splits']}"
    )

    log(
        "  outer repeats: "
        f"{mode_configuration['outer_repeats']}"
    )

    log(
        "  inner folds: "
        f"{mode_configuration['inner_splits']}"
    )

    log(
        "  GridSearchCV n_jobs: "
        f"{arguments.n_jobs}"
    )

    validate_input_paths()

    feature_sets = (
        load_feature_sets()
    )

    (
        clear_table,
        y,
        barriers,
    ) = load_analysis_dataset(
        feature_sets=feature_sets
    )

    log()
    log(
        "Clear low/high dataset:"
    )

    log(
        f"  samples: {len(y)}"
    )

    log(
        f"  class counts: "
        f"{y.value_counts().to_dict()}"
    )

    log()
    log(
        "Formal feature sets:"
    )

    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        log(
            f"  {feature_set}: "
            f"{len(feature_sets[feature_set])}"
        )

    outer_split_indices = (
        build_outer_splits(
            y=y,
            outer_splits=(
                mode_configuration[
                    "outer_splits"
                ]
            ),
            outer_repeats=(
                mode_configuration[
                    "outer_repeats"
                ]
            ),
        )
    )

    all_metric_rows = []
    all_prediction_rows = []
    all_parameter_rows = []
    all_selected_feature_rows = []

    for feature_set in (
        FORMAL_FEATURE_SETS
    ):
        (
            metric_rows,
            prediction_rows,
            parameter_rows,
            selected_feature_rows,
        ) = evaluate_feature_set(
            clear_table=clear_table,
            y=y,
            barriers=barriers,
            feature_set=feature_set,
            features=feature_sets[
                feature_set
            ],
            outer_split_indices=(
                outer_split_indices
            ),
            outer_splits=(
                mode_configuration[
                    "outer_splits"
                ]
            ),
            inner_splits=(
                mode_configuration[
                    "inner_splits"
                ]
            ),
            n_jobs=arguments.n_jobs,
        )

        all_metric_rows.extend(
            metric_rows
        )

        all_prediction_rows.extend(
            prediction_rows
        )

        all_parameter_rows.extend(
            parameter_rows
        )

        all_selected_feature_rows.extend(
            selected_feature_rows
        )

    (
        dummy_metric_rows,
        dummy_prediction_rows,
    ) = evaluate_dummy_baseline(
        y=y,
        barriers=barriers,
        outer_split_indices=(
            outer_split_indices
        ),
        outer_splits=(
            mode_configuration[
                "outer_splits"
            ]
        ),
    )

    model_metrics = pd.DataFrame(
        all_metric_rows
    )

    model_predictions = pd.DataFrame(
        all_prediction_rows
    )

    best_parameters = pd.DataFrame(
        all_parameter_rows
    )

    selected_features = pd.DataFrame(
        all_selected_feature_rows
    )

    dummy_metrics = pd.DataFrame(
        dummy_metric_rows
    )

    dummy_predictions = pd.DataFrame(
        dummy_prediction_rows
    )

    validate_output_counts(
        mode_configuration=(
            mode_configuration
        ),
        model_metrics=model_metrics,
        model_predictions=(
            model_predictions
        ),
        best_parameters=(
            best_parameters
        ),
        selected_features=(
            selected_features
        ),
        dummy_metrics=dummy_metrics,
        dummy_predictions=(
            dummy_predictions
        ),
    )

    summary = summarize_metrics(
        model_metrics=model_metrics,
        dummy_metrics=dummy_metrics,
    )

    feature_set_manifest = (
        build_feature_set_manifest_table(
            feature_sets
        )
    )

    model_metrics.to_csv(
        output_directory
        / "nested_cv_fold_metrics.tsv",
        sep="\t",
        index=False,
    )

    model_predictions.to_csv(
        output_directory
        / "nested_cv_predictions.tsv",
        sep="\t",
        index=False,
    )

    best_parameters.to_csv(
        output_directory
        / "nested_cv_best_parameters.tsv",
        sep="\t",
        index=False,
    )

    selected_features.to_csv(
        output_directory
        / "nested_cv_selected_features.tsv",
        sep="\t",
        index=False,
    )

    dummy_metrics.to_csv(
        output_directory
        / "nested_cv_dummy_metrics.tsv",
        sep="\t",
        index=False,
    )

    dummy_predictions.to_csv(
        output_directory
        / "nested_cv_dummy_predictions.tsv",
        sep="\t",
        index=False,
    )

    summary.to_csv(
        output_directory
        / "nested_cv_summary.tsv",
        sep="\t",
        index=False,
    )

    feature_set_manifest.to_csv(
        output_directory
        / "feature_set_manifest.tsv",
        sep="\t",
        index=False,
    )

    total_elapsed_seconds = (
        time.perf_counter()
        - overall_start_time
    )

    save_run_configuration(
        output_file=(
            output_directory
            / "run_configuration.tsv"
        ),
        mode=arguments.mode,
        mode_configuration=(
            mode_configuration
        ),
        n_jobs=arguments.n_jobs,
        feature_sets=feature_sets,
        elapsed_seconds=(
            total_elapsed_seconds
        ),
    )

    log()
    log("=" * 72)
    log(
        "Nested-CV summary"
    )
    log("=" * 72)

    summary_columns = [
        "model_type",
        "feature_set",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "roc_auc_low_mean",
        "roc_auc_low_std",
        "mcc_mean",
        "mcc_std",
        "recall_low_mean",
        "precision_low_mean",
    ]

    available_summary_columns = [
        column
        for column in summary_columns
        if column in summary.columns
    ]

    log(
        summary[
            available_summary_columns
        ].to_string(
            index=False
        )
    )

    log()
    log(
        "Output row counts:"
    )

    log(
        "  model metrics: "
        f"{len(model_metrics)}"
    )

    log(
        "  model predictions: "
        f"{len(model_predictions)}"
    )

    log(
        "  best parameters: "
        f"{len(best_parameters)}"
    )

    log(
        "  selected-feature rows: "
        f"{len(selected_features)}"
    )

    log(
        "  dummy metrics: "
        f"{len(dummy_metrics)}"
    )

    log(
        "  dummy predictions: "
        f"{len(dummy_predictions)}"
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