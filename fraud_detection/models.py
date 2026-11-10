"""Construct fraud classifiers with preprocessing fitted on training data."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from xgboost import XGBClassifier

from fraud_detection.features import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    NULLABLE_NUMERIC_COLUMNS,
    build_features,
)

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")


def build_pipeline(model_name: str = "logistic_regression", *, seed: int = 42) -> Pipeline:
    """Use the same inputs for all models; only Logistic Regression needs scaling."""
    categorical: OneHotEncoder | OrdinalEncoder = OneHotEncoder(handle_unknown="ignore")
    if model_name == "logistic_regression":
        # The expanded numeric feature set can need more optimizer steps.
        # This is a ceiling: fitting stops sooner when convergence is reached.
        estimator = LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=seed
        )
    elif model_name == "random_forest":
        # Bound tree size and parallelism for a laptop with 16 GB RAM.
        # balanced_subsample gives fraud more weight within each bootstrap sample.
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        )
    elif model_name == "xgboost":
        # Native category splits keep the input compact and numeric zeros explicit.
        # Codes are category labels, not numeric ranks; unseen labels become NaN.
        categorical = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=float("nan")
        )
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            tree_method="hist",
            device="cpu",
            enable_categorical=True,
            feature_types=["q"] * len(NUMERIC_COLUMNS) + ["c"] * len(CATEGORICAL_COLUMNS),
            n_jobs=2,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    nullable_columns = list(NULLABLE_NUMERIC_COLUMNS)
    nullable_numeric: str | Pipeline = "passthrough"
    if model_name != "xgboost":
        # Fit D/distance/V replacements and missing flags on training data. Amount and
        # C inputs retain their existing preparation; XGBoost keeps native NaNs.
        nullable_steps = [
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
        ]
        if model_name == "logistic_regression":
            nullable_steps.append(("scale", StandardScaler()))
        nullable_numeric = Pipeline(nullable_steps)

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler() if model_name == "logistic_regression" else "passthrough",
                [column for column in NUMERIC_COLUMNS if column not in nullable_columns],
            ),
            ("nullable_numeric", nullable_numeric, nullable_columns),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, validate=False)),
            ("preprocess", preprocessing),
            ("classifier", estimator),
        ]
    )
