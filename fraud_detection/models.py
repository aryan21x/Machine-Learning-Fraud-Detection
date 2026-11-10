"""Construct fraud classifiers with preprocessing fitted on training data."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud_detection.features import CATEGORICAL_COLUMNS, build_features

MODEL_NAMES = ("logistic_regression", "random_forest")


def build_pipeline(model_name: str = "logistic_regression", *, seed: int = 42) -> Pipeline:
    """Use the same inputs for both models; only Logistic Regression needs scaling."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed
        )
    elif model_name == "random_forest":
        # Bound tree size and parallelism for a laptop with 16 GB RAM.
        # balanced_subsample gives fraud more weight within each bootstrap sample.
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler() if model_name == "logistic_regression" else "passthrough",
                ["log_amount"],
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_COLUMNS)),
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
