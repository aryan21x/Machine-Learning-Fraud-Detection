"""Construct Logistic Regression with its preprocessing in a Pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud_detection.features import build_features

MODEL_NAME = "logistic_regression"


def build_pipeline(*, seed: int = 42) -> Pipeline:
    """Create an unfitted Logistic Regression pipeline."""
    estimator = LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed
    )

    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["log_amount"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["type"]),
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
