"""Train, evaluate, and save classifiers with held-out predictions and results."""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from fraud_detection.evaluation import evaluate_scores, select_threshold
from fraud_detection.features import CATEGORICAL_COLUMNS
from fraud_detection.models import MODEL_NAMES, build_pipeline
from fraud_detection.splitting import chronological_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "precision", "recall", "f1", "average_precision", "roc_auc",
        "true_positives", "false_positives", "false_negatives",
    )
    return {
        field: round(metrics[field], 4) if isinstance(metrics[field], float) else metrics[field]
        for field in fields
    }


def _prediction_frame(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    # Keep selected inputs alongside test predictions for inspecting model errors.
    columns = ["TransactionID", "TransactionDT", "TransactionAmt", *CATEGORICAL_COLUMNS]
    if "isFraud" in frame:
        columns.append("isFraud")
    result = frame[columns].copy()
    result["fraud_score"] = scores
    result["threshold"] = threshold
    result["prediction"] = (scores >= threshold).astype("int8")
    if "isFraud" in result:
        result["outcome"] = np.where(
            result["prediction"] == result["isFraud"],
            "correct",
            np.where(result["prediction"] == 1, "false_positive", "false_negative"),
        )
    return result


def run_training(
    frame: pd.DataFrame,
    source: str | Path,
    output: str | Path | None = None,
    *,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Compare classifiers and choose their thresholds using validation data.

    Save directly to artifacts/ by default, replacing the existing model files,
    report, and test-prediction CSVs on each execution.
    """
    config = config or TrainingConfig()
    source = Path(source)
    output = Path(output) if output is not None else Path("artifacts")

    split = chronological_split(
        frame,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    report: dict[str, Any] = {
        "report_version": 4,
        "dataset": source.name,
        "config": asdict(config),
        "models": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    report["output_directory"] = str(output)

    validation_ranking: dict[str, tuple[float, float]] = {}
    for model_name in MODEL_NAMES:
        LOGGER.info("Fitting %s on training only", model_name)
        model = build_pipeline(model_name, seed=config.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(split.train, split.train["isFraud"])
            except ConvergenceWarning as exc:
                raise ValueError(
                    f"{model_name} did not converge; review scaling and training settings."
                ) from exc

        scores = model.predict_proba(split.validation)[:, 1]
        # Pick this model's threshold with the highest validation F1.
        # Reuse it unchanged for test evaluation and future predictions.
        threshold = select_threshold(split.validation["isFraud"], scores)
        validation = evaluate_scores(split.validation["isFraud"], scores, threshold)
        validation_ranking[model_name] = (validation["f1"], validation["average_precision"])
        test_scores = model.predict_proba(split.test)[:, 1]
        # Score all input rows with the same trained model and validation threshold.
        # This includes training rows; it is separate from held-out performance.
        full_scores = model.predict_proba(frame)[:, 1]
        report["models"][model_name] = {
            "selected_threshold": threshold,
            "validation": {
                "average_precision": round(validation["average_precision"], 4),
                "f1": round(validation["f1"], 4),
                "false_positives": validation["false_positives"],
                "false_negatives": validation["false_negatives"],
            },
            "test_data_only": _metrics_summary(
                evaluate_scores(split.test["isFraud"], test_scores, threshold)
            ),
            "full_run": {
                "includes_training_data": True,
                **_metrics_summary(evaluate_scores(frame["isFraud"], full_scores, threshold)),
            },
        }
        bundle = {"pipeline": model, "threshold": threshold}
        joblib.dump(bundle, output / f"{model_name}.joblib", compress=3)
        _prediction_frame(split.test, test_scores, threshold).to_csv(
            output / f"{model_name}_test_predictions.csv", index=False
        )

    # Report the validation winner using unrounded F1, then AP to break ties.
    # Test metrics never influence the model choice.
    report["preferred_model"] = max(validation_ranking, key=lambda name: validation_ranking[name])
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Completed training in %s", output)
    return report
