"""Train, evaluate, persist, and reload the minimum working ML system."""

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
from fraud_detection.models import MODEL_NAME, build_pipeline
from fraud_detection.splitting import chronological_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15


def _create_run_directory() -> Path:
    """Create artifacts/run1, run2, etc. without reusing an existing folder."""
    root = Path("artifacts")
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        directory = root / f"run{number}"
        try:
            directory.mkdir()
        except FileExistsError:
            number += 1
        else:
            return directory


def _test_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "precision", "recall", "f1", "average_precision", "roc_auc",
        "true_positives", "false_positives", "false_negatives",
    )
    return {
        field: round(metrics[field], 4) if isinstance(metrics[field], float) else metrics[field]
        for field in fields
    }


def _prediction_frame(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    columns = ["step", "type", "amount"]
    columns.extend(column for column in ("nameOrig", "nameDest", "isFraud") if column in frame)
    result = frame[columns].copy()
    result.insert(0, "source_row", frame.index)
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
    """Train Logistic Regression and choose its threshold using validation data.

    By default, save to a new artifacts/runN folder. An explicit nonempty output
    directory fails rather than overwriting files. report.json marks completion.
    """
    config = config or TrainingConfig()
    source = Path(source)
    output = Path(output) if output is not None else None
    if output is not None and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Run output is not empty: {output}. Choose a new run directory.")

    split = chronological_split(
        frame,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    report: dict[str, Any] = {
        "report_version": 3,
        "dataset": source.name,
        "config": asdict(config),
        "models": {},
    }
    if output is None:
        output = _create_run_directory()
    else:
        output.mkdir(parents=True, exist_ok=True)
    report["output_directory"] = str(output)

    LOGGER.info("Fitting %s on training only", MODEL_NAME)
    model = build_pipeline(seed=config.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(split.train, split.train["isFraud"])
        except ConvergenceWarning as exc:
            raise ValueError(
                "Model did not converge; review scaling and training settings."
            ) from exc

    scores = model.predict_proba(split.validation)[:, 1]
    # Pick the threshold with the highest validation F1 (precision/recall balance).
    # Use that same threshold for test evaluation and future predictions.
    threshold = select_threshold(split.validation["isFraud"], scores)
    validation = evaluate_scores(split.validation["isFraud"], scores, threshold)
    test_scores = model.predict_proba(split.test)[:, 1]
    report["preferred_model"] = MODEL_NAME
    report["models"][MODEL_NAME] = {
        "selected_threshold": threshold,
        "validation": {
            "average_precision": round(validation["average_precision"], 4),
            "f1": round(validation["f1"], 4),
            "false_positives": validation["false_positives"],
            "false_negatives": validation["false_negatives"],
        },
        "test": _test_summary(evaluate_scores(split.test["isFraud"], test_scores, threshold)),
    }
    bundle = {"pipeline": model, "threshold": threshold}
    joblib.dump(bundle, output / f"{MODEL_NAME}.joblib", compress=3)
    _prediction_frame(split.test, test_scores, threshold).to_csv(
        output / f"{MODEL_NAME}_test_predictions.csv", index=False
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Completed training in %s", output)
    return report


def load_model(path: str | Path) -> dict[str, Any]:
    """Load the saved pipeline and threshold from a local joblib file."""
    return joblib.load(path)


def predict_batch(
    model_path: str | Path, frame: pd.DataFrame, output: str | Path | None = None
) -> tuple[Path, pd.DataFrame]:
    """Save predictions and return their CSV path and scored transactions."""
    output = Path(output) if output is not None else None
    bundle = load_model(model_path)
    scores = bundle["pipeline"].predict_proba(frame)[:, 1]
    predictions = _prediction_frame(frame, scores, bundle["threshold"])
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output, index=False, mode="x")
        return output, predictions

    directory = Path("reports")
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        filename = "predictions.csv" if number == 1 else f"predictions{number}.csv"
        path = directory / filename
        try:
            predictions.to_csv(path, index=False, mode="x")
        except FileExistsError:
            number += 1
        else:
            return path, predictions
