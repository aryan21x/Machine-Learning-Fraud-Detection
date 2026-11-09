"""Command-line entry points. No database or external services required."""

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame

LOGGER = logging.getLogger(__name__)


def load_transactions(path: str | Path) -> "DataFrame":
    """Read CSV or Parquet and convert the fields used by the models."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, engine="pyarrow")
    else:
        frame = pd.read_csv(
            path,
            dtype={"nameOrig": "string", "nameDest": "string", "type": "string"},
        )
    frame["step"] = frame["step"].astype("int64")
    frame["amount"] = frame["amount"].astype("float64")
    frame["type"] = frame["type"].astype("category")
    if "isFraud" in frame:
        frame["isFraud"] = frame["isFraud"].astype("int8")
    return frame


class _MainParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return (
            "Fraud detection\n\n"
            f"Run: {self.prog} COMMAND [options]\n\n"
            "Commands:\n"
            "  train     Train Logistic Regression and measure results\n"
            "  predict   Make predictions using a saved model\n\n"
            "Train your model:\n"
            f"  {self.prog} train --input data/FinancialData.csv\n\n"
            "More help:\n"
            f"  {self.prog} train --help\n"
        )


def _print_prediction_summary(predictions: "DataFrame") -> None:
    flagged = predictions["prediction"].eq(1)
    total = len(predictions)
    flagged_count = int(flagged.sum())
    print("\nPrediction summary for this input file")
    print(f"  Transactions scored: {total:,}")
    print(f"  Flagged as fraud: {flagged_count:,}")
    print(f"  Not flagged: {total - flagged_count:,}")

    if "isFraud" not in predictions:
        print("  Actual fraud labels unavailable; error counts cannot be calculated.")
        return

    actual_fraud = predictions["isFraud"].eq(1)
    caught = int((flagged & actual_fraud).sum())
    false_alarms = flagged_count - caught
    missed = int((~flagged & actual_fraud).sum())
    print("\nCompared with known fraud labels:")
    print(f"  Fraud caught (true positives): {caught:,}")
    print(f"  False alarms (false positives): {false_alarms:,}")
    print(f"  Missed fraud (false negatives): {missed:,}")


def _parser() -> argparse.ArgumentParser:
    parser = _MainParser(
        prog="python -m fraud_detection",
        usage="%(prog)s COMMAND [options]",
    )
    commands = parser.add_subparsers(
        dest="command", required=True, prog=parser.prog, parser_class=argparse.ArgumentParser
    )
    train = commands.add_parser(
        "train",
        usage="%(prog)s --input FILE [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train Logistic Regression and save the model and results in artifacts/.",
        epilog="Example: %(prog)s --input data/FinancialData.csv",
    )
    train.add_argument(
        "--input", required=True, type=Path, metavar="FILE", help="Labeled transaction file"
    )
    train.add_argument(
        "--output", type=Path, metavar="FOLDER",
        help="Optional results folder; otherwise saves to a new folder in artifacts/",
    )
    train.add_argument(
        "--seed", type=int, default=os.environ.get("FRAUD_SEED", "42"),
        help="Random seed (default: %(default)s)",
    )
    train.add_argument(
        "--train-fraction", type=float, default=0.70,
        help="Share of time steps for training (default: 0.70)",
    )
    train.add_argument(
        "--validation-fraction", type=float, default=0.15,
        help="Share of time steps for validation (default: 0.15)",
    )

    predict = commands.add_parser(
        "predict",
        usage="%(prog)s --model FILE --input FILE [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Print prediction findings and save individual predictions in reports/.",
        epilog=(
            "Example: %(prog)s --model artifacts/run1/logistic_regression.joblib "
            "--input data/FinancialData.csv"
        ),
    )
    predict.add_argument(
        "--model", required=True, type=Path, metavar="FILE", help="Saved model (.joblib)"
    )
    predict.add_argument(
        "--input", required=True, type=Path, metavar="FILE", help="Transaction file"
    )
    predict.add_argument(
        "--output", type=Path, metavar="FILE",
        help="Optional CSV path; otherwise saves a new file in reports/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        LOGGER.info("Loading %s", args.input)
        frame = load_transactions(args.input)
        if args.command == "train":
            from fraud_detection.training import TrainingConfig, run_training

            report = run_training(
                frame,
                args.input,
                args.output,
                config=TrainingConfig(args.seed, args.train_fraction, args.validation_fraction),
            )
            print(f"Training complete. Model: {report['preferred_model']}")
            print(f"Saved to: {report['output_directory']}")
        else:
            from fraud_detection.training import predict_batch

            output, predictions = predict_batch(args.model, frame, args.output)
            _print_prediction_summary(predictions)
            print(f"Predictions saved to: {output}")
    except (OSError, ValueError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
