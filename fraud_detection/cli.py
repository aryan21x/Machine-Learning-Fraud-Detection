"""Command-line entry points. No database or external services required."""

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame

LOGGER = logging.getLogger(__name__)


def _read_columns(path: Path, columns: tuple[str, ...]) -> "DataFrame":
    """Read selected fields without loading hundreds of unused IEEE-CIS columns."""
    import pandas as pd

    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as parquet

        available = parquet.read_schema(path).names
        return pd.read_parquet(
            path, columns=[column for column in columns if column in available], engine="pyarrow"
        )
    return pd.read_csv(path, usecols=lambda column: column in columns)


def load_transactions(path: str | Path) -> "DataFrame":
    """Read IEEE-CIS transactions and add DeviceType from the matching identity file."""
    import pandas as pd

    from fraud_detection.features import CATEGORICAL_COLUMNS

    path = Path(path)
    columns = (
        "TransactionID", "TransactionDT", "TransactionAmt", "isFraud", *CATEGORICAL_COLUMNS
    )
    frame = _read_columns(path, columns)
    frame["TransactionID"] = frame["TransactionID"].astype("int64")
    frame["TransactionDT"] = frame["TransactionDT"].astype("int64")
    frame["TransactionAmt"] = frame["TransactionAmt"].astype("float64")
    if "isFraud" in frame:
        frame["isFraud"] = frame["isFraud"].astype("int8")

    # A saved prediction batch already contains DeviceType and can be scored again.
    if "DeviceType" not in frame:
        identity_path = path.with_name(path.name.replace("_transaction", "_identity"))
        if identity_path != path and identity_path.is_file():
            identity = _read_columns(identity_path, ("TransactionID", "DeviceType"))
            # Keep transactions without identity matches; prevent duplicate join keys
            # from multiplying transactions and their labels.
            frame = frame.merge(
                identity, on="TransactionID", how="left", sort=False, validate="many_to_one"
            )
            LOGGER.info("Loaded DeviceType from %s", identity_path)
        else:
            frame["DeviceType"] = pd.NA
            LOGGER.info("No matching identity file; DeviceType will be missing")
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
            f"  {self.prog} train\n\n"
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
        usage="%(prog)s [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train Logistic Regression and save the model and results in artifacts/.",
        epilog="Example: %(prog)s",
    )
    train.add_argument(
        "--input", type=Path, metavar="FILE",
        default=Path("data/ieee-fraud-detection/train_transaction.csv"),
        help="Optional transaction file (default: %(default)s)",
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
        help="Share of distinct time values for training (default: 0.70)",
    )
    train.add_argument(
        "--validation-fraction", type=float, default=0.15,
        help="Share of distinct time values for validation (default: 0.15)",
    )

    predict = commands.add_parser(
        "predict",
        usage="%(prog)s [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Print prediction findings and save individual predictions in reports/.",
        epilog="Example: %(prog)s",
    )
    predict.add_argument(
        "--model", type=Path, metavar="FILE",
        help="Optional model path; defaults to the latest completed run in artifacts/",
    )
    predict.add_argument(
        "--input", type=Path, metavar="FILE",
        default=Path("data/ieee-fraud-detection/train_transaction.csv"),
        help="Optional transaction file (default: %(default)s)",
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
        if args.command == "predict":
            from fraud_detection.training import latest_model_path

            args.model = args.model if args.model is not None else latest_model_path()
            if not args.model.is_file():
                raise FileNotFoundError(
                    f"Model not found: {args.model}. Use its full path, such as "
                    "artifacts/run1/logistic_regression.joblib, or omit --model "
                    "to use the latest completed run."
                )
            LOGGER.info("Using model %s", args.model)
        LOGGER.info("Loading %s", args.input)
        try:
            frame = load_transactions(args.input)
        except KeyError as exc:
            LOGGER.error(
                "Missing transaction column %s. Use a transaction file as input; "
                "the matching identity file is joined automatically.", exc,
            )
            return 2
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
