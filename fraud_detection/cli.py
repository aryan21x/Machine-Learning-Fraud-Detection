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
    """Read named input columns from CSV or Parquet."""
    import pandas as pd

    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as parquet

        available = parquet.read_schema(path).names
        return pd.read_parquet(
            path, columns=[column for column in columns if column in available], engine="pyarrow"
        )
    return pd.read_csv(path, usecols=lambda column: column in columns)


def load_transactions(path: str | Path) -> "DataFrame":
    """Read transactions and join DeviceType from the matching identity file."""
    import pandas as pd

    from fraud_detection.features import FEATURE_COLUMNS

    path = Path(path)
    columns = ("TransactionID", "isFraud", *FEATURE_COLUMNS)
    frame = _read_columns(path, columns)
    frame["TransactionID"] = frame["TransactionID"].astype("int64")
    frame["TransactionDT"] = frame["TransactionDT"].astype("int64")
    frame["TransactionAmt"] = frame["TransactionAmt"].astype("float64")
    if "isFraud" in frame:
        frame["isFraud"] = frame["isFraud"].astype("int8")

    # Join DeviceType only when it is not already supplied in the input.
    if "DeviceType" not in frame:
        identity_path = path.with_name(path.name.replace("_transaction", "_identity"))
        if identity_path != path and identity_path.is_file():
            identity = _read_columns(identity_path, ("TransactionID", "DeviceType"))
            # Keep transactions without identity matches; prevent duplicate join keys
            # from multiplying transactions and their labels.
            frame = frame.merge(
                identity, on="TransactionID", how="left", sort=False, validate="many_to_one"
            )
            LOGGER.info("Joined DeviceType from %s", identity_path)
        else:
            LOGGER.info("No matching identity file; DeviceType will be missing")
        if "DeviceType" not in frame:
            frame["DeviceType"] = pd.NA
    return frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fraud_detection",
        usage="%(prog)s [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compare Logistic Regression, Random Forest, and XGBoost; save results in artifacts/.",
        epilog="Example: %(prog)s",
    )
    parser.add_argument(
        "--input", type=Path, metavar="FILE",
        default=Path("data/ieee-fraud-detection/train_transaction.csv"),
        help="Optional transaction file (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, metavar="FOLDER",
        help="Optional results folder (default: artifacts/); replaces existing generated files",
    )
    parser.add_argument(
        "--seed", type=int, default=os.environ.get("FRAUD_SEED", "42"),
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.70,
        help="Share of distinct time values for training (default: 0.70)",
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=0.15,
        help="Share of distinct time values for validation (default: 0.15)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        LOGGER.info("Loading %s", args.input)
        try:
            frame = load_transactions(args.input)
        except KeyError as exc:
            LOGGER.error(
                "Missing transaction column %s. Use a transaction file as input; "
                "the matching identity file is joined automatically.", exc,
            )
            return 2
        from fraud_detection.training import TrainingConfig, run_training

        report = run_training(
            frame,
            args.input,
            args.output,
            config=TrainingConfig(args.seed, args.train_fraction, args.validation_fraction),
        )
        print("\nHeld-out test results")
        print(f"{'Model':<22} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AP':>9}")
        for name, results in report["models"].items():
            values = [
                "n/a" if results["test_data_only"][field] is None
                else f"{results['test_data_only'][field]:.4f}"
                for field in ("precision", "recall", "f1", "average_precision")
            ]
            print(f"{name:<22} " + " ".join(f"{value:>9}" for value in values))
        print(f"Best validation model: {report['preferred_model']}")
        print(f"Saved to: {report['output_directory']}")
    except (OSError, ValueError, ImportError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
