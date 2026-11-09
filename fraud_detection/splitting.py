"""Chronological splitting that keeps identical transaction times together."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame, *, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> TemporalSplit:
    """Fractions apply to distinct TransactionDT values, not rows or elapsed time."""
    if not (
        np.isfinite(train_fraction)
        and np.isfinite(validation_fraction)
        and 0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and sum to less than 1.")
    times = np.sort(frame["TransactionDT"].unique())
    train_end = int(len(times) * train_fraction)
    validation_end = int(len(times) * (train_fraction + validation_fraction))
    if not 0 < train_end < validation_end < len(times):
        raise ValueError("Too few distinct transaction times for three non-empty partitions.")

    # Stable sorting preserves source order when transactions have the same time.
    ordered = frame.sort_values("TransactionDT", kind="stable")
    first_validation_time = times[train_end]
    first_test_time = times[validation_end]
    return TemporalSplit(
        train=ordered.loc[ordered["TransactionDT"] < first_validation_time].copy(),
        validation=ordered.loc[
            (ordered["TransactionDT"] >= first_validation_time)
            & (ordered["TransactionDT"] < first_test_time)
        ].copy(),
        test=ordered.loc[ordered["TransactionDT"] >= first_test_time].copy(),
    )
