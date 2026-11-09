"""Chronological splitting that keeps tied time steps in one partition."""

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
    """Fractions apply to observed distinct steps, not rows or elapsed clock time."""
    if not (
        np.isfinite(train_fraction)
        and np.isfinite(validation_fraction)
        and 0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and sum to less than 1.")
    steps = np.sort(frame["step"].unique())
    train_end = int(len(steps) * train_fraction)
    validation_end = int(len(steps) * (train_fraction + validation_fraction))
    if not 0 < train_end < validation_end < len(steps):
        raise ValueError("Too few distinct steps for three non-empty time partitions.")

    # Stable order preserves source order within a step without assuming event chronology.
    ordered = frame.sort_values("step", kind="stable")
    first_validation_step = steps[train_end]
    first_test_step = steps[validation_end]
    return TemporalSplit(
        train=ordered.loc[ordered["step"] < first_validation_step].copy(),
        validation=ordered.loc[
            (ordered["step"] >= first_validation_step) & (ordered["step"] < first_test_step)
        ].copy(),
        test=ordered.loc[ordered["step"] >= first_test_step].copy(),
    )
