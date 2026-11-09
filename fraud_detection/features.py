"""Stateless features available when a transaction is submitted."""

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit allowlist: future balances, IDs, flags, and labels cannot enter."""
    amount = frame["amount"].to_numpy(dtype=float)
    # log1p(amount) means log(1 + amount). It compresses large amounts before
    # Logistic Regression applies its learned weight; +1 also handles zero amounts.
    return pd.DataFrame(
        {"log_amount": np.log1p(amount), "type": frame["type"].astype(str)},
        index=frame.index,
    )
