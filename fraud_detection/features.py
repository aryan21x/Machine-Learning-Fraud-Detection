"""Selected IEEE-CIS inputs available when a transaction is scored."""

import numpy as np
import pandas as pd


# These are labels, not quantities. A card/address code of 150 does not mean
# "larger" or "riskier" than a code of 100.
CATEGORICAL_COLUMNS = (
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
)
CODE_COLUMNS = ("card1", "card2", "card3", "card5", "addr1", "addr2")
C_COLUMNS = tuple(f"C{number}" for number in range(1, 15))
NUMERIC_COLUMNS = ("log_amount", *C_COLUMNS)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare amount, numeric C inputs, and selected transaction categories."""
    features = frame[list(CATEGORICAL_COLUMNS)].copy()
    # The same code may arrive as 150, 150.0, or "150.0" in CSV/Parquet.
    for column in CODE_COLUMNS:
        features[column] = pd.to_numeric(features[column]).astype("Int64")
    # Make category labels consistent and give blank entries a "missing" label.
    features = features.astype("string").fillna("missing")
    # Logistic Regression and Random Forest one-hot encode these labels:
    # e.g. card4="visa" gives card4_visa=1, card4_mastercard=0.
    # XGBoost uses codes marked as categories, without assuming a numeric order.
    # Both encoders learn categories on training data only and handle unseen labels.
    amount = frame["TransactionAmt"].to_numpy(dtype=float)
    # log1p(amount) means log(1 + amount). It compresses large amounts before
    # Logistic Regression applies its learned weight; +1 also handles zero amounts.
    features.insert(0, "log_amount", np.log1p(amount))
    # Keep supplied C values numeric; they are not category codes.
    # models.py scales them using training statistics for Logistic Regression.
    features[list(C_COLUMNS)] = frame[list(C_COLUMNS)].astype("float64")
    return features
