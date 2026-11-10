"""IEEE-CIS transaction predictors, with DeviceType from the identity file."""

import numpy as np
import pandas as pd


# These are labels, not quantities. A card/address code of 150 does not mean
# "larger" or "riskier" than a code of 100. M fields are categorical match flags.
M_COLUMNS = tuple(f"M{number}" for number in range(1, 10))
CATEGORICAL_COLUMNS = (
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", *M_COLUMNS, "DeviceType",
)
CODE_COLUMNS = ("card1", "card2", "card3", "card5", "addr1", "addr2")
C_COLUMNS = tuple(f"C{number}" for number in range(1, 15))
D_COLUMNS = tuple(f"D{number}" for number in range(1, 16))
V_COLUMNS = tuple(f"V{number}" for number in range(1, 340))
NULLABLE_NUMERIC_COLUMNS = (*D_COLUMNS, "dist1", "dist2", *V_COLUMNS)
RAW_NUMERIC_COLUMNS = (
    "TransactionAmt", "TransactionDT", *C_COLUMNS, *NULLABLE_NUMERIC_COLUMNS,
)
NUMERIC_COLUMNS = ("log_amount", *RAW_NUMERIC_COLUMNS[1:])
FEATURE_COLUMNS = (*RAW_NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare every transaction predictor except the target and transaction ID."""
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
    # Build numeric columns together to avoid fragmenting a wide DataFrame.
    numeric = frame[list(RAW_NUMERIC_COLUMNS)].astype("float64")
    # log1p(amount) means log(1 + amount). It compresses large amounts before
    # Logistic Regression applies its learned weight; +1 also handles zero amounts.
    numeric["TransactionAmt"] = np.log1p(numeric["TransactionAmt"])
    numeric.rename(columns={"TransactionAmt": "log_amount"}, inplace=True)
    # Retain D, distance, and V blanks for model-specific preprocessing.
    # TransactionDT is a numeric predictor as well as the chronological split key.
    return pd.concat([numeric, features], axis=1)
