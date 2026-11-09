# Machine Learning Fraud Detection

Fraud detection with Python, scikit-learn, and Logistic Regression on IEEE-CIS data.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview)
contains anonymized e-commerce transactions supplied by Vesta.
See the [data description and downloads](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
Kaggle requires sign-in and acceptance of the competition rules.

Keep the original filenames under `data/ieee-fraud-detection/`:

```text
train_transaction.csv
train_identity.csv
```

The loader reads selected columns and automatically joins `DeviceType` from the
matching identity file using `TransactionID`. Transactions without identity
information stay in the data. The identity file is optional.

## Setup

Use Python and run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Commands

```text
train     Train Logistic Regression and measure results
predict   Make predictions using a saved model
```

Train:

```powershell
python -m fraud_detection train
```

Training defaults to `data/ieee-fraud-detection/train_transaction.csv` and
automatically joins `DeviceType` from `train_identity.csv` in the same folder.
Use `--input FILE` only to train on a different file.
Do not pass `train_identity.csv` as the main input.

The model, compact report, and held-out test predictions are saved automatically
to `artifacts/run1/`, `run2/`, and so on. Use the run folder printed by `train`.
Old PaySim models must be retrained for the new inputs.

Predict with a saved model:

```powershell
python -m fraud_detection predict
```

Prediction automatically uses the highest numbered completed run in `artifacts/`.
Use `--model artifacts/run1/logistic_regression.joblib` to choose a specific model.

Prediction defaults to the same transaction file and identity join. Use
`--input FILE` to score a different batch. Scoring the whole source file includes
training rows; use the held-out test metrics in `report.json` to judge performance.

Predictions are saved to `reports/predictions.csv`, then `predictions2.csv`, and so on.
The terminal prints flagged and unflagged totals, plus correct flags, false alarms,
and missed fraud when labels are available. A new batch can be a CSV or Parquet
transaction file; a matching `*_identity` file in the same format is loaded automatically.

## Model inputs

| Columns | Preparation |
| --- | --- |
| `TransactionAmt` | Log-transformed, missing amounts filled with the training median, then scaled. |
| `ProductCD` | Product category. |
| `card1`–`card6` | Categorical card attributes, including encoded fields. |
| `addr1`, `addr2` | Categorical address codes. |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains. |
| `DeviceType` | Device category from the identity file. |

Categorical fields use one-hot encoding, with missing values represented as
`missing`. Categories and numeric preprocessing are learned from training data only.

`TransactionID` is used for joining and identifying predictions. `TransactionDT`
is used for splitting. `isFraud` is the answer for training and evaluation.
None of these three columns is a model input.

The model currently leaves out `C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339`,
`id_*`, `DeviceInfo`, `dist1`, and `dist2`. These can be assessed in later
experiments; their usefulness has not been ruled out.

## Evaluation

The earliest 70% of distinct `TransactionDT` values are used for training, the
next 15% for validation, and the rest for test. Equal time values stay together;
these fractions do not guarantee exact percentages of rows.

Validation chooses the threshold with the highest F1. That threshold stays fixed
for test evaluation and prediction. `report.json` contains the chosen threshold
and validation/test metrics, including false positives and false negatives.
Kaggle's separate test files have no fraud labels, so local evaluation uses the
held-out part of the labeled training file.

A full-data run with seed 42 measured precision **0.1752**, recall **0.2610**,
F1 **0.2096**, average precision **0.1504**, and ROC-AUC **0.7622** on 88,081
held-out transactions. The validation-selected threshold was **0.845922**.
Low recall remains a limitation of this feature set.

Fraud is uncommon, so accuracy alone can hide missed fraud. Use precision,
recall, F1, average precision, and ROC-AUC together. Scores are not calibrated
estimates of real-world fraud probability, and many source fields are anonymized.
