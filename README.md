# Machine Learning Fraud Detection

Fraud detection with Python and scikit-learn, comparing Logistic Regression and
Random Forest on IEEE-CIS data.

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

## Run

```powershell
python -m fraud_detection
```

Training defaults to `data/ieee-fraud-detection/train_transaction.csv` and
automatically joins `DeviceType` from `train_identity.csv` in the same folder.
Use `--input FILE` only to train on a different file.
Do not pass `train_identity.csv` as the main input.

Both models, a compact comparison report, and each model's held-out test predictions
are saved directly in `artifacts/`. Each execution replaces those files with the
latest results. The terminal prints a comparison table.

The test-prediction CSVs include fraud scores, the chosen threshold, predicted
labels, and false-positive/false-negative outcomes. Input can be CSV or Parquet;
a matching `*_identity` file in the same format is joined automatically.

## Model inputs

| Columns | Preparation |
| --- | --- |
| `TransactionAmt` | Log-transformed; also scaled for Logistic Regression. |
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

The models currently leave out `C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339`,
`id_*`, `DeviceInfo`, `dist1`, and `dist2`. These can be assessed in later
experiments; their usefulness has not been ruled out.

Both models use these same inputs. Random Forest combines 100 decision trees,
with depth limited to 20 and at least 5 training records per leaf. It uses class
weights to give fraud more influence and two CPU workers to limit resource use.
These settings are a starting point for tuning on validation data.

## Evaluation

The earliest 70% of distinct `TransactionDT` values are used for training, the
next 15% for validation, and the rest for test. Equal time values stay together;
these fractions do not guarantee exact percentages of rows.

Validation chooses each model's threshold with the highest F1. That threshold
stays fixed for all evaluation. Each model in `report.json` has:

- `validation`: results used to choose the threshold.
- `test_data_only`: results on the held-out test partition.
- `full_run`: results across the entire input file, including training rows.

Use `test_data_only` to assess performance on unseen records. `full_run` uses
the same trained model and threshold; it includes rows the model learned from.
Both sections include precision, recall, F1, average precision, ROC-AUC, and
true-positive, false-positive, and false-negative counts.
Kaggle's separate test files have no fraud labels, so local evaluation uses the
held-out part of the labeled training file.

A full-data comparison with seed 42 measured these results on the same 88,081
held-out transactions:

| Model | Precision | Recall | F1 | Average precision | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.1752 | 0.2610 | 0.2096 | 0.1504 | 0.7622 |
| Random Forest | 0.2394 | 0.2777 | 0.2571 | 0.1617 | 0.7724 |

Logistic Regression had higher validation F1: 0.2812 versus 0.2602.
Random Forest performed better on this test period, while
both models still missed most fraud.

Fraud is uncommon, so accuracy alone can hide missed fraud. Use precision,
recall, F1, average precision, and ROC-AUC together. Scores are not calibrated
estimates of real-world fraud probability, and many source fields are anonymized.
