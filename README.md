# Machine Learning Fraud Detection

A reproducible comparison of Logistic Regression, Random Forest, and XGBoost
for fraud detection on IEEE-CIS transaction data.

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

Each run replaces the three saved models, held-out prediction CSVs, and
`report.json` in `artifacts/`.

## Model inputs

| Columns | Preparation |
| --- | --- |
| `TransactionAmt` | Log-transformed; also scaled for Logistic Regression. |
| `C1`–`C14` | Provided numeric features; scaled for Logistic Regression. |
| `ProductCD` | Product category. |
| `card1`–`card6` | Categorical card attributes, including encoded fields. |
| `addr1`, `addr2` | Categorical address codes. |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains. |
| `DeviceType` | Device category from the identity file. |

Logistic Regression and Random Forest use one-hot encoding for categorical fields.
XGBoost uses native categorical splits with category codes learned from training
data. These codes identify categories without implying an order; unseen labels
are treated as unknown. Missing categorical values are represented as `missing`.
Category mappings and numeric preprocessing are learned from training data only.

`TransactionID` is used for joining and identifying predictions. `TransactionDT`
is used for splitting. `isFraud` is the answer for training and evaluation.
None of these three columns is a model input.

The models currently leave out `D1`–`D15`, `M1`–`M9`, `V1`–`V339`,
`id_*`, `DeviceInfo`, `dist1`, and `dist2`. These can be assessed in later
experiments; their usefulness has not been ruled out.

All three models use the same 27 source columns. The numeric `C` values are used
directly; their individual business meanings are anonymized.
Random Forest combines 200 decision trees,
with depth limited to 30 and at least 5 training records per leaf.
XGBoost adds 300 trees sequentially, with maximum depth 6 and learning rate 0.05.
Each new tree learns corrections to the predictions made by the earlier trees.
It uses CPU histogram training, two workers, and a fraud class weight calculated
from the training partition. These are initial settings for comparison.

## Evaluation

By default, the earliest 70% of distinct `TransactionDT` values are used for
training, the next 15% for validation, and the final 15% for testing.
Equal time values stay together; these percentages refer to distinct time
values, so the percentages of transaction rows can differ.

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

## Results

**The table shows results on held-out test data only: 88,081 transactions, with
seed 42.** Each model's threshold was selected using validation data and applied
unchanged to the test data. Thresholds below are rounded to six decimal places.
These are exploratory results from repeated experiments on the same split.

| Model | Selected threshold | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.846915 | 0.1909 | 0.2656 | 0.2221 | 0.7724 |
| Random Forest | 0.709416 | 0.4171 | 0.3834 | 0.3995 | 0.8278 |
| XGBoost | 0.818520 | 0.5759 | 0.4111 | 0.4797 | 0.8601 |


The four metrics below range from 0 to 1; higher is better:

- **Precision:** The fraction of flagged transactions that were actually
  fraudulent. Higher precision means fewer false alarms among the flags.
- **Recall:** The fraction of actual fraudulent transactions that the model
  caught. Higher recall means less missed fraud.
- **F1:** Combines precision and recall using their harmonic mean:
  `2 * precision * recall / (precision + recall)`. A high F1 requires both to be strong.
- **ROC-AUC:** Area under the curve comparing recall with the false-positive
  rate (the fraction of legitimate transactions incorrectly flagged) across
  thresholds. It summarizes how well fraud scores separate fraudulent and
  legitimate transactions: **0.5** means random ranking and **1.0** means
  perfect separation.

**Selected threshold:** The minimum fraud score needed to flag a transaction.
It is chosen to maximize F1 on validation data, then kept fixed for testing.
A higher threshold generally produces fewer fraud flags.
