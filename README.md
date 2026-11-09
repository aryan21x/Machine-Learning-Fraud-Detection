# Machine Learning Fraud Detection

Fraud detection with Python, scikit-learn, and the PaySim dataset.

## Dataset

[Download PaySim from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1?resource=download).
It contains simulated financial transactions with known fraud labels.

Extract the CSV into `data/` and name it `FinancialData.csv`.

## Setup

Use Python 3.12+. Run these commands from the project folder:

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
python -m fraud_detection train --input data/FinancialData.csv
```

The model and results are saved automatically to `artifacts/run1/`, `run2/`, and so on.

Predict:

```powershell
python -m fraud_detection predict --model artifacts/run1/logistic_regression.joblib --input data/FinancialData.csv
```

Predictions are saved to `reports/predictions.csv`, then `predictions2.csv`, and so on.
The terminal prints flagged and unflagged totals, plus correct flags, false alarms,
and missed fraud when the input includes fraud labels.

## How it works

Logistic Regression learns from transaction amount and type.
Labels, account IDs, and balances are excluded from model inputs.
Preprocessing is fitted only on training data. Validation chooses the alert
threshold; test data measures final performance.

## Chronological split

The earliest 70% of time steps are used for training, the next 15% for validation,
and the rest for test. Transactions from the same time step stay together.
These percentages refer to time steps rather than transaction counts.
