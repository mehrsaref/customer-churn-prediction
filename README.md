# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, using the
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset
(7,043 customers, 21 features).

## The business problem

Retaining an existing customer costs far less than acquiring a new one. If we can identify
customers who are about to churn, the retention team can target them with an offer before
they leave.

This shapes how the model is evaluated. **A missed churner (false negative) is expensive** —
we lose the customer entirely. **A false alarm (false positive) is cheap** — we waste a
discount on someone who would have stayed anyway. So this project optimises for *recall on
the churn class*, not accuracy.

## Dataset

- 7,043 rows, 21 columns, one row per customer
- Target: `Churn` (Yes/No)
- **Class imbalance: 26.5% churn / 73.5% no churn.** A model that predicts "No" for everyone
  scores 73.5% accuracy while being completely useless. Accuracy is not a valid metric here.

## Findings from exploratory analysis

| Factor | Finding |
|---|---|
| **Contract type** | Strongest signal. Month-to-month customers churn at **42.7%**, one-year at 11.3%, two-year at just 2.8% |
| **Tenure** | Churners average **18 months** vs 37.6 months for those who stay — churn happens early |
| **Monthly charges** | Churners pay more on average (**$74.44** vs $61.27) |
| **Internet service** | Fiber optic customers churn noticeably more than DSL customers |
| **Gender** | No meaningful effect |

### Data quality note

`TotalCharges` is stored as a string, not a number. 11 rows contain a blank space instead of
a value — all of them customers with `tenure = 0`, i.e. brand new customers who have not been
billed yet. `df.isna().sum()` reports **zero** missing values because pandas treats `" "` as a
valid string, so this has to be caught by checking dtypes rather than by a null check.

## Results

_To be filled in as models are built._

| Model | Precision (churn) | Recall (churn) | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (predict majority) | – | 0.00 | – | 0.50 |
| Logistic Regression | | | | |
| Random Forest | | | | |
| Gradient Boosting | | | | |

## Project structure

```
data/         raw dataset (CSV)
notebooks/    exploratory analysis and modelling
src/          reusable code (data loading, preprocessing, training)
models/       serialised trained models
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
