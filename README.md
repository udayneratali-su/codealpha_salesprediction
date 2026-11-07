# Task 4: Sales Prediction using Python

Predicts future sales from **advertising spend, target segment and platform**, shows how changing ad spend
affects sales, and gives marketing recommendations.

## How it matches the task
| Task requirement | Where in `sales_prediction.py` |
|---|---|
| Predict future sales from ad spend, segment, platform | STEP 2 + the `predict()` function |
| Data cleaning, transformation, feature selection | STEP 1 (fill missing, drop duplicates, one-hot encode, drop redundant features) |
| Regression to forecast sales | STEP 2 (Linear Regression vs Random Forest, best chosen by R²) |
| Analyze how changes in advertising impact sales | STEP 3 (+10% spend per channel) and the right-hand graph |
| Actionable insights for marketing strategy | STEP 4 (where to invest, what to cut, best platform/segment) |

## Setup and run
```bash
pip install -r requirements.txt
python sales_prediction.py                    # demo data
python sales_prediction.py yourfile.csv       # your dataset
```
Your CSV needs a target column named `Sales`. Other number columns = ad spend; text columns = platform/segment.
(If your column has another name, change `TARGET = "Sales"` at the top of the script.)

## Output
- Terminal: the 4 steps with metrics, forecast, impact of spend changes and insights.
- `results.png`: Actual vs Predicted sales, and the spend-vs-sales impact curves (also pops up on screen).

## Notes
If your data has dates, use a time-based train/test split instead of a random one.
The demo data is synthetic; run on your real dataset for real conclusions.
