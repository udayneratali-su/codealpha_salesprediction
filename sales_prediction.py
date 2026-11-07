"""Task 4: Sales Prediction using Python
Run:  python sales_prediction.py                (demo data)
      python sales_prediction.py yourfile.csv   (your dataset; target column must be 'Sales')
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "Sales"

# ---------- Load data (your CSV, or a small demo dataset) ----------
if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1])
else:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({"TV": rng.uniform(1, 300, n), "Radio": rng.uniform(0, 50, n),
                       "Newspaper": rng.uniform(0, 100, n),
                       "Platform": rng.choice(["Social Media", "Search", "Display", "Email"], n),
                       "Segment": rng.choice(["Youth", "Adults", "Seniors"], n)})
    df[TARGET] = (3 + 4.5 * np.sqrt(df.TV) + 0.2 * df.Radio + 0.002 * df.Newspaper
                  + df.Platform.map({"Social Media": 3, "Search": 2, "Display": 0, "Email": 1})
                  + df.Segment.map({"Youth": 2, "Adults": 1, "Seniors": 0}) + rng.normal(0, 1.5, n))

# ---------- 1. DATA PREPARATION: cleaning, transformation, feature selection ----------
print("=" * 60, "\nSTEP 1: DATA PREPARATION\n" + "=" * 60)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].drop_duplicates().dropna(subset=[TARGET])
spend = df.select_dtypes("number").columns.drop(TARGET).tolist()       # ad spend columns
cats = df.select_dtypes(exclude="number").columns.tolist()             # platform / segment
for c in spend: df[c] = df[c].fillna(df[c].median())                   # cleaning: fill missing
for c in cats: df[c] = df[c].fillna(df[c].mode()[0])
print(f"Cleaned data: {len(df)} rows | spend columns: {spend} | categories: {cats}")

X = pd.get_dummies(df.drop(columns=TARGET), dtype=float)               # transformation: encode text
y = df[TARGET]
corr = X.corr().abs()                                                  # feature selection: drop
redundant = [c for i, c in enumerate(X.columns)                        # constant or duplicate-like
             if X[c].nunique() <= 1 or (corr.iloc[:i][c] > 0.9).any()]  # (corr > 0.9) columns
keep = X.columns.drop(redundant)
print("Selected features:", list(keep))
print("Dropped (redundant):", redundant or "none")
print("Correlation of each ad spend with sales:\n" + df[spend + [TARGET]].corr()[TARGET].drop(TARGET).round(2).to_string())

# ---------- 2. REGRESSION MODEL: train and forecast sales ----------
print("\n" + "=" * 60, "\nSTEP 2: SALES PREDICTION MODEL\n" + "=" * 60)
X_tr, X_te, y_tr, y_te = train_test_split(X[keep], y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)}
best_score, model = -1, None
for name, m in models.items():
    m.fit(X_tr, y_tr)
    r2 = r2_score(y_te, m.predict(X_te))
    print(f"{name:18s} R2 = {r2:.3f} | MAE = {mean_absolute_error(y_te, m.predict(X_te)):.2f}")
    if r2 > best_score: best_score, model = r2, m
print(f"Best model: {type(model).__name__}")


def predict(frame):
    """Predict sales for new campaign(s) given raw columns (spend + categories)."""
    enc = pd.get_dummies(frame, dtype=float).reindex(columns=X.columns, fill_value=0)
    return model.predict(enc[keep])


base = {**{c: df[c].median() for c in spend}, **{c: df[c].mode()[0] for c in cats}}
shown = {k: (round(float(v), 1) if k in spend else v) for k, v in base.items()}
print(f"\nForecast for a typical upcoming campaign {shown}:\n  -> predicted {TARGET} = "
      f"{predict(pd.DataFrame([base]))[0]:.2f}")

# ---------- 3. IMPACT ANALYSIS: how do changes in ad spend change sales? ----------
print("\n" + "=" * 60, "\nSTEP 3: IMPACT OF CHANGING ADVERTISING SPEND\n" + "=" * 60)
base_pred = predict(pd.DataFrame([base]))[0]
impact = {}
for c in spend:
    up = {**base, c: base[c] * 1.10}
    impact[c] = predict(pd.DataFrame([up]))[0] - base_pred
    print(f"+10% {c:10s} spend -> {impact[c]:+.2f} sales ({impact[c] / base_pred:+.1%})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(y_te, model.predict(X_te[keep]), alpha=0.6)
ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax[0].set(title="Actual vs Predicted Sales", xlabel="Actual", ylabel="Predicted")
for c in spend:                                                         # spend vs predicted sales
    grid = np.linspace(df[c].min(), df[c].max(), 30)
    ax[1].plot(grid / grid.max() * 100, predict(pd.DataFrame([{**base, c: g} for g in grid])), label=c)
ax[1].set(title="Impact of spend on sales (spend as % of max)", xlabel="% of max spend", ylabel="Predicted sales")
ax[1].legend()
plt.tight_layout()
plt.savefig("results.png", dpi=150)

# ---------- 4. ACTIONABLE INSIGHTS ----------
print("\n" + "=" * 60, "\nSTEP 4: INSIGHTS FOR MARKETING STRATEGY\n" + "=" * 60)
best_ch, worst_ch = max(impact, key=impact.get), min(impact, key=impact.get)
print(f"1. Invest more in {best_ch}: it gives the biggest sales lift per extra spend.")
print(f"2. Cut or review {worst_ch}: it adds the least sales per extra spend.")
for c in cats:                                                          # compare categories (same spend)
    pred = pd.Series({v: predict(pd.DataFrame([{**base, c: v}]))[0] for v in df[c].unique()}).sort_values()
    print(f"3. Best {c}: '{pred.index[-1]}' (predicted sales {pred.iloc[-1]:.1f}); "
          f"weakest: '{pred.index[0]}' ({pred.iloc[0]:.1f}). Focus budget on the best one.")
print("4. Watch the right-hand graph: when a channel's line flattens, extra spend stops paying off.")
print("\nGraphs saved to results.png")
plt.show()
