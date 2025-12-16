import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge

from matplotlib.lines import Line2D
# === Load data ===
df = pd.read_csv("train_week15.csv")
# df = pd.read_csv("train_week11_OHE.csv")

# Drop non-feature columns if present
for col in ['team1', 'team2']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Separate features and target
y = df['spread']
X = df.drop(columns=['spread','ML','total'])

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

# === Define preprocessing + model pipeline ===
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
    # ("scaler", MinMaxScaler(feature_range=(0, 1))),
    # ("lasso", Lasso(alpha=0.1, random_state=42))
    ("BayesianRidge", BayesianRidge())
])

# === Fit model ===
model.fit(X, y)

# === Predict entire dataset ===
y_pred = model.predict(X)

# === Evaluate performance ===
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# === Plot results ===
plt.figure(figsize=(7,7))
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], lw=2)
plt.xlabel("Actual Spread", fontsize=18)
plt.ylabel("Predicted Spread", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create proxy artists (invisible lines) for the legend
legend_elements = [
    Line2D([0], [0], color='black', lw=0, label=f'$R^2$ = {round(r2,2)}'),
    Line2D([0], [0], color='black', lw=0, label=f'RMSE = {round(rmse,2)}')
]

# Add legend
plt.legend(handles=legend_elements, loc='upper left',fontsize=14)
# plt.title("Bayesian Ridge: Actual vs Predicted Spread")
plt.savefig('spread.png')
plt.show()


test_set = pd.read_csv("week15_input.csv")
# Keep only numeric columns
X_test = test_set.select_dtypes(include=[np.number])
scaler = RobustScaler()
scaler.fit(X)
test_scaled = scaler.transform(X_test)
y_test_pred = model.predict(test_scaled)
# Assume df is your original DataFrame
df_teams = test_set[['team1', 'team2']].copy()
df_teams['Predicted Spread'] = y_test_pred
df_teams.to_csv("week15_predictions.csv", index=False)