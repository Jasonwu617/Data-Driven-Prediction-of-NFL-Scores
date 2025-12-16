import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap

# ====================================
# 1. LOAD & PREP DATA
# ====================================
df = pd.read_csv("train_week15.csv")

# drop non-feature columns if they exist
for col in ['team1', 'team2']:
    if col in df.columns:
        df = df.drop(columns=[col])

# target + features
y = df['spread']
X = df.drop(columns=['spread', 'ML', 'total'])

# numeric columns only
X = X.select_dtypes(include=[np.number])

# ====================================
# 2. MODEL PIPELINE
# ====================================
model = Pipeline([
    # ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
    ("BayesianRidge", BayesianRidge())
])

model.fit(X, y)

# ====================================
# 3. SHAP EXPLAINER + VALUES
# ====================================
# Preprocess X manually to match pipeline input
X_processed = model.named_steps["scaler"].transform(
    X
)

# SHAP auto-explainer (works with linear models)
explainer = shap.Explainer(
    model.named_steps["BayesianRidge"],
    X_processed
)

shap_values = explainer(X_processed)

# Ensure feature names are correct (SHAP sometimes drops them)
shap_values.feature_names = X.columns.tolist()

# ====================================
# 4. SELECT TOP 5 MOST IMPORTANT FEATURES
# ====================================
# Compute mean absolute SHAP impact per feature
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# Get indices of top 5
top5_idx = np.argsort(mean_abs_shap)[-10:][::-1]
# top5_idx = np.array([19,43, 7,31])
# print(top5_idx)
# Slice shap values to top 5 features
shap_values_top5 = shap.Explanation(
    values = shap_values.values[:, top5_idx],
    base_values = shap_values.base_values,
    data = shap_values.data[:, top5_idx],
    feature_names = [shap_values.feature_names[i] for i in top5_idx]
)

# ====================================
# 5. SHAP BEESWARM (TOP 5 ONLY)
# ====================================
plt.figure(figsize=(16, 10))
shap.plots.beeswarm(shap_values_top5, max_display=10)
# shap.plots.beeswarm(shap_values, max_display=10)