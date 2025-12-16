import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer

# Models to try
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler



# === Load data ===
df = pd.read_csv("train_week11.csv")
# df = pd.read_csv("train_OHE.csv")

# === Drop columns that aren't features ===
for col in ['team1', 'team2']:
    if col in df.columns:
        df = df.drop(columns=[col])

# === Separate features (X) and target (y) ===
y = df['spread'].values
X = df.drop(columns=['spread'])

# Keep only numeric columns for training
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols].copy()

print(f"Using {len(numeric_cols)} numeric feature columns for training.")
print(f"Number of rows: {df.shape[0]}")

# === 10-fold cross validation setup ===
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# === Models to test ===
# Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "SVR": SVR(kernel='rbf', C=10, gamma=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=1000, learning_rate_init=0.001),
    "BayesianRidge": BayesianRidge()

    # "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
}

# === Evaluate models ===
results = []
for name, model in models.items():
    # Create pipeline: Imputer -> Scaler -> Model
    pipe = Pipeline([
        # ("imputer", SimpleImputer(strategy="mean")),
        # ("scaler", StandardScaler()),
        # ("scaler", MinMaxScaler(feature_range=(0, 1))),
        ("scaler", RobustScaler()),
        ("model", model)
    ])

    # RMSE (via negative MSE scores)
    neg_mse_scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)

    # R^2 scores
    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    results.append({
        "model": name,
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std()
    })

# === Display results ===
res_df = pd.DataFrame(results).sort_values("rmse_mean")
pd.set_option("display.float_format", "{:.4f}".format)
print("\nCross-validation results (10-fold):")
print(res_df.reset_index(drop=True))

# === Save results ===
res_df.to_csv("cv_results.csv", index=False)
print("\nSaved CV summary to: cv_results.csv")
