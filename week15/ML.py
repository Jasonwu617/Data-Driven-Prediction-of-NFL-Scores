import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib.lines import Line2D

# === Load data ===
df = pd.read_csv("train_week15.csv")

# Keep a copy of the original rows so we can print incorrect ones later
df_original = df.copy()

# Drop non-feature columns if present
for col in ['team1', 'team2']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Separate features and target
y = df['ML']
X = df.drop(columns=['spread','ML','total'])

# Convert labels to 0/1/2... if needed
y = np.array(y) - 1

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# === Define preprocessing + model pipeline ===
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
    ("logreg", LogisticRegression(max_iter=500))
])

# === Fit model ===
model.fit(X, y)

# === Predict ===
y_pred = model.predict(X)

# === Confusion Matrix ===
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(8,8))     # bigger figure
disp.plot(cmap="Greys", ax=ax, colorbar=False)

# Increase tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=18)

# Increase text inside the boxes (TP, FP, etc.)
for text in ax.texts:
    text.set_fontsize(40)

# plt.title("Confusion Matrix — Logistic Regression", fontsize=20)
plt.xlabel("Predicted ML", fontsize=18)
plt.ylabel("True ML", fontsize=18)

plt.tight_layout()
plt.savefig('ML.png')
plt.show()

# === Print Incorrectly Classified Rows ===
incorrect_mask = (y != y_pred)
incorrect_rows = df_original[incorrect_mask]

print("\n=== Incorrectly Classified Rows ===")
print(incorrect_rows)
# === Histogram of team1 + team2 from incorrect rows ===

# Extract team columns
teams = pd.concat([
    incorrect_rows['team1'],
    incorrect_rows['team2']
], axis=0)



plt.figure(figsize=(10,6))

counts, bins, patches = plt.hist(teams, bins=30)

# Assign a different color to each bar
for p in patches:
    p.set_facecolor(np.random.rand(3,))

plt.xlabel("Team", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Histogram of Team Appearances in Incorrect Predictions", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('errors.png')
plt.show()