import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("extended_data.csv")

# Columns to exclude
exclude_cols = ["team1", "team2", "total", "spread", "ML"]

# Select numeric columns except excluded
cols = [c for c in df.columns if c not in exclude_cols]
X = df[cols].select_dtypes(include=['number']).dropna()

# Run KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(X)

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

ml_values = df.loc[X.index, "ML"]
total_values = df.loc[X.index, "total"]
spread_values = df.loc[X.index, "spread"]

# ─────────────────────────────
# Plot 1: Color by KMeans Cluster
# ─────────────────────────────
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, s=50)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Plot — Colored by KMeans Cluster")
plt.tight_layout()
plt.show()

# ─────────────────────────────
# Plot 2: Color by ML Column
# ─────────────────────────────
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=ml_values, s=50)
plt.xlabel("PC1", fontsize=18)
plt.ylabel("PC2", fontsize=18)
# plt.title("PCA Plot — Colored by ML Column")
# plt.colorbar(label="ML Value")
tick_length = 10
tick_width = 1.5
plt.tick_params(axis='both', which='major', length=tick_length, width=tick_width)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Make plot borders (spines) same thickness as tick lines
for spine in plt.gca().spines.values():
    spine.set_linewidth(tick_width)
plt.tight_layout()
plt.savefig('ML.png')
plt.show()

# ─────────────────────────────
# Plot 2: Color by ML Column
# ─────────────────────────────
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=total_values, s=50)
plt.xlabel("PC1", fontsize=18)
plt.ylabel("PC2", fontsize=18)
# plt.title("PCA Plot — Colored by total points Column")
plt.colorbar(label="Total Points")
tick_length = 10
tick_width = 1.5
plt.tick_params(axis='both', which='major', length=tick_length, width=tick_width)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Make plot borders (spines) same thickness as tick lines
for spine in plt.gca().spines.values():
    spine.set_linewidth(tick_width)
plt.tight_layout()
plt.savefig('total.png')
plt.show()

# ─────────────────────────────
# Plot 2: Color by ML Column
# ─────────────────────────────
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=spread_values, s=50)
plt.xlabel("PC1", fontsize=18)
plt.ylabel("PC2", fontsize=18)
# plt.title("PCA Plot — Colored by spread Column")
plt.colorbar(label="Spread")
tick_length = 10
tick_width = 1.5
plt.tick_params(axis='both', which='major', length=tick_length, width=tick_width)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Make plot borders (spines) same thickness as tick lines
for spine in plt.gca().spines.values():
    spine.set_linewidth(tick_width)
plt.tight_layout()
plt.savefig('spread.png')

plt.show()

