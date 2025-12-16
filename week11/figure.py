import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('cv_results.csv')

# Generate a different color for each bar
colors = plt.cm.tab20(np.linspace(0, 1, len(df)))

plt.figure(figsize=(10,6))
plt.barh(df["model"], df["rmse_mean"], color=colors)
# plt.ylabel('Model', fontsize=18, fontweight='bold')
plt.xlabel('RMSE', fontsize=18)
# plt.title('10-Fold Cross-Validation for Different Scikit-learn Models', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Make tick lines bigger
tick_length = 10
tick_width = 1.5
plt.tick_params(axis='both', which='major', length=tick_length, width=tick_width)

# Make plot borders (spines) same thickness as tick lines
for spine in plt.gca().spines.values():
    spine.set_linewidth(tick_width)
plt.tight_layout()
plt.savefig('CV.png')
plt.show()
