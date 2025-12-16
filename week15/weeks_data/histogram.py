import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example load
df = pd.read_excel("Different_weekaverages_predictions.xlsx")

game_cols = [f"{i} games" for i in range(1, 14)]
x = np.arange(1, 14)

# Fraction (percentage) of games below the Line
fractions_below_line = [
    (df[col] < df["Line"]).mean()
    for col in game_cols
]

counts_below_line = [
    (df[col] < df["Line"]).sum()
    for col in game_cols
]

plt.figure(figsize=(8, 6))

plt.bar(
    x,
    counts_below_line,
    width=1.0,               # bars touch
    edgecolor="black",
    linewidth=1.0,
    color="#4C72B0"          # professional muted blue
)

# 50% reference line
plt.axhline(
    8,
    linestyle="--",
    linewidth=2,
    color="red"
)

plt.xlabel("Number of Games Used for Input Average", fontsize=14)
plt.ylabel("Number of Correctly Predicted Spread Picks", fontsize=14)

plt.xticks(x, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.5, 13.5)
plt.ylim(0, 16)

# Ensure full border around plot
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
ax.legend(['50%'], loc="upper right")
plt.tight_layout()
plt.savefig('histogram_differentweeks.pdf')
plt.show()
