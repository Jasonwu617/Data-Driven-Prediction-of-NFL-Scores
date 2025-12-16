import pandas as pd
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_excel("Different_weekaverages_predictions.xlsx")

game_cols = [f"{i} games" for i in range(1, 14)]

n_plots = len(df)
n_cols = 4
n_rows = math.ceil(n_plots / n_cols)

# Smaller fonts globally
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
})

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(14, 3 * n_rows),
    sharey=True,
    sharex=True
)

axes = axes.flatten()

for ax, (_, row) in zip(axes, df.iterrows()):
    values = row[game_cols].values
    line_value = row["Line"]

    colors = ["red" if v > line_value else "green" for v in values]

    ax.bar(game_cols, values, color=colors)
    ax.axhline(line_value, linestyle="--", linewidth=1)

    title = (
        f"{row['team1']} vs {row['team2']}\n"
        f"Winning Bet: {row['Winning Bet']}"
    )
    ax.set_title(title)

    ax.set_xticks(range(len(game_cols)))
    ax.set_xticklabels(game_cols, rotation=45)

# Turn off unused axes
for ax in axes[n_plots:]:
    ax.axis("off")

# One shared y-label
fig.supylabel("Predicted ")

plt.tight_layout(h_pad=0.8, w_pad=0.6)

# Export to PDF
plt.savefig("week_averages_barplots.pdf", bbox_inches="tight")
plt.close()
