import pandas as pd
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_excel("Different_weekaverages_predictions.xlsx")

game_cols = [f"{i} games" for i in range(1, 14)]

n_plots = len(df)
n_cols = 4
n_rows = math.ceil(n_plots / n_cols)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
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
labels = [1,2,3,4,5,6,7,8,9,10,11,12,13]
for ax, (_, row) in zip(axes, df.iterrows()):
    values = row[game_cols].values
    line_value = row["Line"]

    colors = ["red" if v > line_value else "green" for v in values]

    ax.bar(game_cols, values, color=colors)
    ax.axhline(line_value, linestyle="--", linewidth=1)

    # Title inside plot (top-left)
    title_text = (
        f"{row['team1']} vs {row['team2']}\n"
        f"Winning Bet: {row['Winning Bet']}"
    )

    ax.text(
        0.5, 0.95,
        title_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontweight='bold'
    )

    ax.set_xticks(range(len(game_cols)))
    ax.set_xticklabels(labels)

# Hide unused axes
for ax in axes[n_plots:]:
    ax.axis("off")

fig.supylabel("Predicted Spread")
fig.supxlabel("Number of Games Used for Input Average")

plt.tight_layout(h_pad=0.6, w_pad=0.5)

plt.savefig("week_averages_barplots.pdf", bbox_inches="tight")
plt.close()
