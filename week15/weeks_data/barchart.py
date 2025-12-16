import pandas as pd
import matplotlib.pyplot as plt

# Example: load your data
df = pd.read_excel("Different_weekaverages_predictions.xlsx")

game_cols = [f"{i} games" for i in range(1, 14)]

for idx, row in df.iterrows():
    print(idx)
    values = row[game_cols].values
    line_value = row["Line"]

    # Color bars based on comparison with the line
    colors = ["red" if v > line_value else "green" for v in values]

    plt.figure(figsize=(8, 4))
    plt.bar(game_cols, values, color=colors)
    plt.axhline(line_value, linestyle="--", linewidth=2)

    plt.title(f"Row {idx}")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
