import pandas as pd
import os

path = "/Users/jasonwu/Desktop/Courses/CBE512_FinalProject/week11"
folder_name = os.path.basename(path)
print(folder_name)

# Load the CSV file
df = pd.read_csv("train.csv")

# Get all columns related to team1
team1_cols = [col for col in df.columns if col.startswith("team1")]

# Group by team1 and average all stats columns
team1_summary = df.groupby("team1")[team1_cols].mean(numeric_only=True).reset_index()

# Save to a new CSV file
team1_summary.to_csv(folder_name + "_averages.csv", index=False)
