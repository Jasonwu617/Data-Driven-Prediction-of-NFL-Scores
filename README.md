environment.yml is the conda environment I used. 

data.py scrapes 2025 nfl data from espn

For each week:
1. averages.py calculates the average stats for each team
2. matchups.py prepares the X_test set
3. optimization.py trains the model and predicts the spread for the X_test set
