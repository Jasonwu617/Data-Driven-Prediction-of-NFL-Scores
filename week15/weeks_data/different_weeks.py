import pandas as pd

def extract_last_n_games_per_team(
    input_csv: str,
    n_games: int,
    output_csv: str
):
    """
    Create a new CSV containing the last n_games played by each team1.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    n_games : int
        Number of most recent games per team1 to keep.
    output_csv : str
        Path where the filtered CSV will be saved.
    """

    df = pd.read_csv(input_csv)

    # Take the last n_games for each team1
    df_filtered = (
        df
        .groupby("team1", group_keys=False)
        .tail(n_games)
        .reset_index(drop=True)
    )

    df_filtered.to_csv(output_csv, index=False)

    return df_filtered

# Last 2 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=1,
    output_csv="last_1_games_per_team.csv"
)

# Last 2 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=2,
    output_csv="last_2_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=3,
    output_csv="last_3_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=4,
    output_csv="last_4_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=5,
    output_csv="last_5_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=6,
    output_csv="last_6_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=7,
    output_csv="last_7_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=8,
    output_csv="last_8_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=9,
    output_csv="last_9_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=10,
    output_csv="last_10_games_per_team.csv"
)

# Last 3 games per team
extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=11,
    output_csv="last_11_games_per_team.csv"
)

extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=12,
    output_csv="last_12_games_per_team.csv"
)

extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=13,
    output_csv="last_13_games_per_team.csv"
)

extract_last_n_games_per_team(
    input_csv="train_week15.csv",
    n_games=14,
    output_csv="last_14_games_per_team.csv"
)
