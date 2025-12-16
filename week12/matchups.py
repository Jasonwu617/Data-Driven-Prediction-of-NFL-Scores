import requests
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd
import re
from io import StringIO

TEAM_ABBR = [
    'ari', 'atl', 'bal', 'buf', 'car', 'chi', 'cin', 'cle',
    'dal', 'den', 'det', 'gb',  'hou', 'ind', 'jax', 'kc',
    'lv',  'lac', 'lar', 'mia', 'min', 'ne',  'no',  'nyg',
    'nyj', 'phi', 'pit', 'sf',  'sea', 'tb',  'ten', 'wsh'
]

abbreviations = {
    'ari': 'Cardinals',
    'atl': 'Falcons',
    'bal': 'Ravens',
    'buf': 'Bills',
    'car': 'Panthers',
    'chi': 'Bears',
    'cin': 'Bengals',
    'cle': 'Browns',
    'dal': 'Cowboys',
    'den': 'Broncos',
    'det': 'Lions',
    'gb':  'Packers',
    'hou': 'Texans',
    'ind': 'Colts',
    'jax': 'Jaguars',
    'kc':  'Chiefs',
    'lv':  'Raiders',
    'lac': 'Chargers',
    'lar': 'Rams',
    'mia': 'Dolphins',
    'min': 'Vikings',
    'ne':  'Patriots',
    'no':  'Saints',
    'nyg': 'Giants',
    'nyj': 'Jets',
    'phi': 'Eagles',
    'pit': 'Steelers',
    'sf':  '49ers',
    'sea': 'Seahawks',
    'tb':  'Buccaneers',
    'ten': 'Titans',
    'wsh': 'Commanders'
}

# === Load data ===
df_teams = pd.read_csv("week12_averages.csv")
df_matchups = pd.read_excel("week12_matchups.xlsx")

all_games = []
for index, row in df_matchups.iterrows():
    team1 = row['team1']
    team2 = row['team2']
    team1_label = TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team1][0])
    team2_label = TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team2][0])
    # Get row(s) where team1 is 'sf'
    team1_row = df_teams[df_teams['team1'] == team1].reset_index(drop=True)
    team2_row = df_teams[df_teams['team1'] == team2].reset_index(drop=True)
    team2_row.columns = team2_row.columns.str.replace('team1', 'team2', regex=False)
    df_combined = pd.concat([team1_row, team2_row], axis=1)
    df_combined.insert(1, 'team2', df_combined.pop('team2'))
    all_games.append(df_combined)

# Combine into one DataFrame
combined_df = pd.concat(all_games, ignore_index=True)
print(combined_df.head())

# Save to CSV
combined_df.to_csv("week12_input.csv", index=False)