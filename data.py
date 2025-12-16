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

SEASON_YEAR = 2025
SCHEDULE_URL = f"https://www.espn.com/nfl/team/_/name/{TEAM_ABBR}"

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
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Go from game page to team stats page
def convert_to_matchup(url):
    # Split the URL by '/'
    parts = url.split('/')
    # Find the index of 'game' and replace it with 'matchup'
    try:
        game_index = parts.index('game')
        parts[game_index] = 'matchup'
    except ValueError:
        return 0
        # raise ValueError("URL does not contain '/game/' segment")
    
    # Keep only the base URL and gameId (discard trailing teams)
    # The gameId is always the part after 'gameId'
    if 'gameId' in parts:
        gameid_index = parts.index('gameId')
        new_parts = parts[:gameid_index+2]  # include 'gameId' and its value
    else:
        return 0
        # raise ValueError("URL does not contain 'gameId'")
    
    # Reconstruct the URL
    new_url = '/'.join(new_parts)
    return new_url

# Gets links for games
def get_game_links(abbreviation):
    SCHEDULE_URL = f"https://www.espn.com/nfl/team/_/name/{abbreviation}"
    resp = requests.get(SCHEDULE_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):   
        if "gameId" in a['href']:
            full_url = a['href']
            full_url = convert_to_matchup(full_url)
            if full_url not in links and full_url != 0:
                links.append(full_url)
    return links

# Extract team stats and transforms into 1 D pandas dataframe
def parse_boxscore(url, abbreviation):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # -----------------------------
    # 🏈 Extract Team Names and Score
    # -----------------------------
    # ESPN pages usually include <span class="ScoreCell__TeamName"> and <span class="ScoreCell__Score">
    # 🕵️ Find the <title> tag whose text ends with "Game Stats - ESPN"
    title_tag = soup.find("title", string=lambda s: s and s.strip().endswith("Game Stats - ESPN"))
    if not title_tag:
        raise RuntimeError("Could not find <title> ending with 'Game Stats - ESPN'")
    title_text = title_tag.get_text(strip=True)
    # 🎯 Extract team names and scores using regex
    pattern = r"^(.*?)\s(\d+)-(\d+)\s(.*?)\s\("
    match = re.search(pattern, title_text)
    if match:
        team1 = match.group(1).strip()
        score1 = int(match.group(2))
        score2 = int(match.group(3))
        team2 = match.group(4).strip()
    else:
        print("Could not extract score from title text.")

    

    # Locate the “Team Stats” section
    # We inspect and see a section starting with <h2>Team Stats</h2> and then a table.
    section = soup.find("h2", string="Team Stats")
    if not section:
        return pd.DataFrame()
        raise RuntimeError("Could not find the Team Stats heading on the page")

    # The table is after this heading
    table = section.find_next("table")
    if not table:
        raise RuntimeError("Could not find the Team Stats table after the heading")

    team_imgs = soup.find_all("img", class_="TeamStats__Logo")
    column_header = team_imgs[0]["alt"]
    # Parse table into DataFrame
    df = pd.read_html(StringIO(str(table)))[0]

    # Detect column count
    if df.shape[1] >= 3:
        # Keep first three columns: Stat, Away Team, Home Team
        df = df.iloc[:, :3]
        df.columns = ['Stat', team1, team2]
    else:
        # Fallback: keep all columns
        df.columns = ['Stat'] + [f'Team{i}' for i in range(1, df.shape[1])]

    # Clean rows where 'Stat' is NaN
    df = df.dropna(subset=['Stat']).reset_index(drop=True)

    new_data = {}

    teams = [team1, team2]
    for index,team in enumerate(teams):
        for _, row in df.iterrows():
            # Clean up stat name: lowercase, replace spaces and punctuation with underscores
            stat_name = (
                row["Stat"]
                .strip()
                .lower()
                .replace(" ", "_")
                .replace(".", "")
                .replace("%", "pct")
            )
            if abbreviations[abbreviation] not in column_header:
                if index == 0:
                    new_col = f"team{index+2}_{stat_name}"
                else:
                    new_col = f"team{index}_{stat_name}"
            else:
                new_col = f"team{index+1}_{stat_name}"
            new_data[new_col] = [row[team]]

    # Create the new single-row DataFrame
    df_out = pd.DataFrame(new_data)
    df_out = df_out.reindex(sorted(df_out.columns), axis=1)
    if team1 == abbreviations[abbreviation]:
        df_out['spread'] = score2-score1
        # df_out.insert(0, 'team2_label', TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team2][0]))
        # df_out.insert(0, 'team1_label', TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team1][0]))
        df_out.insert(0, 'team2', team2)
        df_out.insert(0, 'team1', team1)
    elif abbreviations[abbreviation] not in column_header or team1 != abbreviations[abbreviation]:
        df_out['spread'] = score1-score2
        # df_out.insert(0, 'team2_label', TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team1][0]))
        # df_out.insert(0, 'team1_label', TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team2][0]))
        df_out.insert(0, 'team2', team1)
        df_out.insert(0, 'team1', team2)
    else:
        # df_out.insert(0, 'team2_label', TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team2][0]))
        # df_out.insert(0, 'team1_label', TEAM_ABBR.index([k for k, v in abbreviations.items() if v == team1][0]))
        df_out['spread'] = score2-score1
        df_out.insert(0, 'team2', team2)
        df_out.insert(0, 'team1', team1)

    new_cols = {}
    # Apply to entire dataframe
    for col in df_out.columns:
        # df_out[col] = df_out[col].apply(lambda x: process_value(col, x))
        if "penalties" in col.lower():
            # Split penalties into two new columns
            num_list = []
            yards_list = []
            for val in df_out[col]:
                match = re.match(r"^(\d+)-(\d+)$", str(val))
                if match:
                    num_list.append(int(match.group(1)))
                    yards_list.append(int(match.group(2)))
                else:
                    # If not matching, fill with original value or NaN
                    num_list.append(None)
                    yards_list.append(None)
            # Create new columns
            team = col.split("_")[0]  # e.g., 'team1'
            new_cols[f"{team}_numberpenalties"] = num_list
            new_cols[f"{team}_penaltyyards"] = yards_list
        else:
            # For other columns, do division if format number-number or number/number
            new_vals = []
            for val in df_out[col]:
                val_str = str(val)
                match_dash = re.match(r"^(\d+)-(\d+)$", val_str)
                match_slash = re.match(r"^(\d+)/(\d+)$", val_str)
                if match_dash:
                    if int(match_dash.group(2)) == 0:
                        new_vals.append(0)
                    else:
                        new_vals.append(float(match_dash.group(1)) / float(match_dash.group(2)))
                elif match_slash:
                    new_vals.append(float(match_slash.group(1)) / float(match_slash.group(2)))
                else:
                    new_vals.append(val)
            new_cols[col] = new_vals
    # Create new DataFrame
    df_new = pd.DataFrame(new_cols)
    return df_new


def main():
    all_stats = []
    for abbr in TEAM_ABBR:
        game_links = get_game_links(abbr)
        # game_links = [game_links[0]]
        
        for link in game_links:
            print("Fetching stats from:", link)
            stats = parse_boxscore(link, abbr)
            # print(stats)
            if not stats.empty:
                all_stats.append(stats)

            else:
                break
            # stats.to_csv('test.csv', index=False)
            # if stats:
            #     all_stats.append(stats)
            time.sleep(1)  # be polite
            # print(stats)

    
    # Combine into one DataFrame
    combined_df = pd.concat(all_stats, ignore_index=True)
    print(combined_df.head())

    # Save to CSV
    combined_df.to_csv("train_week12.csv", index=False)

    

if __name__ == "__main__":
    main()
