"""
feature_engineering.py
-----------------------
Merges ELO ratings + player strength scores into the final training dataset.

Features (16 total):
  ELO:      elo_diff, elo_diff_sq
  Form:     team1_recent_winrate, team2_recent_winrate
  History:  head_to_head, venue_winrate_team1
  Players:  t1_bat_score, t2_bat_score, t1_bowl_score, t2_bowl_score,
            bat_score_diff, bowl_score_diff, matchup_diff
  Context:  toss_winner_is_team1, t1_season_match_num, t2_season_match_num

Run order:
  1. data_parser.py
  2. elo_rating.py
  3. player_features.py
  4. feature_engineering.py   ← this file
  5. train_model.py
"""

import pandas as pd
import numpy as np
from collections import defaultdict

ELO_PATH    = "data/processed/ipl_matches_with_elo.csv"
PLAYER_PATH = "data/processed/player_match_scores.csv"
OUTPUT_PATH = "data/processed/final_dataset.csv"

FORM_WINDOW = 10


def create_features():

    elo_df    = pd.read_csv(ELO_PATH,    parse_dates=["date"])
    player_df = pd.read_csv(PLAYER_PATH, parse_dates=["date"])

    elo_df    = elo_df.sort_values("date").reset_index(drop=True)
    player_df = player_df.sort_values("date").reset_index(drop=True)

    merged = elo_df.merge(
        player_df[["date","team1","team2",
                   "t1_bat_score","t1_bowl_score","t1_matchup_score",
                   "t2_bat_score","t2_bowl_score","t2_matchup_score"]],
        on=["date","team1","team2"],
        how="left"
    )

    for col in ["t1_bat_score","t1_bowl_score","t1_matchup_score",
                "t2_bat_score","t2_bowl_score","t2_matchup_score"]:
        merged[col] = merged[col].fillna(0)

    recent_results = defaultdict(list)
    h2h            = defaultdict(lambda: {"wins": 0, "total": 0})
    venue_results  = defaultdict(lambda: {"wins": 0, "total": 0})
    season_counts  = defaultdict(int)
    current_season = {}

    rows = []

    for _, row in merged.iterrows():
        t1          = row["team1"]
        t2          = row["team2"]
        winner      = row["winner"]
        toss_winner = row["toss_winner"]
        venue       = row["venue"]
        match_date  = row["date"]
        elo_diff    = row["elo_diff"]
        t1_won      = (winner == t1)
        year        = match_date.year

        for team in (t1, t2):
            if current_season.get(team) != year:
                current_season[team] = year
                season_counts[team]  = 0

        def win_rate(team):
            res = recent_results[team][-FORM_WINDOW:]
            return round(sum(res) / len(res), 4) if res else 0.5

        key    = (t1, t2)
        h2h_s  = h2h[key]
        h2h_wr = round(h2h_s["wins"] / h2h_s["total"], 4) if h2h_s["total"] > 0 else 0.5

        vk1      = (venue, t1)
        vs       = venue_results[vk1]
        venue_wr = round(vs["wins"] / vs["total"], 4) if vs["total"] > 0 else 0.5

        elo_diff_sq = np.sign(elo_diff) * (elo_diff ** 2) / 10000

        bat_diff     = row["t1_bat_score"]     - row["t2_bat_score"]
        bowl_diff    = row["t1_bowl_score"]    - row["t2_bowl_score"]
        matchup_diff = row["t1_matchup_score"] - row["t2_matchup_score"]

        rows.append({
            "date":   match_date, "team1": t1, "team2": t2, "winner": winner,
            "elo_diff":             elo_diff,
            "elo_diff_sq":          round(elo_diff_sq, 4),
            "team1_recent_winrate": win_rate(t1),
            "team2_recent_winrate": win_rate(t2),
            "head_to_head":         h2h_wr,
            "venue_winrate_team1":  venue_wr,
            "t1_bat_score":         row["t1_bat_score"],
            "t2_bat_score":         row["t2_bat_score"],
            "t1_bowl_score":        row["t1_bowl_score"],
            "t2_bowl_score":        row["t2_bowl_score"],
            "bat_score_diff":       round(bat_diff, 3),
            "bowl_score_diff":      round(bowl_diff, 3),
            "matchup_diff":         round(matchup_diff, 3),
            "toss_winner_is_team1": 1 if toss_winner == t1 else 0,
            "t1_season_match_num":  season_counts[t1],
            "t2_season_match_num":  season_counts[t2],
            "target": 1 if t1_won else 0,
        })

        recent_results[t1].append(1 if t1_won else 0)
        recent_results[t2].append(0 if t1_won else 1)
        h2h[key]["total"] += 1
        if t1_won:
            h2h[key]["wins"] += 1
        venue_results[(venue, t1)]["total"] += 1
        venue_results[(venue, t2)]["total"] += 1
        if t1_won:
            venue_results[(venue, t1)]["wins"] += 1
        else:
            venue_results[(venue, t2)]["wins"] += 1
        season_counts[t1] += 1
        season_counts[t2] += 1

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_PATH, index=False)
    n = len([c for c in final_df.columns if c not in ("date","team1","team2","winner","target")])
    print(f"Final dataset saved : {OUTPUT_PATH}")
    print(f"Rows: {len(final_df)}  |  Features: {n}  |  Balance: {final_df['target'].mean():.1%}")
    return final_df


if __name__ == "__main__":
    create_features()