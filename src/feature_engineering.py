import pandas as pd
from collections import defaultdict

INPUT_PATH = "data/processed/ipl_matches_with_elo.csv"
OUTPUT_PATH = "data/processed/final_dataset.csv"


def create_features():

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Track stats
    last_5_results = defaultdict(list)
    head_to_head = defaultdict(lambda: {"wins": 0, "matches": 0})

    team1_recent = []
    team2_recent = []
    h2h_feature = []
    toss_binary = []

    for _, row in df.iterrows():

        t1 = row["team1"]
        t2 = row["team2"]
        winner = row["winner"]
        toss_winner = row["toss_winner"]

        # ---------- Recent form (last 5 matches) ----------
        def recent_win_rate(team):
            results = last_5_results[team][-5:]
            if len(results) == 0:
                return 0.5  # neutral
            return sum(results) / len(results)

        team1_recent.append(recent_win_rate(t1))
        team2_recent.append(recent_win_rate(t2))

        # ---------- Head-to-head ----------
        key = tuple(sorted([t1, t2]))
        h2h = head_to_head[key]

        if h2h["matches"] == 0:
            h2h_feature.append(0.5)
        else:
            # win rate for team1
            if t1 < t2:
                h2h_feature.append(h2h["wins"] / h2h["matches"])
            else:
                h2h_feature.append(1 - (h2h["wins"] / h2h["matches"]))

        # ---------- Toss feature ----------
        toss_binary.append(1 if toss_winner == t1 else 0)

        # ---------- Update stats AFTER ----------
        last_5_results[t1].append(1 if winner == t1 else 0)
        last_5_results[t2].append(1 if winner == t2 else 0)

        h2h["matches"] += 1
        if winner == min(t1, t2):
            h2h["wins"] += 1

    # Add features
    df["team1_recent_winrate"] = team1_recent
    df["team2_recent_winrate"] = team2_recent
    df["head_to_head"] = h2h_feature
    df["toss_winner_is_team1"] = toss_binary

    # Target variable
    df["target"] = (df["winner"] == df["team1"]).astype(int)

    # Select features
    final_df = df[
        [
            "elo_diff",
            "team1_recent_winrate",
            "team2_recent_winrate",
            "head_to_head",
            "toss_winner_is_team1",
            "target",
        ]
    ]

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Final dataset created:", OUTPUT_PATH)


if __name__ == "__main__":
    create_features()