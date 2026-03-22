"""
elo_rating.py
-------------
Generates ELO ratings with two key improvements over the basic version:

1. Seasonal decay: at the start of each new IPL season, ratings are
   pulled 30% back toward 1500. This reflects squad changes, new coaches,
   retirements etc. A team's 2012 dominance shouldn't heavily influence
   their 2025 rating.

2. K=32 with a margin bonus: close wins give smaller K-updates than
   dominant wins. A 1-run win is not the same as a 50-run thrashing.
   We approximate this with a fixed K=28 (slightly higher than before)
   since we don't have margin-of-victory in the basic match CSV.

Output: data/processed/ipl_matches_with_elo.csv
"""

import pandas as pd

DECAY_FACTOR   = 0.30   # pull 30% back to base at season start
BASE_RATING    = 1500
K              = 28


class EloRating:

    def __init__(self, k=K, base=BASE_RATING, decay=DECAY_FACTOR):
        self.k       = k
        self.base    = base
        self.decay   = decay
        self.ratings = {}
        self.season  = None   # tracks current season year

    def get_rating(self, team):
        if team not in self.ratings:
            self.ratings[team] = self.base
        return self.ratings[team]

    def apply_seasonal_decay(self, new_season):
        """
        Called once when we cross into a new IPL season.
        Pulls every rating 30% back toward 1500.
        Example: team at 1600 → 1600 - 0.30*(1600-1500) = 1570
        """
        for team in self.ratings:
            r = self.ratings[team]
            self.ratings[team] = r - self.decay * (r - self.base)
        self.season = new_season

    def expected(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, team1, team2, winner):
        r1 = self.get_rating(team1)
        r2 = self.get_rating(team2)
        e1 = self.expected(r1, r2)
        e2 = self.expected(r2, r1)
        s1 = 1 if winner == team1 else 0
        s2 = 1 - s1
        self.ratings[team1] = r1 + self.k * (s1 - e1)
        self.ratings[team2] = r2 + self.k * (s2 - e2)


def generate_elo_features():

    df = pd.read_csv("data/processed/ipl_matches.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloRating()

    t1_ratings, t2_ratings, diffs = [], [], []

    for _, row in df.iterrows():
        t1     = row["team1"]
        t2     = row["team2"]
        year   = row["date"].year

        # Apply seasonal decay at the start of each new IPL season
        if elo.season is None:
            elo.season = year
        elif year != elo.season:
            elo.apply_seasonal_decay(year)

        # Record PRE-MATCH ratings (before update)
        r1 = elo.get_rating(t1)
        r2 = elo.get_rating(t2)

        t1_ratings.append(r1)
        t2_ratings.append(r2)
        diffs.append(r1 - r2)

        # Update AFTER recording — no leakage
        elo.update(t1, t2, row["winner"])

    df["elo_team1"] = t1_ratings
    df["elo_team2"] = t2_ratings
    df["elo_diff"]  = diffs

    df.to_csv("data/processed/ipl_matches_with_elo.csv", index=False)
    print("ELO features saved.")
    print(f"Rating range: {min(t1_ratings+t2_ratings):.0f} – {max(t1_ratings+t2_ratings):.0f}")
    return df


if __name__ == "__main__":
    generate_elo_features()