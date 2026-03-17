import pandas as pd

class EloRating:
    
    def __init__(self, k=32, base_rating=1500):
        self.k = k
        self.base_rating = base_rating
        self.ratings = {}  # stores team ratings

    def get_rating(self, team):
        # If team not seen before → assign base rating
        if team not in self.ratings:
            self.ratings[team] = self.base_rating
        return self.ratings[team]

    def expected_score(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update_ratings(self, team1, team2, winner):
        
        r1 = self.get_rating(team1)
        r2 = self.get_rating(team2)

        e1 = self.expected_score(r1, r2)
        e2 = self.expected_score(r2, r1)

        # Actual result
        s1 = 1 if winner == team1 else 0
        s2 = 1 if winner == team2 else 0

        # Update ratings
        self.ratings[team1] = r1 + self.k * (s1 - e1)
        self.ratings[team2] = r2 + self.k * (s2 - e2)


def generate_elo_features():

    df = pd.read_csv("data/processed/ipl_matches.csv")

    # IMPORTANT: sort by date to avoid data leakage
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo = EloRating()

    elo_team1 = []
    elo_team2 = []
    elo_diff = []

    for _, row in df.iterrows():

        t1 = row["team1"]
        t2 = row["team2"]

        r1 = elo.get_rating(t1)
        r2 = elo.get_rating(t2)

        elo_team1.append(r1)
        elo_team2.append(r2)
        elo_diff.append(r1 - r2)

        # update AFTER storing (very important!)
        elo.update_ratings(t1, t2, row["winner"])

    df["elo_team1"] = elo_team1
    df["elo_team2"] = elo_team2
    df["elo_diff"] = elo_diff

    df.to_csv("data/processed/ipl_matches_with_elo.csv", index=False)

    print("ELO features added and saved.")


if __name__ == "__main__":
    generate_elo_features()