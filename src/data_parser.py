import os
import yaml
import pandas as pd

RAW_DATA_PATH = "data/raw"
OUTPUT_PATH = "data/processed/ipl_matches.csv"


def parse_matches():

    matches = []

    for file in os.listdir(RAW_DATA_PATH):

        if not file.endswith(".yaml"):
            continue

        filepath = os.path.join(RAW_DATA_PATH, file)

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        info = data.get("info", {})

        teams = info.get("teams", [])

        if len(teams) != 2:
            continue

        team1, team2 = teams

        venue = info.get("venue")
        city = info.get("city")

        dates = info.get("dates", [])
        date = dates[0] if dates else None

        toss = info.get("toss", {})
        toss_winner = toss.get("winner")
        toss_decision = toss.get("decision")

        outcome = info.get("outcome", {})
        winner = outcome.get("winner")

        if winner is None:
            continue

        matches.append({
            "date": date,
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "city": city,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner
        })

    df = pd.DataFrame(matches)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset created:", OUTPUT_PATH)
    print("Total matches:", len(df))


if __name__ == "__main__":
    parse_matches()