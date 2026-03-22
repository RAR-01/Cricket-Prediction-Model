"""
player_features.py
------------------
Builds player-level rolling stats from Cricsheet YAMLs, then aggregates
them into three per-team strength scores for each match:

  bat_score     — weighted batting quality of this team's squad
                  (rolling avg × strike rate for each batter, summed)
  bowl_score    — weighted bowling quality of this team's squad
                  (rolling wicket rate / economy for each bowler, summed)
  matchup_score — how well this team's batters have historically performed
                  against the SPECIFIC players in the opponent's squad
                  (captures Kohli vs Bumrah type matchups)

All stats use only matches BEFORE the current one — no leakage.

Also builds ipl_2026_team_scores.csv — the current squad strength scores
for predicting upcoming 2026 matches.

Run order:
  1. data_parser.py
  2. elo_rating.py
  3. player_features.py        ← this file
  4. feature_engineering.py
  5. train_model.py
"""

import os
import re
import json
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict

RAW_PATH        = "data/raw"
SQUADS_PATH     = "data/squads/squads.json"
OUTPUT_PATH     = "data/processed/player_match_scores.csv"
CURRENT_OUTPUT  = "data/processed/ipl_2026_team_scores.csv"

BATTING_WINDOW  = 15
BOWLING_WINDOW  = 12


# ── Name normalisation ─────────────────────────────────────────────────────────
# Cricsheet uses abbreviated names (V Kohli, RG Sharma).
# Our squad JSON has full names (Virat Kohli, Rohit Sharma).
# Strategy: build a full-name → cricsheet-name map from the YAML registries,
# then also fuzzy-match on last name as a fallback.

def build_name_registry(raw_path):
    """
    Scan all YAMLs and collect every player name that appears in:
      - info.players (playing XI lists)
      - info.registry.people
    Returns: set of all known cricsheet names
    """
    known = set()
    for fname in os.listdir(raw_path):
        if not fname.endswith(".yaml"):
            continue
        with open(os.path.join(raw_path, fname), encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except Exception:
                continue
        info = data.get("info", {})
        # From playing XI
        for team_players in info.get("players", {}).values():
            known.update(team_players)
        # From registry
        for name in info.get("registry", {}).get("people", {}).keys():
            known.add(name)
    return known


def build_full_to_cricsheet(known_names, squad_json):
    """
    For each full name in squad_json, find the best matching cricsheet name.

    Matching priority:
      1. Exact match
      2. Last name match (unique)
      3. Last two tokens match
      4. Manual overrides for known mismatches
    """
    # Manual overrides — common mismatches between ESPNCricinfo and Cricsheet
    OVERRIDES = {
        "Virat Kohli":           "V Kohli",
        "Rohit Sharma":          "RG Sharma",
        "Jasprit Bumrah":        "JJ Bumrah",
        "Hardik Pandya":         "HH Pandya",
        "Ravindra Jadeja":       "RA Jadeja",
        "Yashaswi Jaiswal":      "YBK Jaiswal",
        "Pat Cummins":           "PJ Cummins",
        "Travis Head":           "TM Head",
        "Suryakumar Yadav":      "SA Yadav",
        "Shubman Gill":          "Shubman Gill",
        "Mohammed Shami":        "Mohammed Shami",
        "Bhuvneshwar Kumar":     "B Kumar",
        "Yuzvendra Chahal":      "YS Chahal",
        "Kuldeep Yadav":         "Kuldeep Yadav",
        "Axar Patel":            "AR Patel",
        "KL Rahul":              "KL Rahul",
        "MS Dhoni":              "MS Dhoni",
        "Rishabh Pant":          "RR Pant",
        "Sanju Samson":          "SV Samson",
        "Shreyas Iyer":          "SS Iyer",
        "Ruturaj Gaikwad":       "RD Gaikwad",
        "Rinku Singh":           "Rinku Singh",
        "Varun Chakaravarthy":   "Varun Chakravarthy",
        "Tilak Verma":           "T Verma",
        "Sunil Narine":          "SP Narine",
        "Ajinkya Rahane":        "AM Rahane",
        "Manish Pandey":         "MK Pandey",
        "Abhishek Sharma":       "Abhishek Sharma",
        "Heinrich Klaasen":      "HE van der Dussen",
        "Nicholas Pooran":       "N Pooran",
        "Shimron Hetmyer":       "SO Hetmyer",
        "Sam Curran":            "SM Curran",
        "Jos Buttler":           "JC Buttler",
        "Kagiso Rabada":         "K Rabada",
        "Rashid Khan":           "Rashid Khan",
        "Trent Boult":           "TA Boult",
        "Mitchell Starc":        "MA Starc",
        "Jofra Archer":          "JC Archer",
        "Josh Hazlewood":        "JR Hazlewood",
        "Arshdeep Singh":        "Arshdeep Singh",
        "Deepak Chahar":         "DL Chahar",
        "Shardul Thakur":        "Shardul Thakur",
        "Harshal Patel":         "Harshal Patel",
        "Avesh Khan":            "Avesh Khan",
        "Rahul Tewatia":         "RK Tewatia",
        "Sai Sudharsan":         "B Sai Sudharsan",
        "Devdutt Padikkal":      "Devdutt Padikkal",
        "Rajat Patidar":         "RR Patidar",
        "Riyan Parag":           "Riyan Parag",
        "Dhruv Jurel":           "Dhruv Jurel",
        "Shivam Dube":           "Shivam Dube",
        "Ishan Kishan":          "IS Kishan",
        "Prabhsimran Singh":     "Prabhsimran Singh",
        "Mitchell Marsh":        "MR Marsh",
        "Marco Jansen":          "MA Jansen",
        "Lockie Ferguson":       "LH Ferguson",
        "Marcus Stoinis":        "MP Stoinis",
        "Nitish Kumar Reddy":    "N Kumar Reddy",
        "Tim David":             "Tim David",
        "Phil Salt":             "PD Salt",
        "Jacob Bethell":         "JMM Bethell",
        "Krunal Pandya":         "KH Pandya",
        "Washington Sundar":     "W Sundar",
        "Prasidh Krishna":       "PJ Krishna",
        "Mohammad Siraj":        "Mohammed Siraj",
        "T Natarajan":           "T Natarajan",
        "Vaibhav Arora":         "Vaibhav Arora",
        "Harshit Rana":          "Harshit Rana",
        "Mayank Yadav":          "Mayank Yadav",
        "Tushar Deshpande":      "T Deshpande",
        "Yash Dayal":            "Yash Dayal",
        "Sandeep Sharma":        "Sandeep Sharma",
        "Noor Ahmad":            "Noor Ahmad",
        "Angkrish Raghuvanshi":  "Angkrish Raghuvanshi",
        "Ramandeep Singh":       "Ramandeep Singh",
        "Rovman Powell":         "R Powell",
        "Umran Malik":           "Umran Malik",
    }

    # Build last-name lookup for fuzzy matching
    last_name_map = defaultdict(list)
    for name in known_names:
        last = name.split()[-1].lower()
        last_name_map[last].append(name)

    mapping = {}
    all_squad_players = set()
    for players in squad_json.values():
        all_squad_players.update(players)

    for full_name in all_squad_players:
        # 1. Manual override
        if full_name in OVERRIDES:
            mapping[full_name] = OVERRIDES[full_name]
            continue

        # 2. Exact match in cricsheet
        if full_name in known_names:
            mapping[full_name] = full_name
            continue

        # 3. Last name match
        last = full_name.split()[-1].lower()
        candidates = last_name_map.get(last, [])
        if len(candidates) == 1:
            mapping[full_name] = candidates[0]
            continue
        elif len(candidates) > 1:
            # Multiple candidates — try matching first initial too
            first_initial = full_name[0].upper()
            refined = [c for c in candidates if c[0].upper() == first_initial]
            if len(refined) == 1:
                mapping[full_name] = refined[0]
                continue

        # 4. No match — will use full_name as-is (may not appear in data)
        mapping[full_name] = full_name

    return mapping


# ── Delivery iterator (handles both Cricsheet formats) ────────────────────────

def iter_innings(data):
    for idx, inning in enumerate(data.get("innings", [])):
        if "team" in inning:                          # new format
            batting_team = inning["team"]
            balls = []
            for over_data in inning.get("overs", []):
                over_num = over_data["over"]
                for ball in over_data.get("deliveries", []):
                    extras   = ball.get("extras", {})
                    wickets  = ball.get("wickets", [])
                    batter   = ball.get("batter") or ball.get("batsman", "")
                    bowler   = ball.get("bowler", "")
                    runs_bat = ball.get("runs", {}).get("batter", 0)
                    runs_tot = ball.get("runs", {}).get("total", 0)
                    is_legal = "wides" not in extras and "noballs" not in extras
                    is_wkt   = len(wickets) > 0
                    wkt_kind = wickets[0].get("kind", "") if is_wkt else ""
                    balls.append(dict(over=over_num, batter=batter, bowler=bowler,
                                      runs_bat=runs_bat, runs_tot=runs_tot,
                                      is_legal=is_legal, is_wkt=is_wkt, wkt_kind=wkt_kind))
            yield idx, batting_team, balls
        else:                                          # old format
            for inning_key, inning_data in inning.items():
                batting_team = inning_data.get("team", "")
                balls = []
                for delivery in inning_data.get("deliveries", []):
                    for over_ball, bd in delivery.items():
                        over_num = int(float(over_ball))
                        extras   = bd.get("extras", {})
                        wicket   = bd.get("wicket")
                        batter   = bd.get("batsman", "")
                        bowler   = bd.get("bowler", "")
                        runs_bat = bd.get("runs", {}).get("batsman", 0)
                        runs_tot = bd.get("runs", {}).get("total", 0)
                        is_legal = "wides" not in extras and "noballs" not in extras
                        is_wkt   = wicket is not None
                        wkt_kind = wicket.get("kind", "") if is_wkt else ""
                        balls.append(dict(over=over_num, batter=batter, bowler=bowler,
                                          runs_bat=runs_bat, runs_tot=runs_tot,
                                          is_legal=is_legal, is_wkt=is_wkt, wkt_kind=wkt_kind))
                yield idx, batting_team, balls


# ── Stat stores ───────────────────────────────────────────────────────────────
# batter_innings[player] = list of {date, runs, balls_faced}
# bowler_spells[player]  = list of {date, runs_given, wickets, legal_balls}
# matchup[batter][bowler]= list of {date, runs, balls}

batter_innings = defaultdict(list)
bowler_spells  = defaultdict(list)
matchup_log    = defaultdict(lambda: defaultdict(list))


def get_bat_score(players, before_date):
    """Rolling batting quality for a list of players."""
    scores = []
    for p in players:
        history = [h for h in batter_innings[p] if h["date"] < before_date]
        history = history[-BATTING_WINDOW:]
        if not history:
            scores.append(0)
            continue
        total_runs  = sum(h["runs"]  for h in history)
        total_balls = sum(h["balls"] for h in history)
        innings     = len(history)
        avg = total_runs / innings
        sr  = (total_runs / total_balls * 100) if total_balls > 0 else 0
        scores.append(avg * sr / 100)   # composite: avg × SR/100
    return round(sum(scores), 3)


def get_bowl_score(players, before_date):
    """Rolling bowling quality for a list of players."""
    scores = []
    for p in players:
        history = [h for h in bowler_spells[p] if h["date"] < before_date]
        history = history[-BOWLING_WINDOW:]
        if not history:
            scores.append(0)
            continue
        total_wkts  = sum(h["wickets"]     for h in history)
        total_balls = sum(h["legal_balls"] for h in history)
        total_runs  = sum(h["runs_given"]  for h in history)
        if total_balls == 0:
            scores.append(0)
            continue
        wicket_rate = total_wkts  / total_balls        # wickets per ball
        economy     = total_runs  / total_balls * 6    # runs per over
        bowl_quality = (wicket_rate * 100) / max(economy, 1)
        scores.append(bowl_quality)
    return round(sum(scores), 3)


def get_matchup_score(batters, bowlers, before_date):
    """
    How well `batters` perform against `bowlers` historically.
    Higher = batters dominate. Lower = bowlers dominate.
    """
    scores = []
    for batter in batters:
        for bowler in bowlers:
            history = [h for h in matchup_log[batter][bowler]
                       if h["date"] < before_date]
            if len(history) < 3:      # need at least 3 balls to be meaningful
                continue
            runs  = sum(h["runs"]  for h in history)
            balls = sum(h["balls"] for h in history)
            sr    = (runs / balls * 100) if balls > 0 else 100
            scores.append(sr)
    return round(np.mean(scores), 3) if scores else 100.0  # 100 = neutral


# ── Main loop ─────────────────────────────────────────────────────────────────

def process_all():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/squads",    exist_ok=True)

    print("Building player name registry from YAMLs...")
    known_names = build_name_registry(RAW_PATH)
    print(f"  {len(known_names)} unique player names found in Cricsheet data")

    # Load squad JSON
    if not os.path.exists(SQUADS_PATH):
        print(f"ERROR: {SQUADS_PATH} not found. Create it first.")
        return

    with open(SQUADS_PATH) as f:
        squad_json = json.load(f)

    name_map = build_full_to_cricsheet(known_names, squad_json)
    print(f"  Mapped {len(name_map)} squad players to Cricsheet names")

    # Build cricsheet-name → team map for current squads
    cricsheet_squad = {}   # cricsheet_name -> team
    for team, players in squad_json.items():
        for p in players:
            cs_name = name_map.get(p, p)
            cricsheet_squad[cs_name] = team

    # Sort YAMLs by date for chronological processing
    files = sorted(f for f in os.listdir(RAW_PATH) if f.endswith(".yaml"))
    print(f"Processing {len(files)} match files...")

    records = []

    for fname in files:
        fpath = os.path.join(RAW_PATH, fname)
        with open(fpath, encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except Exception:
                continue

        info    = data.get("info", {})
        teams   = info.get("teams", [])
        outcome = info.get("outcome", {})
        winner  = outcome.get("winner")
        dates   = info.get("dates", [])
        date    = pd.to_datetime(dates[0]) if dates else None
        players_xi = info.get("players", {})  # {team: [player_list]}

        if len(teams) != 2 or winner is None or date is None:
            continue

        team1, team2 = sorted(teams)

        # Get playing XIs for this match
        xi_t1 = players_xi.get(team1, [])
        xi_t2 = players_xi.get(team2, [])

        # ── Compute pre-match scores (using data BEFORE this match) ───────
        rec = {
            "date":    date,
            "team1":   team1,
            "team2":   team2,
            "winner":  winner,
            # Team1 scores
            "t1_bat_score":     get_bat_score(xi_t1,  date),
            "t1_bowl_score":    get_bowl_score(xi_t1, date),
            "t1_matchup_score": get_matchup_score(xi_t1, xi_t2, date),
            # Team2 scores
            "t2_bat_score":     get_bat_score(xi_t2,  date),
            "t2_bowl_score":    get_bowl_score(xi_t2, date),
            "t2_matchup_score": get_matchup_score(xi_t2, xi_t1, date),
        }
        records.append(rec)

        # ════ UPDATE STAT STORES AFTER RECORDING ════════════════════════════
        for inning_idx, batting_team, balls in iter_innings(data):
            # Track each batter's innings
            batter_ball_counts = defaultdict(int)
            batter_run_counts  = defaultdict(int)

            for b in balls:
                bat = b["batter"]
                bow = b["bowler"]

                if b["is_legal"]:
                    batter_ball_counts[bat] += 1
                    if not b["is_wkt"] or b["wkt_kind"] == "run out":
                        batter_run_counts[bat] += b["runs_bat"]

                # Matchup log: every legal ball batter faced vs this bowler
                if b["is_legal"] and bat and bow:
                    matchup_log[bat][bow].append({
                        "date":  date,
                        "runs":  b["runs_bat"],
                        "balls": 1,
                    })

            # Store batter innings summaries
            for batter, balls_faced in batter_ball_counts.items():
                batter_innings[batter].append({
                    "date":  date,
                    "runs":  batter_run_counts.get(batter, 0),
                    "balls": balls_faced,
                })

            # Store bowler spell summaries
            bowler_ball_counts  = defaultdict(int)
            bowler_run_counts   = defaultdict(int)
            bowler_wkt_counts   = defaultdict(int)

            for b in balls:
                bow = b["bowler"]
                if not bow:
                    continue
                if b["is_legal"]:
                    bowler_ball_counts[bow] += 1
                bowler_run_counts[bow]  += b["runs_tot"]
                if b["is_wkt"] and b["wkt_kind"] not in ("run out", "obstructing the field"):
                    bowler_wkt_counts[bow] += 1

            for bowler, legal_balls in bowler_ball_counts.items():
                bowler_spells[bowler].append({
                    "date":         date,
                    "runs_given":   bowler_run_counts.get(bowler, 0),
                    "wickets":      bowler_wkt_counts.get(bowler, 0),
                    "legal_balls":  legal_balls,
                })

    # ── Save per-match scores ─────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPer-match player scores saved: {OUTPUT_PATH}")
    print(f"Matches: {len(df)}")

    # ── Compute CURRENT squad scores for 2026 predictions ────────────────────
    print("\nComputing current squad strength scores for 2026...")
    today = pd.Timestamp("2026-01-01")   # use all data up to now

    current_rows = []
    for team, full_names in squad_json.items():
        cs_names = [name_map.get(p, p) for p in full_names]
        bat  = get_bat_score(cs_names,  today)
        bowl = get_bowl_score(cs_names, today)
        current_rows.append({"team": team, "bat_score": bat, "bowl_score": bowl})

    current_df = pd.DataFrame(current_rows).sort_values("bat_score", ascending=False)
    current_df.to_csv(CURRENT_OUTPUT, index=False)

    print("\nCurrent squad strength scores (batting):")
    for _, r in current_df.iterrows():
        bar = "█" * int(r["bat_score"] / 5)
        print(f"  {r['team']:<35} bat={r['bat_score']:6.1f}  bowl={r['bowl_score']:5.2f}  {bar}")

    return df


if __name__ == "__main__":
    process_all()