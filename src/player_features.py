"""
player_features.py
------------------
Builds player-level rolling stats from Cricsheet YAMLs and aggregates
them into team strength scores per match.

REQUIRES: data/raw/people.csv from https://cricsheet.org/register/people.csv
  Download it once:  curl -o data/raw/people.csv https://cricsheet.org/register/people.csv
  Or visit the URL and save manually.

This file maps full names (Virat Kohli) ↔ Cricsheet names (V Kohli)
automatically, so no manual override table is needed.
"""

import os, re, json, yaml
import pandas as pd
import numpy as np
from collections import defaultdict

RAW_PATH       = "data/raw"
PEOPLE_CSV     = "data/squads/people.csv"
SQUADS_PATH    = "data/squads/squads.json"
OUTPUT_PATH    = "data/processed/player_match_scores.csv"
CURRENT_OUTPUT = "data/processed/ipl_2026_team_scores.csv"

BATTING_WINDOW = 15
BOWLING_WINDOW = 12


# ── Name matching ──────────────────────────────────────────────────────────────

def load_name_map(people_csv_path, known_cricsheet_names):
    """
    Build full_name → cricsheet_name mapping using people.csv.
    people.csv columns: identifier, name, unique_name, ...
      - 'name' is the full name (Virat Kohli)
      - 'unique_name' is the Cricsheet name (V Kohli)
    Falls back to last-name matching if people.csv is missing.
    """
    full_to_cs = {}   # full_name (lower) → cricsheet_name

    if os.path.exists(people_csv_path):
        try:
            people = pd.read_csv(people_csv_path)
            # Column names vary slightly — find the right ones
            cols = [c.strip().lower() for c in people.columns]
            people.columns = cols

            name_col   = next((c for c in cols if c in ("name","full_name")), None)
            unique_col = next((c for c in cols if "unique" in c), None)
            key_col    = next((c for c in cols if c in ("key","identifier","id")), None)

            if name_col and unique_col:
                for _, row in people.iterrows():
                    full  = str(row[name_col]).strip()
                    cs    = str(row[unique_col]).strip()
                    if full and cs and full != "nan" and cs != "nan":
                        full_to_cs[full.lower()] = cs
                print(f"  Loaded {len(full_to_cs)} name mappings from people.csv")
            else:
                print(f"  people.csv columns: {cols}")
                print("  Could not find name/unique_name columns — falling back to last-name match")
        except Exception as e:
            print(f"  Warning: could not parse people.csv: {e}")
    else:
        print(f"  people.csv not found at {people_csv_path}")
        print("  Download from: https://cricsheet.org/register/people.csv")
        print("  Falling back to last-name matching (less accurate)")

    # Build last-name lookup from known Cricsheet names as fallback
    last_name_map = defaultdict(list)
    for name in known_cricsheet_names:
        last = name.split()[-1].lower()
        last_name_map[last].append(name)

    def resolve(full_name):
        # Clean input
        clean = re.sub(r'\s+', ' ', full_name).strip()

        # 1. people.csv exact match (case-insensitive)
        cs = full_to_cs.get(clean.lower())
        if cs and cs in known_cricsheet_names:
            return cs

        # 2. people.csv match but name not in known set — still use it
        if cs:
            return cs

        # 3. Exact match in known cricsheet names
        if clean in known_cricsheet_names:
            return clean

        # 4. Last name match (unique)
        last = clean.split()[-1].lower()
        candidates = last_name_map.get(last, [])
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            first_init = clean[0].upper()
            refined = [c for c in candidates if c[0].upper() == first_init]
            if len(refined) == 1:
                return refined[0]

        # 5. No match — return as-is
        return clean

    return resolve


def build_known_names(raw_path):
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
        for team_players in info.get("players", {}).values():
            known.update(team_players)
        for name in info.get("registry", {}).get("people", {}).keys():
            known.add(name)
    return known


# ── Delivery iterator ──────────────────────────────────────────────────────────

def iter_innings(data):
    for idx, inning in enumerate(data.get("innings", [])):
        if "team" in inning:
            batting_team = inning["team"]
            balls = []
            for over_data in inning.get("overs", []):
                over_num = over_data["over"]
                for ball in over_data.get("deliveries", []):
                    extras  = ball.get("extras", {})
                    wickets = ball.get("wickets", [])
                    batter  = ball.get("batter") or ball.get("batsman", "")
                    bowler  = ball.get("bowler", "")
                    runs_b  = ball.get("runs", {}).get("batter", 0)
                    runs_t  = ball.get("runs", {}).get("total", 0)
                    legal   = "wides" not in extras and "noballs" not in extras
                    is_wkt  = len(wickets) > 0
                    wk_kind = wickets[0].get("kind","") if is_wkt else ""
                    balls.append(dict(over=over_num, batter=batter, bowler=bowler,
                                      runs_bat=runs_b, runs_tot=runs_t,
                                      is_legal=legal, is_wkt=is_wkt, wkt_kind=wk_kind))
            yield idx, batting_team, balls
        else:
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
                        runs_b   = bd.get("runs", {}).get("batsman", 0)
                        runs_t   = bd.get("runs", {}).get("total", 0)
                        legal    = "wides" not in extras and "noballs" not in extras
                        is_wkt   = wicket is not None
                        wk_kind  = wicket.get("kind","") if is_wkt else ""
                        balls.append(dict(over=over_num, batter=batter, bowler=bowler,
                                          runs_bat=runs_b, runs_tot=runs_t,
                                          is_legal=legal, is_wkt=is_wkt, wkt_kind=wk_kind))
                yield idx, batting_team, balls


# ── Stat stores ────────────────────────────────────────────────────────────────

batter_innings = defaultdict(list)   # name → [{date, runs, balls}]
bowler_spells  = defaultdict(list)   # name → [{date, runs_given, wickets, legal_balls}]
matchup_log    = defaultdict(lambda: defaultdict(list))  # bat→bowl→[{date,runs,balls}]


def get_bat_score(players, before_date):
    scores = []
    for p in players:
        hist = [h for h in batter_innings[p] if h["date"] < before_date][-BATTING_WINDOW:]
        if not hist:
            continue
        runs  = sum(h["runs"]  for h in hist)
        balls = sum(h["balls"] for h in hist)
        n     = len(hist)
        avg   = runs / n
        sr    = (runs / balls * 100) if balls > 0 else 0
        scores.append(avg * sr / 100)
    # Normalise by squad size so larger squads don't get unfair advantage
    return round(np.mean(scores) * 11, 3) if scores else 0.0


def get_bowl_score(players, before_date):
    scores = []
    for p in players:
        hist = [h for h in bowler_spells[p] if h["date"] < before_date][-BOWLING_WINDOW:]
        if not hist:
            continue
        wkts  = sum(h["wickets"]     for h in hist)
        balls = sum(h["legal_balls"] for h in hist)
        runs  = sum(h["runs_given"]  for h in hist)
        if balls == 0:
            continue
        wr   = wkts / balls
        econ = runs / balls * 6
        scores.append((wr * 100) / max(econ, 1))
    return round(np.mean(scores) * 5, 3) if scores else 0.0   # ×5 for top bowlers


def get_matchup_score(batters, bowlers, before_date):
    srs = []
    for bat in batters:
        for bow in bowlers:
            hist = [h for h in matchup_log[bat][bow] if h["date"] < before_date]
            if len(hist) < 4:
                continue
            runs  = sum(h["runs"]  for h in hist)
            balls = sum(h["balls"] for h in hist)
            srs.append((runs / balls * 100) if balls > 0 else 100)
    return round(np.mean(srs), 3) if srs else 100.0


# ── Main ───────────────────────────────────────────────────────────────────────

def process_all():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/squads",    exist_ok=True)

    print("Building player name registry...")
    known = build_known_names(RAW_PATH)
    print(f"  {len(known)} unique Cricsheet player names found")

    resolve = load_name_map(PEOPLE_CSV, known)

    with open(SQUADS_PATH) as f:
        squad_json = json.load(f)

    # Map squad full names → cricsheet names
    squad_cs = {}   # team → [cricsheet_names]
    matched = unmatched = 0
    for team, players in squad_json.items():
        cs_names = []
        for p in players:
            cs = resolve(p)
            cs_names.append(cs)
            if cs in known:
                matched += 1
            else:
                unmatched += 1
        squad_cs[team] = cs_names

    print(f"  Squad name matching: {matched} matched, {unmatched} not in Cricsheet data")
    print(f"  (Players not in data are new to IPL or have very few appearances)")

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
        xi      = info.get("players", {})

        if len(teams) != 2 or not winner or not date:
            continue

        team1, team2 = sorted(teams)
        xi1 = xi.get(team1, [])
        xi2 = xi.get(team2, [])

        records.append({
            "date":   date,
            "team1":  team1,
            "team2":  team2,
            "winner": winner,
            "t1_bat_score":     get_bat_score(xi1,  date),
            "t1_bowl_score":    get_bowl_score(xi1, date),
            "t1_matchup_score": get_matchup_score(xi1, xi2, date),
            "t2_bat_score":     get_bat_score(xi2,  date),
            "t2_bowl_score":    get_bowl_score(xi2, date),
            "t2_matchup_score": get_matchup_score(xi2, xi1, date),
        })

        # Update stat stores after recording
        for _, batting_team, balls in iter_innings(data):
            b_balls  = defaultdict(int)
            b_runs   = defaultdict(int)
            bo_balls = defaultdict(int)
            bo_runs  = defaultdict(int)
            bo_wkts  = defaultdict(int)

            for b in balls:
                bat, bow = b["batter"], b["bowler"]
                if b["is_legal"]:
                    b_balls[bat] += 1
                    bo_balls[bow] += 1
                    if not b["is_wkt"] or b["wkt_kind"] == "run out":
                        b_runs[bat] += b["runs_bat"]
                    if bat and bow:
                        matchup_log[bat][bow].append(
                            {"date": date, "runs": b["runs_bat"], "balls": 1})
                bo_runs[bow] += b["runs_tot"]
                if b["is_wkt"] and b["wkt_kind"] not in ("run out","obstructing the field"):
                    bo_wkts[bow] += 1

            for batter, balls_faced in b_balls.items():
                batter_innings[batter].append(
                    {"date": date, "runs": b_runs.get(batter, 0), "balls": balls_faced})
            for bowler, lb in bo_balls.items():
                bowler_spells[bowler].append(
                    {"date": date, "runs_given": bo_runs.get(bowler,0),
                     "wickets": bo_wkts.get(bowler,0), "legal_balls": lb})

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPer-match scores saved: {OUTPUT_PATH}  ({len(df)} matches)")

    # Current squad scores for 2026
    print("\nCurrent squad strength (using people.csv name matching):")
    today = pd.Timestamp("2026-01-01")
    rows  = []
    for team, cs_names in squad_cs.items():
        bat  = get_bat_score(cs_names,  today)
        bowl = get_bowl_score(cs_names, today)
        rows.append({"team": team, "bat_score": bat, "bowl_score": bowl})

    cur = pd.DataFrame(rows).sort_values("bat_score", ascending=False)
    cur.to_csv(CURRENT_OUTPUT, index=False)

    for _, r in cur.iterrows():
        bar = "█" * int(r["bat_score"] / 3)
        print(f"  {r['team']:<35} bat={r['bat_score']:6.1f}  bowl={r['bowl_score']:5.2f}  {bar}")


if __name__ == "__main__":
    process_all()