"""
player_features.py  (v4 — fixed name resolver)
------------------------------------------------
Key fix from v3: the name resolver now strips disambiguation suffixes
from people.csv entries before doing the YAML lookup.

people.csv uses 'Rashid Khan (3)' to distinguish between multiple
cricketers named Rashid Khan globally. But IPL YAMLs just say
'Rashid Khan'. This fix strips the ' (3)' suffix so the lookup
correctly finds 100+ IPL appearances for the GT spinner.

Same fix applies to Abdul Samad (3), Mohsin Khan (2), and any
other player whose people.csv unique_name has a number suffix.

DO NOT edit people.csv to remove these suffixes — they exist for
good reason (disambiguation across all global cricketers). Fix it
in code instead.

Run order:
  1. data_parser.py
  2. elo_rating.py
  3. player_features.py        ← this file
  4. feature_engineering.py
  5. train_model.py
"""

import os, re, json, yaml
import pandas as pd
import numpy as np
from collections import defaultdict

RAW_PATH       = "data/raw"
PEOPLE_CSV     = "data/raw/people.csv"
SQUADS_PATH    = "data/squads/squads.json"
OUTPUT_PATH    = "data/processed/player_match_scores.csv"
CURRENT_OUTPUT = "data/processed/ipl_2026_team_scores.csv"

BATTING_WINDOW = 15
BOWLING_WINDOW = 12


# ── Name matching ──────────────────────────────────────────────────────────────

def load_name_map(people_csv_path, known_cricsheet_names):
    """
    Build a resolver function: full_name (ESPNCricinfo) → cricsheet_name (YAML).

    Uses people.csv as primary source, last-name matching as fallback.

    Critical fix: people.csv uses disambiguation suffixes like 'Rashid Khan (3)'
    to distinguish global cricketers with the same name. IPL YAMLs just use
    'Rashid Khan'. We strip the suffix before the YAML lookup so these players
    are correctly matched to their full IPL history.
    """
    full_to_cs = {}

    if os.path.exists(people_csv_path):
        try:
            people = pd.read_csv(people_csv_path)
            cols   = [c.strip().lower() for c in people.columns]
            people.columns = cols
            name_col   = next((c for c in cols if c in ("name", "full_name")), None)
            unique_col = next((c for c in cols if "unique" in c), None)

            if name_col and unique_col:
                for _, row in people.iterrows():
                    full = str(row[name_col]).strip()
                    cs   = str(row[unique_col]).strip()
                    if full and cs and full != "nan" and cs != "nan":
                        full_to_cs[full.lower()] = cs
                print(f"  Loaded {len(full_to_cs)} name mappings from people.csv")
            else:
                print(f"  Warning: could not find name/unique columns in people.csv")
                print(f"  Columns found: {cols}")
        except Exception as e:
            print(f"  Warning: could not parse people.csv: {e}")
    else:
        print(f"  people.csv not found at {people_csv_path}")
        print(f"  Download from: https://cricsheet.org/register/people.csv")
        print(f"  Falling back to last-name matching only")

    # Build last-name lookup from known Cricsheet names as fallback
    last_name_map = defaultdict(list)
    for name in known_cricsheet_names:
        last = name.split()[-1].lower()
        last_name_map[last].append(name)

    def resolve(full_name):
        """
        Resolve a full name to the Cricsheet name used in YAML files.
        Priority order:
          1. people.csv exact match (after stripping number suffix)
          2. Direct match in known Cricsheet names
          3. Last-name match (unique)
          4. Last-name + first-initial match
          5. Return as-is (no match found)
        """
        clean = re.sub(r'\s+', ' ', full_name).strip()

        # ── 1. people.csv lookup ───────────────────────────────────────────────
        cs = full_to_cs.get(clean.lower())
        if cs:
            # Strip disambiguation suffix: 'Rashid Khan (3)' → 'Rashid Khan'
            # This suffix exists in people.csv to distinguish global cricketers
            # with the same name, but IPL YAMLs don't use the suffix.
            cs_stripped = re.sub(r'\s*\(\d+\)\s*$', '', cs).strip()

            if cs_stripped in known_cricsheet_names:
                return cs_stripped   # clean name found in YAMLs — best case
            if cs in known_cricsheet_names:
                return cs            # original with suffix somehow in YAMLs
            # people.csv found a mapping but player hasn't appeared in IPL YAMLs
            # (e.g. overseas player with no IPL history) — return stripped version
            return cs_stripped

        # ── 2. Direct match ────────────────────────────────────────────────────
        if clean in known_cricsheet_names:
            return clean

        # ── 3 & 4. Last-name fallback ──────────────────────────────────────────
        last       = clean.split()[-1].lower()
        candidates = last_name_map.get(last, [])

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # Multiple candidates — narrow down by first initial
            first_initial = clean[0].upper()
            refined = [c for c in candidates if c[0].upper() == first_initial]
            if len(refined) == 1:
                return refined[0]

        # ── 5. No match — return as-is ─────────────────────────────────────────
        return clean

    return resolve


def build_known_names(raw_path):
    """
    Scan all YAML files and collect every player name that appears in
    playing XI lists or registry sections. These are the canonical
    Cricsheet names that appear in ball-by-ball data.
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
        for team_players in info.get("players", {}).values():
            known.update(team_players)
        for name in info.get("registry", {}).get("people", {}).keys():
            known.add(name)
    return known


# ── Delivery iterator (handles both old and new Cricsheet YAML formats) ────────

def iter_innings(data):
    """
    Yields (inning_index, batting_team, balls_list) for each innings.
    Handles both formats:
      Old (pre-2017): innings → '1st innings' → deliveries → {0.1: {...}}
      New (post-2017): innings → {team, overs} → deliveries → [{batter, ...}]
    """
    for idx, inning in enumerate(data.get("innings", [])):
        if "team" in inning:
            # New format
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
                    is_leg  = "wides" not in extras and "noballs" not in extras
                    is_wkt  = len(wickets) > 0
                    wk_kind = wickets[0].get("kind", "") if is_wkt else ""
                    balls.append(dict(over=over_num, batter=batter, bowler=bowler,
                                      runs_bat=runs_b, runs_tot=runs_t,
                                      is_legal=is_leg, is_wkt=is_wkt,
                                      wkt_kind=wk_kind))
            yield idx, batting_team, balls
        else:
            # Old format
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
                        is_leg   = "wides" not in extras and "noballs" not in extras
                        is_wkt   = wicket is not None
                        wk_kind  = wicket.get("kind", "") if is_wkt else ""
                        balls.append(dict(over=over_num, batter=batter, bowler=bowler,
                                          runs_bat=runs_b, runs_tot=runs_t,
                                          is_legal=is_leg, is_wkt=is_wkt,
                                          wkt_kind=wk_kind))
                yield idx, batting_team, balls


# ── Per-player stat stores ─────────────────────────────────────────────────────

batter_innings = defaultdict(list)   # player → [{date, runs, balls}]
bowler_spells  = defaultdict(list)   # player → [{date, runs_given, wickets, legal_balls}]
matchup_log    = defaultdict(lambda: defaultdict(list))  # batter → bowler → [{date, runs, balls}]


# ── Per-player score calculators ───────────────────────────────────────────────

def get_player_bat(player, before_date):
    """
    Individual batting quality = rolling avg × strike_rate / 100
    Requires at least 3 innings of history. Returns None if insufficient data.

    Example: avg=35, SR=145 → score = 35 × 145 / 100 = 50.75
    A score of 50+ = elite T20 batter
    """
    hist = [h for h in batter_innings[player]
            if h["date"] < before_date][-BATTING_WINDOW:]
    if len(hist) < 3:
        return None
    runs  = sum(h["runs"]  for h in hist)
    balls = sum(h["balls"] for h in hist)
    n     = len(hist)
    avg   = runs / n
    sr    = (runs / balls * 100) if balls > 0 else 0
    return avg * sr / 100


def get_player_bowl(player, before_date):
    """
    Individual bowling quality = (wicket_rate × 100) / economy
    Requires at least 3 spells and 12 legal balls. Returns None if insufficient.

    Higher = better bowler. Bumrah (WR~0.05, econ~6.5) → ~0.77
    Rewards wicket-taking ability AND economy together.
    Economy alone isn't enough — a bowler with 6 econ but 0 wickets scores 0.
    """
    hist = [h for h in bowler_spells[player]
            if h["date"] < before_date][-BOWLING_WINDOW:]
    if len(hist) < 3:
        return None
    wkts  = sum(h["wickets"]     for h in hist)
    balls = sum(h["legal_balls"] for h in hist)
    runs  = sum(h["runs_given"]  for h in hist)
    if balls < 12:
        return None
    wr   = wkts / balls
    econ = (runs / balls * 6) if balls > 0 else 8.0
    return (wr * 100) / max(econ, 1.0)


def get_bat_score(players, before_date):
    """
    Team batting score = mean individual score × 11
    Uses mean so squad size doesn't inflate the score.
    Players with < 3 innings of history don't contribute
    (avoids pulling the mean down with zeros).
    """
    scores = [s for p in players
              if (s := get_player_bat(p, before_date)) is not None]
    if not scores:
        return 0.0
    return round(np.mean(scores) * 11, 3)


def get_bowl_score(players, before_date):
    """
    Team bowling score = mean of top-5 individual bowl scores × 5
    Only top 5 because you bowl your best 5 bowlers the most overs.
    This correctly weights Bumrah over a mediocre 6th bowler.
    """
    scores = [s for p in players
              if (s := get_player_bowl(p, before_date)) is not None]
    if not scores:
        return 0.0
    top5 = sorted(scores, reverse=True)[:5]
    return round(np.mean(top5) * 5, 3)


def get_matchup_score(batters, bowlers, before_date):
    """
    How well batters historically perform against these specific bowlers.
    Returns average strike rate across all batter-bowler combinations.
    Minimum 4 balls between a pair to be included (avoids single-ball noise).
    100 = neutral (run a ball). >100 = batters dominate. <100 = bowlers dominate.
    """
    strike_rates = []
    for batter in batters:
        for bowler in bowlers:
            hist = [h for h in matchup_log[batter][bowler]
                    if h["date"] < before_date]
            if len(hist) < 4:
                continue
            runs  = sum(h["runs"]  for h in hist)
            balls = sum(h["balls"] for h in hist)
            strike_rates.append((runs / balls * 100) if balls > 0 else 100)
    return round(np.mean(strike_rates), 3) if strike_rates else 100.0


# ── Main processing function ───────────────────────────────────────────────────

def process_all():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/squads",    exist_ok=True)

    # Step 1: build known Cricsheet names from all YAMLs
    print("Building player name registry from YAMLs...")
    known = build_known_names(RAW_PATH)
    print(f"  {len(known)} unique Cricsheet player names found")

    # Step 2: build name resolver using people.csv
    resolve = load_name_map(PEOPLE_CSV, known)

    # Step 3: load squad JSON and resolve all names
    with open(SQUADS_PATH) as f:
        squad_json = json.load(f)

    squad_cs = {}   # team → [resolved cricsheet names]
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

    print(f"  Squad matching: {matched} matched, {unmatched} not in Cricsheet data")
    print(f"  (Unmatched = genuinely new players with no IPL history)")

    # Step 4: process all YAML files chronologically
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
        xi      = info.get("players", {})   # playing XIs for this match

        if len(teams) != 2 or not winner or not date:
            continue

        # Always alphabetical so team1/team2 is consistent with rest of pipeline
        team1, team2 = sorted(teams)
        xi1 = xi.get(team1, [])
        xi2 = xi.get(team2, [])

        # Compute PRE-MATCH scores (all stats from before this match date)
        records.append({
            "date":             date,
            "team1":            team1,
            "team2":            team2,
            "winner":           winner,
            "t1_bat_score":     get_bat_score(xi1,  date),
            "t1_bowl_score":    get_bowl_score(xi1, date),
            "t1_matchup_score": get_matchup_score(xi1, xi2, date),
            "t2_bat_score":     get_bat_score(xi2,  date),
            "t2_bowl_score":    get_bowl_score(xi2, date),
            "t2_matchup_score": get_matchup_score(xi2, xi1, date),
        })

        # ════ UPDATE STAT STORES AFTER RECORDING — no leakage ════════════════
        for _, batting_team, balls in iter_innings(data):

            # Per-batter accumulators for this innings
            b_balls = defaultdict(int)
            b_runs  = defaultdict(int)
            # Per-bowler accumulators for this spell
            bo_balls = defaultdict(int)
            bo_runs  = defaultdict(int)
            bo_wkts  = defaultdict(int)

            for b in balls:
                bat = b["batter"]
                bow = b["bowler"]

                if b["is_legal"]:
                    # Batter faced a legal ball
                    b_balls[bat] += 1
                    bo_balls[bow] += 1

                    # Runs credited to batter (not run outs from their end)
                    if not b["is_wkt"] or b["wkt_kind"] == "run out":
                        b_runs[bat] += b["runs_bat"]

                    # Matchup: record every legal ball between this batter and bowler
                    if bat and bow:
                        matchup_log[bat][bow].append({
                            "date":  date,
                            "runs":  b["runs_bat"],
                            "balls": 1,
                        })

                # Total runs off this delivery (including extras) charged to bowler
                bo_runs[bow] += b["runs_tot"]

                # Wicket credited to bowler (not run outs or obstructions)
                if b["is_wkt"] and b["wkt_kind"] not in (
                        "run out", "obstructing the field"):
                    bo_wkts[bow] += 1

            # Store batter innings summary
            for batter, balls_faced in b_balls.items():
                batter_innings[batter].append({
                    "date":  date,
                    "runs":  b_runs.get(batter, 0),
                    "balls": balls_faced,
                })

            # Store bowler spell summary
            for bowler, legal_balls in bo_balls.items():
                bowler_spells[bowler].append({
                    "date":        date,
                    "runs_given":  bo_runs.get(bowler, 0),
                    "wickets":     bo_wkts.get(bowler, 0),
                    "legal_balls": legal_balls,
                })

    # Step 5: save per-match scores
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPer-match scores saved: {OUTPUT_PATH}  ({len(df)} matches)")

    # Step 6: compute CURRENT squad scores for 2026 predictions
    print("\nCurrent squad strength scores (using full IPL history up to 2026):")
    today = pd.Timestamp("2026-01-01")
    rows  = []
    for team, cs_names in squad_cs.items():
        bat  = get_bat_score(cs_names,  today)
        bowl = get_bowl_score(cs_names, today)
        rows.append({"team": team, "bat_score": bat, "bowl_score": bowl})

    cur = pd.DataFrame(rows)

    # Soft normalise to 0.4–1.0 so no team is 2x another
    def soft_norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return series * 0 + 0.7
        return 0.4 + ((series - mn) / (mx - mn)) * 0.6

    cur["bat_score_norm"]  = soft_norm(cur["bat_score"]).round(4)
    cur["bowl_score_norm"] = soft_norm(cur["bowl_score"]).round(4)
    cur = cur.sort_values("bat_score_norm", ascending=False)
    cur.to_csv(CURRENT_OUTPUT, index=False)

    # Print summary table
    print(f"\n  {'Team':<35} {'Bat(raw)':>9} {'Bowl(raw)':>10} "
          f"{'Bat(norm)':>10} {'Bowl(norm)':>11}")
    print(f"  {'-'*79}")
    for _, r in cur.iterrows():
        bar = "█" * int(r["bat_score_norm"] * 30)
        print(f"  {r['team']:<35} {r['bat_score']:>9.1f} {r['bowl_score']:>10.3f} "
              f"{r['bat_score_norm']:>10.3f} {r['bowl_score_norm']:>11.3f}  {bar}")

    print("\nDone. Run feature_engineering.py next.")


if __name__ == "__main__":
    process_all()