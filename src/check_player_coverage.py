"""
check_player_coverage.py
------------------------
Run this to see exactly which squad players have strong IPL records,
thin records, or zero records in your Cricsheet data.

Usage: python src/check_player_coverage.py
"""

import json, yaml, os
from collections import defaultdict
import pandas as pd

SQUADS_PATH  = "data/squads/squads.json"
PEOPLE_CSV   = "data/squads/people.csv"
RAW_PATH     = "data/raw"

# ── Load squads ────────────────────────────────────────────────────────────────
squads = json.load(open(SQUADS_PATH))

# ── Load people.csv ────────────────────────────────────────────────────────────
people = pd.read_csv(PEOPLE_CSV)
cols   = [c.strip().lower() for c in people.columns]
people.columns = cols
name_col   = next(c for c in cols if c in ('name', 'full_name'))
unique_col = next(c for c in cols if 'unique' in c)

full_to_cs = {}
for _, row in people.iterrows():
    full = str(row[name_col]).strip()
    cs   = str(row[unique_col]).strip()
    if full and cs and full != 'nan':
        full_to_cs[full.lower()] = cs

# ── Build known Cricsheet names ────────────────────────────────────────────────
print("Scanning YAMLs for known player names...")
known = set()
innings_count = defaultdict(int)   # how many IPL matches each player appeared in

files = [f for f in os.listdir(RAW_PATH) if f.endswith('.yaml')]
for fname in files:
    fpath = os.path.join(RAW_PATH, fname)
    with open(fpath, encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
        except:
            continue
    info = data.get('info', {})
    for tp in info.get('players', {}).values():
        for p in tp:
            known.add(p)
            innings_count[p] += 1

print(f"  {len(known)} unique players found in {len(files)} YAML files\n")

# ── Name resolver ──────────────────────────────────────────────────────────────
def resolve(full_name):
    cs = full_to_cs.get(full_name.lower())
    if cs and cs in known:
        return cs
    if cs:
        return cs
    if full_name in known:
        return full_name
    last   = full_name.split()[-1].lower()
    cands  = [n for n in known if n.split()[-1].lower() == last]
    if len(cands) == 1:
        return cands[0]
    elif len(cands) > 1:
        fi      = full_name[0].upper()
        refined = [c for c in cands if c[0].upper() == fi]
        if len(refined) == 1:
            return refined[0]
    return full_name

# ── Check coverage per team ────────────────────────────────────────────────────
STRONG   = 20   # 20+ IPL apps = strong record
MODERATE = 5    # 5-19 = moderate
# < 5 = thin, 0 = none

all_results = []
for team, players in sorted(squads.items()):
    for p in players:
        cs   = resolve(p)
        apps = innings_count.get(cs, 0)
        cat  = "STRONG" if apps >= STRONG else ("MODERATE" if apps >= MODERATE else ("THIN" if apps > 0 else "NONE"))
        all_results.append({"team": team, "full_name": p, "cs_name": cs, "ipl_apps": apps, "category": cat})

df = pd.DataFrame(all_results)

# ── Print summary per team ─────────────────────────────────────────────────────
print(f"{'Team':<35} {'Strong(20+)':>11} {'Moderate(5-19)':>14} {'Thin(1-4)':>9} {'None':>6} {'Total':>6}")
print("-" * 85)

for team in sorted(squads.keys()):
    t = df[df['team'] == team]
    s  = len(t[t['category'] == 'STRONG'])
    m  = len(t[t['category'] == 'MODERATE'])
    th = len(t[t['category'] == 'THIN'])
    n  = len(t[t['category'] == 'NONE'])
    print(f"  {team:<33} {s:>11} {m:>14} {th:>9} {n:>6} {len(t):>6}")

print("-" * 85)
totals = df['category'].value_counts()
print(f"  {'TOTAL':<33} {totals.get('STRONG',0):>11} {totals.get('MODERATE',0):>14} "
      f"{totals.get('THIN',0):>9} {totals.get('NONE',0):>6} {len(df):>6}")

# ── Print NONE and THIN players ────────────────────────────────────────────────
print("\n\nPlayers with NO IPL record (model uses squad average for these):")
print(f"  {'Team':<35} {'Player':<30} {'Cricsheet Name':<30}")
print("  " + "-" * 95)
for _, r in df[df['category'] == 'NONE'].sort_values('team').iterrows():
    cs_flag = " ← name not resolved" if r['cs_name'] == r['full_name'] else ""
    print(f"  {r['team']:<35} {r['full_name']:<30} {r['cs_name']:<30}{cs_flag}")

print(f"\n\nPlayers with THIN record (1-4 IPL apps):")
print(f"  {'Team':<35} {'Player':<30} {'Apps':>5}")
print("  " + "-" * 75)
for _, r in df[df['category'] == 'THIN'].sort_values(['team','ipl_apps']).iterrows():
    print(f"  {r['team']:<35} {r['full_name']:<30} {r['ipl_apps']:>5}")

print(f"\n\nTop 5 most experienced players per team:")
for team in sorted(squads.keys()):
    t    = df[df['team'] == team].sort_values('ipl_apps', ascending=False).head(5)
    top5 = ', '.join(f"{r['full_name']} ({r['ipl_apps']})" for _, r in t.iterrows())
    print(f"  {team:<35} {top5}")