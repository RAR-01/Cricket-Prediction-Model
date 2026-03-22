"""
train_model.py
--------------
Trains the IPL prediction model and provides a predict_match() function
that uses current squad data for upcoming matches.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATA_PATH    = "data/processed/final_dataset.csv"
CURRENT_PATH = "data/processed/ipl_2026_team_scores.csv"
SQUADS_PATH  = "data/squads/squads.json"
MODEL_DIR    = "models"

NON_FEATURE  = {"date","team1","team2","winner","target"}

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

FEATURES = [c for c in df.columns if c not in NON_FEATURE]
X = df[FEATURES]
y = df["target"]

print(f"Total matches : {len(df)}")
print(f"Features      : {len(FEATURES)} → {FEATURES}")
print(f"Class balance : {y.mean():.1%}\n")

# ── Walk-forward cross-validation ─────────────────────────────────────────────
df["year"] = df["date"].dt.year
test_seasons = sorted(df["year"].unique())[-4:]
print(f"Walk-forward CV — test seasons: {test_seasons}\n")

wf_scores = {"LR": [], "RF": [], "GB": []}

for test_year in test_seasons:
    tr = df["year"] < test_year
    te = df["year"] == test_year
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    if len(X_tr) < 50 or len(X_te) < 5:
        continue

    sc = StandardScaler()
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    lr.fit(sc.fit_transform(X_tr), y_tr)

    rf = RandomForestClassifier(n_estimators=300, max_depth=5,
                                min_samples_leaf=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, subsample=0.8,
                                    min_samples_leaf=12, random_state=42)
    gb.fit(X_tr, y_tr)

    baseline = max(y_te.mean(), 1 - y_te.mean())
    lr_acc = accuracy_score(y_te, lr.predict(sc.transform(X_te)))
    rf_acc = accuracy_score(y_te, rf.predict(X_te))
    gb_acc = accuracy_score(y_te, gb.predict(X_te))

    wf_scores["LR"].append(lr_acc)
    wf_scores["RF"].append(rf_acc)
    wf_scores["GB"].append(gb_acc)

    print(f"Season {test_year} (n={len(X_te):3d}) baseline={baseline:.1%} "
          f"LR={lr_acc:.1%} RF={rf_acc:.1%} GB={gb_acc:.1%}")

print()
for name, scores in wf_scores.items():
    if scores:
        print(f"  {name}: avg={np.mean(scores):.1%}  "
              f"min={min(scores):.1%}  max={max(scores):.1%}")

# ── Final model ────────────────────────────────────────────────────────────────
print("\n--- Final model (80/20 time split) ---")
split      = int(len(df) * 0.80)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_tr)
X_te_sc    = scaler.transform(X_te)

gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                learning_rate=0.05, subsample=0.8,
                                min_samples_leaf=12, random_state=42)
gb.fit(X_tr, y_tr)
acc      = accuracy_score(y_te, gb.predict(X_te))
baseline = accuracy_score(y_te, [int(y_tr.mode()[0])] * len(y_te))
print(f"Baseline : {baseline:.1%}   Model : {acc:.1%}   "
      f"Lift : +{(acc-baseline)*100:.1f}pp")

print("\nFeature importances (Gradient Boosting):")
for feat, imp in sorted(zip(FEATURES, gb.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 60)
    print(f"  {feat:<28} {imp:.4f}  {bar}")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(gb,      f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler,  f"{MODEL_DIR}/scaler.pkl")
joblib.dump(FEATURES,f"{MODEL_DIR}/features.pkl")
print(f"\nModel saved → {MODEL_DIR}/")


# ── Predict upcoming match using current squad scores ──────────────────────────
def predict_match(team_a, team_b):
    """
    Predict winner between team_a and team_b for an upcoming match.
    Uses current squad strength scores from ipl_2026_team_scores.csv.
    Pass teams in any order.
    """
    model  = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    feats  = joblib.load(f"{MODEL_DIR}/features.pkl")

    t1, t2 = sorted([team_a, team_b])

    # Get current squad scores
    if os.path.exists(CURRENT_PATH):
        cur = pd.read_csv(CURRENT_PATH).set_index("team")
        t1_bat  = cur.loc[t1, "bat_score"]  if t1 in cur.index else 0
        t2_bat  = cur.loc[t2, "bat_score"]  if t2 in cur.index else 0
        t1_bowl = cur.loc[t1, "bowl_score"] if t1 in cur.index else 0
        t2_bowl = cur.loc[t2, "bowl_score"] if t2 in cur.index else 0
    else:
        t1_bat = t2_bat = t1_bowl = t2_bowl = 0

    # Use the most recent historical row for ELO, form, h2h, venue
    pair = df[(df["team1"] == t1) & (df["team2"] == t2)]
    if pair.empty:
        pair = df[(df["team1"] == t2) | (df["team2"] == t1)]
    if pair.empty:
        print(f"No history for {t1} vs {t2}")
        return

    base = pair.iloc[-1][feats].copy()

    # Overwrite player features with CURRENT squad scores
    if "t1_bat_score"    in feats: base["t1_bat_score"]    = t1_bat
    if "t2_bat_score"    in feats: base["t2_bat_score"]    = t2_bat
    if "t1_bowl_score"   in feats: base["t1_bowl_score"]   = t1_bowl
    if "t2_bowl_score"   in feats: base["t2_bowl_score"]   = t2_bowl
    if "bat_score_diff"  in feats: base["bat_score_diff"]  = t1_bat  - t2_bat
    if "bowl_score_diff" in feats: base["bowl_score_diff"] = t1_bowl - t2_bowl
    if "matchup_diff"    in feats: base["matchup_diff"]    = 0   # neutral without live XI

    X_pred = pd.DataFrame([base[feats]])
    prob   = model.predict_proba(X_pred)[0]
    winner = t1 if prob[1] >= 0.5 else t2

    print(f"\n{'='*50}")
    print(f"  {t1}  vs  {t2}")
    print(f"  Predicted winner : {winner}")
    print(f"  {t1:<35} {prob[1]:.1%}")
    print(f"  {t2:<35} {prob[0]:.1%}")
    print(f"{'='*50}\n")
    return winner, prob


if __name__ == "__main__":
    predict_match("Chennai Super Kings", "Mumbai Indians")
    predict_match("Royal Challengers Bengaluru", "Kolkata Knight Riders")
    predict_match("Sunrisers Hyderabad", "Rajasthan Royals")