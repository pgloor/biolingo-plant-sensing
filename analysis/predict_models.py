"""
Prediction models: CO2 + ECG + VOC + RMSSD → Valence / Arousal
───────────────────────────────────────────────────────────────
Usage:
  python predict_models.py
  python predict_models.py --file co2_emotion_log.csv --lag 60
"""

import argparse, warnings, copy, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from scipy import stats
warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file",     default="co2_emotion_log.csv")
parser.add_argument("--lag",      type=int, default=60)
parser.add_argument("--max-hour", type=int, default=18,
                    help="Exclude rows after this hour (default 18 = 6PM). "
                         "Use 24 to include all data.")
parser.add_argument("--hour-control", action="store_true",
                    help="Add circular hour-of-day (sin/cos) as a control feature.")
parser.add_argument("--exclude-dates", type=str, default=None,
                    help="Comma-separated dates to exclude e.g. '2026-04-10,2026-04-14' "
                         "(use for sessions with known camera/detection issues)")
args = parser.parse_args()

# ── Load & clean ─────────────────────────────────────────────────
df = pd.read_csv(args.file, on_bad_lines="skip")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
for c in df.columns:
    if c != "timestamp":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Artifact filters
df = df[df["ecg_mv"].isna() | (df["ecg_mv"] < 3290)]
if "ecg2_mv" in df.columns:
    df = df[df["ecg2_mv"].isna() | (df["ecg2_mv"] < 3290)]
if "rmssd" in df.columns:
    df = df[df["rmssd"].isna() | df["rmssd"].between(1, 200)]

# ── HRV-derived emotion (Russell circumplex, Kreibig 2010) ───────
# Compute session-normalised valence_hrv and arousal_hrv from Polar H10 data.
# valence_hrv  = zscore(rmssd)  — high HRV = calm/positive valence
# arousal_hrv  = zscore(hr_bpm) — high HR  = high arousal
# Using rolling 10-min window z-score then clipped to [-1,+1]
if "rmssd" in df.columns and "hr_bpm" in df.columns:
    WINDOW_HRV = 600  # 10 minutes
    df["valence_hrv"] = (df["rmssd"]
        .rolling(WINDOW_HRV, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1] - x.mean()) / (x.std() + 1e-6), -3, 3) / 3,
               raw=False))
    df["arousal_hrv"] = (df["hr_bpm"]
        .rolling(WINDOW_HRV, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1] - x.mean()) / (x.std() + 1e-6), -3, 3) / 3,
               raw=False))
    print(f"HRV-derived emotion computed: valence_hrv n={df['valence_hrv'].notna().sum():,}  "
          f"arousal_hrv n={df['arousal_hrv'].notna().sum():,}")

# ── ECG2 temporal alignment ───────────────────────────────────────
if "ecg2_mv" in df.columns:
    df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()

# Face-present rows (non-zero valence)
df_face = df[df["valence"].notna() & (df["valence"] != 0.0)].copy()

# ── FER quality filter ────────────────────────────────────────────
# Remove likely false detections: extreme valence jumps between consecutive
# rows (>0.6 change in 1s = implausible for genuine emotion, likely belly/object)
# and rows where |valence| > 0.95 (HSEmotion rarely produces these for real faces)
n_before = len(df_face)
df_face["valence_jump"] = df_face["valence"].diff().abs()
df_face = df_face[
    (df_face["valence"].abs() <= 0.95) &          # clip extreme values
    (df_face["valence_jump"].isna() |              # keep first row
     (df_face["valence_jump"] < 0.6))             # remove sudden jumps
].drop(columns=["valence_jump"])
print(f"FER quality filter: removed {n_before - len(df_face):,} likely false detections "
      f"({100*(n_before-len(df_face))/n_before:.1f}%)")

# ── Hour-of-day filter ────────────────────────────────────────────
df_face["hour"] = df_face["timestamp"].dt.hour + df_face["timestamp"].dt.minute / 60
if args.max_hour < 24:
    n_before = len(df_face)
    df_face = df_face[df_face["hour"] <= args.max_hour].reset_index(drop=True)
    print(f"Hour filter: kept rows with hour ≤ {args.max_hour}:00  "
          f"({n_before - len(df_face):,} evening rows excluded)")

if args.exclude_dates:
    dates = [d.strip() for d in args.exclude_dates.split(",")]
    n_before = len(df_face)
    mask = df_face["timestamp"].dt.strftime("%Y-%m-%d").isin(dates)
    df_face = df_face[~mask].reset_index(drop=True)
    print(f"Date exclusion: removed {n_before - len(df_face):,} rows from {dates}")

# ── Circular hour encoding (sin/cos) ─────────────────────────────
# Maps hour 0-24 onto a circle so 23:59 → 0:00 is continuous
df_face["hour_sin"] = np.sin(2 * np.pi * df_face["hour"] / 24)
df_face["hour_cos"] = np.cos(2 * np.pi * df_face["hour"] / 24)
print(f"Total rows: {len(df)}  |  Face-present after hour filter: {len(df_face)}")

# ── Session detection ─────────────────────────────────────────────
# Sessions = continuous blocks of face-present data (emotion != 'none')
# separated by gaps > 20 minutes. This uses camera presence as the
# ground truth for "Peter is at his desk" rather than clock time.
def assign_sessions(df, gap_minutes=60):
    df = df.copy()
    # Only face-present rows define session boundaries
    dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df["session"] = (dt > gap_minutes * 60).cumsum()
    return df

df_face = assign_sessions(df_face)
sessions = df_face["session"].unique()
print(f"Sessions detected: {len(sessions)}  (face-present, 60-min gap = calendar day)")
print(f"  Within-day absences <60 min (lunch, errands) treated as breaks within session.")
for s in sessions:
    sdf = df_face[df_face["session"] == s]
    print(f"  Session {s}: {len(sdf):,} rows  "
          f"{sdf['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} → "
          f"{sdf['timestamp'].iloc[-1].strftime('%H:%M')}  "
          f"({(sdf['timestamp'].iloc[-1]-sdf['timestamp'].iloc[0]).total_seconds()/60:.0f} min)")

# ── Within-session detrending ─────────────────────────────────────
# Remove session mean from every signal column to eliminate sensor
# baseline drift across sessions while preserving within-session dynamics
DETREND_COLS = ["co2", "ecg_mv", "ecg2_mv", "valence", "arousal",
                "valence_hrv", "arousal_hrv", "rmssd", "hr_bpm", "voc_index"]
# Note: temperature, humidity, voc_raw excluded — they proxy room conditions / time-of-day
# not individual physiology, so detrending them would be misleading

def detrend_sessions(df, cols):
    df = df.copy()
    for col in cols:
        if col not in df.columns: continue
        # Subtract session mean (z-score optional — use mean-only to keep original units)
        df[col + "_d"] = df.groupby("session")[col].transform(
            lambda x: x - x.mean())
    return df

df_face = detrend_sessions(df_face, DETREND_COLS)
detrended_cols = [c + "_d" for c in DETREND_COLS if c in df_face.columns]
print(f"\nDetrended columns added: {detrended_cols}")

# ── Mediation-derived lagged features ────────────────────────────
# From lagged mediation analysis:
#   plant(t) → CO2(t+34s) → valence(t+41s)
#   CO2(t) → valence(t+7s)
#   ecg_mv_d → valence_d best r at lag=25s
#   hr_bpm → arousal best r at lag=25-30s
# Add these specific lags as features so models can use the known causal delays

LAG_FEATURES = [
    ("ecg_mv",   34, "ecg_lag34"),
    ("ecg_mv_d", 25, "ecg_d_lag25"),
    ("ecg2_mv",  34, "ecg2_lag34"),    # Stefan sensor 34s lag
    ("ecg2_mv_d",25, "ecg2_d_lag25"),  # Stefan sensor detrended 25s lag
    ("co2",       7, "co2_lag7"),
    ("co2_d",     7, "co2_d_lag7"),
    ("hr_bpm",   30, "hr_lag30"),
    ("hr_bpm_d", 30, "hr_d_lag30"),
]

for src, lag, name in LAG_FEATURES:
    if src not in df_face.columns: continue
    # Shift within session only — don't bleed across session boundaries
    df_face[name] = df_face.groupby("session")[src].transform(
        lambda x: x.shift(lag))

lag_cols = [name for _, _, name in LAG_FEATURES if name in df_face.columns]
print(f"Lagged features added: {lag_cols}")

# ── Session-normalised cross-validation ──────────────────────────
def session_cv_r2(X_df, y_series, model, sessions_col):
    """Leave-one-session-out CV with within-session z-score normalisation."""
    session_ids = sessions_col.values
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
        return np.nan, np.nan

    all_preds, all_true = [], []
    for test_sess in unique_sessions:
        train_mask = session_ids != test_sess
        test_mask  = session_ids == test_sess
        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue

        X_train = X_df[train_mask].copy()
        X_test  = X_df[test_mask].copy()
        y_train = y_series[train_mask].copy()
        y_test  = y_series[test_mask].copy()

        # Z-score each session's features and target independently
        for col in X_train.columns:
            mu, sd = X_train[col].mean(), X_train[col].std()
            if sd > 0:
                X_train[col] = (X_train[col] - mu) / sd
                X_test[col]  = (X_test[col]  - mu) / sd  # use train stats

        y_mu, y_sd = y_train.mean(), y_train.std()
        if y_sd > 0:
            y_train_n = (y_train - y_mu) / y_sd
        else:
            continue

        m = copy.deepcopy(model)
        m.fit(X_train.values, y_train_n.values)
        pred_n = m.predict(X_test.values)
        # De-normalise back to original scale
        pred = pred_n * y_sd + y_mu
        all_preds.extend(pred)
        all_true.extend(y_test.values)

    if len(all_true) < 10:
        return np.nan, np.nan
    r2  = r2_score(all_true, all_preds)
    r,p = stats.pearsonr(all_preds, all_true)
    return r2, r

# ── Helpers ──────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

def sig(p):
    return "***" if p<.001 else "** " if p<.01 else "*  " if p<.05 else "   "

def within_session_cv_r2(X_df, y_series, model, sessions_col):
    """TimeSeriesSplit CV run separately within each session, results averaged.
    This is the honest test for detrended data — does the within-day
    fluctuation pattern in the first half predict the second half?"""
    session_ids = sessions_col.values
    unique_sessions = np.unique(session_ids)
    all_r2s = []
    for sess in unique_sessions:
        mask = session_ids == sess
        X_s = X_df[mask].values
        y_s = y_series[mask].values
        if len(y_s) < 40:
            continue
        n_splits = min(5, len(y_s) // 40)
        if n_splits < 2:
            continue
        cv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(copy.deepcopy(model), X_s, y_s,
                                 cv=cv, scoring="r2")
        all_r2s.extend(scores.tolist())
    if not all_r2s:
        return np.nan
    return float(np.mean(all_r2s))

def evaluate(name, X_df, y, model, sess_col):
    # Within-session CV (honest for detrended data)
    ws_r2 = within_session_cv_r2(X_df, y, copy.deepcopy(model), sess_col)
    # Session-level leave-one-out CV (cross-day generalisation)
    sess_r2, sess_r = session_cv_r2(X_df, y, copy.deepcopy(model), sess_col)
    # In-sample fit
    model.fit(X_df.values, y.values)
    pred = model.predict(X_df.values)
    r2  = r2_score(y.values, pred)
    r_p_res = stats.pearsonr(pred, y.values)
    r_p = float(np.atleast_1d(r_p_res.statistic if hasattr(r_p_res,'statistic') else r_p_res[0])[0])
    p_val = float(np.atleast_1d(r_p_res.pvalue if hasattr(r_p_res,'pvalue') else r_p_res[1])[0])
    ws_str   = f"  WithinCV={ws_r2:+.3f}"   if not np.isnan(ws_r2)   else ""
    sess_str = f"  SessCV={sess_r2:+.3f}"   if not np.isnan(sess_r2) else ""
    sess_r_str = f"(r={sess_r:+.3f})" if not np.isnan(sess_r2) else ""
    print(f"    {name:30s}  R²={r2:+.3f}{ws_str}{sess_str}{sess_r_str}  r={r_p:+.3f}{sig(p_val)}")
    return model, pred, sess_r2, sess_r

MODELS = {
    "RF":  RandomForestRegressor(n_estimators=200, max_depth=6,
                                  random_state=42, n_jobs=-1),
    "XGB": XGBRegressor(n_estimators=200, max_depth=4,
                        learning_rate=0.05, random_state=42,
                        n_jobs=-1, verbosity=0,
                        tree_method="hist"),  # hist = fast histogram-based splitting
}

# ── Build feature sets dynamically based on available columns ────
ALWAYS   = ["co2", "ecg_mv"]
OPTIONAL = ["ecg2_mv", "rmssd", "hr_bpm", "voc_index"]
available = [c for c in OPTIONAL if c in df_face.columns and df_face[c].notna().sum() > 100]

# Define feature bundles — physiological signals only
# Naming: Plant1 = ecg_mv (SCD41 board, RC lowpass, 5s avg)
#          Plant2 = ecg2_mv (Stefan/Biolingo board, 0.1-20Hz bandpass, 500ms avg)
feat_bundles = [
    ("CO2 only",                        ["co2"]),
    ("Plant1 only",                     ["ecg_mv"]),
    ("CO2 + Plant1",                    ["co2", "ecg_mv"]),
]
if "ecg2_mv" in available:
    feat_bundles.append(("Plant2 only",                 ["ecg2_mv"]))
    feat_bundles.append(("Plant1 + Plant2",             ["ecg_mv", "ecg2_mv"]))
    feat_bundles.append(("CO2 + Plant1 + Plant2",       ["co2", "ecg_mv", "ecg2_mv"]))
if "hr_bpm" in available:
    feat_bundles.append(("Plant1 + HR",                 ["ecg_mv", "hr_bpm"]))
    feat_bundles.append(("CO2 + Plant1 + HR",           ["co2", "ecg_mv", "hr_bpm"]))
if "rmssd" in available:
    feat_bundles.append(("CO2 + Plant1 + RMSSD",        ["co2", "ecg_mv", "rmssd"]))
if "rmssd" in available and "hr_bpm" in available:
    feat_bundles.append(("Plant1 + HR + RMSSD",         ["ecg_mv", "hr_bpm", "rmssd"]))
    feat_bundles.append(("CO2 + Plant1 + HR + RMSSD",   ["co2", "ecg_mv", "hr_bpm", "rmssd"]))
if "voc_index" in available:
    feat_bundles.append(("CO2 + Plant1 + VOC",          ["co2", "ecg_mv", "voc_index"]))
if "voc_index" in available and "hr_bpm" in available:
    feat_bundles.append(("Plant1 + HR + VOC",           ["ecg_mv", "hr_bpm", "voc_index"]))
if "voc_index" in available and "rmssd" in available:
    feat_bundles.append(("All: CO2+Plant1+VOC+RMSSD",
                         ["co2", "ecg_mv", "rmssd", "voc_index"]))
if "voc_index" in available and "rmssd" in available and "hr_bpm" in available:
    feat_bundles.append(("All+HR: CO2+Plant1+VOC+RMSSD+HR",
                         ["co2", "ecg_mv", "hr_bpm", "rmssd", "voc_index"]))
if "rmssd" in available:
    feat_bundles.append(("RMSSD only",  ["rmssd"]))
if "hr_bpm" in available:
    feat_bundles.append(("HR only",     ["hr_bpm"]))

# ── Lagged bundles — encode mediation-derived delays ─────────────
if "ecg_d_lag25" in df_face.columns:
    feat_bundles.append(("Plant1+HR + lags",
                         ["ecg_mv", "hr_bpm",
                          "ecg_d_lag25", "co2_d_lag7"]))
if "ecg_d_lag25" in df_face.columns and "hr_d_lag30" in df_face.columns:
    feat_bundles.append(("Plant1+HR+RMSSD + lags",
                         ["ecg_mv", "hr_bpm", "rmssd",
                          "ecg_d_lag25", "co2_d_lag7", "hr_d_lag30"]))
if "voc_index" in available and "ecg_d_lag25" in df_face.columns:
    feat_bundles.append(("All detrended + lags",
                         ["co2_d", "ecg_mv_d", "hr_bpm_d",
                          "ecg_d_lag25", "co2_d_lag7", "hr_d_lag30"]))

# ── Optionally add circular hour-of-day control to every bundle ───
if args.hour_control:
    print("\nHour-of-day control (sin/cos) added to all feature bundles.")
    feat_bundles = [(name, cols + ["hour_sin", "hour_cos"])
                    for name, cols in feat_bundles]

# ── Detrended versions — skip pure single-signal baselines ───────
SKIP_DETREND = {"CO2 only", "Plant ECG only", "RMSSD only", "HR only"}
feat_bundles_d = []
for name, cols in feat_bundles:
    base_name = name.replace(" [detrended]", "")
    if base_name in SKIP_DETREND: continue
    dcols = [c + "_d" for c in cols if c + "_d" in df_face.columns]
    if len(dcols) == len(cols):
        feat_bundles_d.append((name + " [detrended]", dcols))

# Detrended targets
TARGETS_D = {"valence": "valence_d", "arousal": "arousal_d",
             "valence_hrv": "valence_hrv_d", "arousal_hrv": "arousal_hrv_d"}

results_best = {}   # best by WithinCV (for prediction plots)
results_rich = {}   # richest multi-feature bundle (for importance plots)
all_results  = []   # every (target, bundle, model, r2, within_cv, sess_cv) for summary table

def run_target(target, label, color):
    print(f"\n{'═'*75}")
    print(f"PREDICT {label.upper()}")
    print("─"*75)
    best_cvr2, best_bundle, best_pred, best_mod, best_cols = -999, None, None, None, None
    target_d = TARGETS_D.get(target, target + "_d")

    # Run raw bundles first, then detrended
    all_bundles = feat_bundles + feat_bundles_d
    raw_done = False
    for fname, fcols in all_bundles:
        is_detrended = fname.endswith("[detrended]")
        if is_detrended and not raw_done:
            print(f"\n  {'─'*40} DETRENDED (session mean removed) {'─'*5}")
            raw_done = True
        tgt = target_d if is_detrended else target
        needed = fcols + [tgt, "session"]
        sub = df_face[needed].dropna()
        if len(sub) < 50:
            print(f"  [{fname}]  skipped (n={len(sub)})")
            continue
        X_df = sub[fcols]
        y    = sub[tgt]
        sess = sub["session"]
        print(f"\n  [{fname}]  n={len(sub):,}")
        for mname, m in MODELS.items():
            mod, pred, sr2, sr = evaluate(mname, X_df, y, copy.deepcopy(m), sess)
            if mname in ("RF", "XGB"):
                ws_r2 = within_session_cv_r2(X_df, y, copy.deepcopy(m), sess)
                all_results.append({
                    "target": target, "bundle": fname, "model": mname,
                    "n": len(sub), "within_cv": ws_r2,
                    "sess_cv": sr2, "sess_r": float(sr) if not np.isnan(sr2) else np.nan
                })
            if mname == "RF":
                ws_r2 = within_session_cv_r2(X_df, y, copy.deepcopy(m), sess)
                score = ws_r2 if not np.isnan(ws_r2) else (sr2 if not np.isnan(sr2) else -999)
                if score > best_cvr2:
                    best_cvr2, best_bundle = score, fname
                    best_pred, best_mod, best_cols = pred, mod, fcols
                    best_y = y.values

    if best_pred is not None:
        results_best[target] = (best_y, best_pred, best_mod, best_cols, best_bundle, color)
        print(f"\n  ✓ Best feature set: [{best_bundle}]  WithinCV-R²={best_cvr2:+.3f}")

    # Also store the richest multi-feature bundle for importance plot
    # (prefer detrended "All" bundle if available)
    for fname, fcols in reversed(all_bundles):
        if len(fcols) < 2: continue
        needed = fcols + [target, "session"]
        sub = df_face[needed].dropna()
        if len(sub) < 50: continue
        m = copy.deepcopy(MODELS["RF"])
        tgt = TARGETS_D.get(target, target) if fname.endswith("[detrended]") else target
        sub2 = df_face[fcols + [tgt, "session"]].dropna()
        if len(sub2) < 50: continue
        m.fit(sub2[fcols].values, sub2[tgt].values)
        results_rich[target] = (m, fcols, fname)
        break

run_target("valence",     "Valence",     "#ce93d8")
run_target("arousal",     "Arousal",     "#80cbc4")
# HRV-derived emotion targets — run only if available
if "valence_hrv" in df_face.columns and df_face["valence_hrv"].notna().sum() > 100:
    run_target("valence_hrv", "Valence HRV", "#2980b9")
if "arousal_hrv" in df_face.columns and df_face["arousal_hrv"].notna().sum() > 100:
    run_target("arousal_hrv", "Arousal HRV", "#8e44ad")

# ── Lagged CO2 → emotion ─────────────────────────────────────────
print(f"\n{'═'*75}")
print(f"LAGGED CORRELATIONS  (lags 0..{args.lag} s)")
print("─"*75)
lags = list(range(0, args.lag + 1, 5))
lag_results = {}

def lagged_corr(df, xcol, ycol, lags):
    rs, ps = [], []
    for lag in lags:
        sub = df[[xcol, ycol]].dropna()
        x = sub[xcol].shift(lag).dropna()
        y = sub.loc[x.index, ycol]
        if len(x) < 20:
            rs.append(np.nan); ps.append(1.0); continue
        r, p = stats.pearsonr(x.values, y.values)
        rs.append(float(np.atleast_1d(r)[0]))
        ps.append(float(np.atleast_1d(p)[0]))
    return rs, ps

print("  Raw signals:")
for sig_col, tgt in [("co2","valence"),("co2","arousal"),
                      ("ecg_mv","valence"),("ecg_mv","arousal"),
                      ("rmssd","valence"),("rmssd","arousal"),
                      ("hr_bpm","valence"),("hr_bpm","arousal")]:
    if sig_col not in df_face.columns: continue
    rs, ps = lagged_corr(df_face, sig_col, tgt, lags)
    best_i = int(np.nanargmax(np.abs(rs)))
    print(f"    {sig_col:8s} → {tgt:8s}  best r={rs[best_i]:+.3f}{sig(ps[best_i])} at lag={lags[best_i]}s")
    lag_results[(sig_col, tgt)] = (rs, ps)

print("  Detrended signals:")
for sig_col, tgt in [("co2_d","valence_d"),("co2_d","arousal_d"),
                      ("ecg_mv_d","valence_d"),("ecg_mv_d","arousal_d"),
                      ("rmssd_d","valence_d"),("rmssd_d","arousal_d"),
                      ("hr_bpm_d","valence_d"),("hr_bpm_d","arousal_d")]:
    if sig_col not in df_face.columns: continue
    rs, ps = lagged_corr(df_face, sig_col, tgt, lags)
    best_i = int(np.nanargmax(np.abs(rs)))
    label = sig_col.replace("_d","")
    tgt_l = tgt.replace("_d","")
    print(f"    {label:8s}Δ → {tgt_l:8s}Δ  best r={rs[best_i]:+.3f}{sig(ps[best_i])} at lag={lags[best_i]}s")
    lag_results[(sig_col, tgt)] = (rs, ps)

if "voc_index" in available:
    print("  VOC index:")
    for tgt, tgt_d in [("valence","valence_d"),("arousal","arousal_d")]:
        rs, ps = lagged_corr(df_face, "voc_index", tgt, lags)
        best_i = int(np.nanargmax(np.abs(rs)))
        print(f"    voc_idx  → {tgt:8s}       best r={rs[best_i]:+.3f}{sig(ps[best_i])} at lag={lags[best_i]}s")
        lag_results[("voc_index", tgt)] = (rs, ps)
        if "voc_index_d" in df_face.columns:
            rs, ps = lagged_corr(df_face, "voc_index_d", tgt_d, lags)
            best_i = int(np.nanargmax(np.abs(rs)))
            print(f"    voc_idxΔ → {tgt:8s}Δ      best r={rs[best_i]:+.3f}{sig(ps[best_i])} at lag={lags[best_i]}s")
            lag_results[("voc_index_d", tgt_d)] = (rs, ps)

# ── Feature importance ───────────────────────────────────────────
print(f"\n{'═'*75}")
print("RF FEATURE IMPORTANCE")
print("─"*75)
for tgt in ["valence", "arousal"]:
    if tgt not in results_best: continue
    _, _, rf_mod, fcols, bundle, _ = results_best[tgt]
    imp = rf_mod.feature_importances_
    print(f"\n  {tgt.upper()} (from [{bundle}]):")
    for f, i in sorted(zip(fcols, imp), key=lambda x: -x[1]):
        bar = "█" * int(i * 50)
        print(f"    {f:20s}  {i:.3f}  {bar}")

# ── Rolling-window prediction (10-minute averages) ───────────────
print(f"\n{'═'*75}")
print("ROLLING WINDOW PREDICTION  (10-min averages — smooths moment-to-moment noise)")
print("─"*75)
print("Rationale: individual-second predictions are too noisy (r~0.13).")
print("Rolling 10-min averages ask: 'Is this person more positive/aroused")
print("than usual THIS WINDOW?' — a question the sensors can actually answer.\n")

WINDOW = 600  # 600 seconds = 10 minutes

def make_rolling(df, cols, window, session_col):
    """Rolling mean within session only."""
    df = df.copy()
    for col in cols:
        if col not in df.columns: continue
        df[col + "_roll"] = df.groupby(session_col)[col].transform(
            lambda x: x.rolling(window, min_periods=window//4, center=False).mean())
    return df

ROLL_SIGNALS = ["co2_d", "ecg_mv_d", "hr_bpm_d", "rmssd_d"]
if "ecg2_mv" in available: ROLL_SIGNALS.append("ecg2_mv_d")
if "voc_index" in available: ROLL_SIGNALS.append("voc_index_d")
df_roll = make_rolling(df_face, ROLL_SIGNALS, WINDOW, "session")
roll_cols = [c + "_roll" for c in ROLL_SIGNALS if c + "_roll" in df_roll.columns]

for target, tgt_d in [("valence", "valence_d"), ("arousal", "arousal_d")]:
    # Rolling target too
    df_roll[tgt_d + "_roll"] = df_roll.groupby("session")[tgt_d].transform(
        lambda x: x.rolling(WINDOW, min_periods=WINDOW//4).mean())
    needed = roll_cols + [tgt_d + "_roll", "session"]
    sub = df_roll[needed].dropna()
    if len(sub) < 100:
        print(f"  {target}: insufficient data after rolling (n={len(sub)})")
        continue
    X = sub[roll_cols]
    y = sub[tgt_d + "_roll"]
    sess = sub["session"]
    m = copy.deepcopy(MODELS["RF"])
    sr2, sr = session_cv_r2(X, y, m, sess)
    ws_r2 = within_session_cv_r2(X, y, copy.deepcopy(MODELS["RF"]), sess)
    m.fit(X.values, y.values)
    pred = m.predict(X.values)
    r_full = float(np.atleast_1d(stats.pearsonr(pred, y.values)[0])[0])
    print(f"  {target:8s}  n={len(sub):,}  "
          f"WithinCV={ws_r2:+.3f}  SessCV={sr2:+.3f}  SessR={sr:+.3f}  "
          f"in-sample r={r_full:+.3f}")

# ── Change-detection prediction ───────────────────────────────────
print(f"\n{'═'*75}")
print("CHANGE DETECTION  (predict direction of valence/arousal shift)")
print("─"*75)
print("Binary classification: did valence/arousal go UP or DOWN in the next 30s?\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

HORIZON = 30  # predict change over next 30 seconds

for target in ["valence_d", "arousal_d"]:
    tgt_label = target.replace("_d", "")
    # Create binary target: 1 = went up, 0 = went down over next HORIZON rows
    df_face[target + "_change"] = df_face.groupby("session")[target].transform(
        lambda x: (x.shift(-HORIZON) - x).apply(lambda v: 1 if v > 0 else 0))

    feat_cols = [c for c in ["ecg_mv_d", "co2_d", "hr_bpm_d", "rmssd_d",
                              "ecg_d_lag25", "co2_d_lag7"]
                 if c in df_face.columns]
    needed = feat_cols + [target + "_change", "session"]
    sub = df_face[needed].dropna()
    if len(sub) < 100: continue

    X = sub[feat_cols].values
    y = sub[target + "_change"].values
    sess = sub["session"].values

    # Leave-one-session-out accuracy
    unique_sess = np.unique(sess)
    accs = []
    for test_s in unique_sess:
        train_m = sess != test_s
        test_m  = sess == test_s
        if train_m.sum() < 50 or test_m.sum() < 10: continue
        clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                     random_state=42, n_jobs=-1)
        clf.fit(X[train_m], y[train_m])
        pred = clf.predict(X[test_m])
        accs.append(accuracy_score(y[test_m], pred))

    mean_acc = np.mean(accs) if accs else np.nan
    chance = 0.5
    print(f"  {tgt_label:8s}  n={len(sub):,}  "
          f"SessCV accuracy={mean_acc:.3f}  "
          f"(chance=0.500, lift={mean_acc-chance:+.3f})")

# ── Summary table ─────────────────────────────────────────────────
print(f"\n{'═'*75}")
print("CROSS-SESSION GENERALISATION SUMMARY")
print("Models where SessCV-R² > 0  (genuinely generalise across days)")
print("─"*75)
res_df = pd.DataFrame(all_results)
positive = res_df[res_df["sess_cv"] > 0].sort_values("sess_cv", ascending=False)
if len(positive):
    print(f"{'Target':10} {'Model':8} {'Bundle':45} {'n':>7} {'WithinCV':>10} {'SessCV':>8} {'SessR':>7}")
    print("─"*100)
    for _, row in positive.iterrows():
        print(f"  {row.target:8} {row.model:8} {row.bundle:45} "
              f"{int(row.n):>7,} {row.within_cv:>10.3f} {row.sess_cv:>8.3f} {row.get('sess_r', float('nan')):>7.3f} ✓")
else:
    print("  No models achieved positive SessCV in this run.")

print()
print("All RF results sorted by |SessR| (top 10) — r is more meaningful than R² for weak signals:")
print(f"{'Target':10} {'Bundle':45} {'n':>7} {'SessCV':>8} {'SessR':>7}")
print("─"*85)
rf_res = res_df[res_df["model"]=="RF"].copy()
rf_res["abs_sess_r"] = rf_res["sess_r"].abs()
top10 = rf_res.dropna(subset=["sess_r"]).sort_values("abs_sess_r", ascending=False).head(10)
for _, row in top10.iterrows():
    flag = " ✓" if row.sess_cv > 0 else ""
    print(f"  {row.target:8} {row.bundle:45} "
          f"{int(row.n):>7,} {row.sess_cv:>8.3f} {row.sess_r:>7.3f}{flag}")

# ── Plots ─────────────────────────────────────────────────────────
n_targets = len(results_best)
fig = plt.figure(figsize=(16, 14), facecolor="white")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.38)

COLORS = {"valence": "#ce93d8", "arousal": "#80cbc4",
          "co2": "#5be4a0", "ecg_mv": "#ff8a65", "voc_index": "#f7c948"}

# Row 0: actual vs predicted for valence + arousal
for col, (tgt, label) in enumerate([("valence","Valence"), ("arousal","Arousal")]):
    if tgt not in results_best: continue
    y_true, y_pred, _, _, bundle, color = results_best[tgt]
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("#f5f5f5")
    ax.plot(y_true,  color="#999999", lw=0.7, label="Actual", alpha=0.8)
    ax.plot(y_pred,  color=color,  lw=1.2, label=f"RF [{bundle}]", alpha=0.9)
    r, p = stats.pearsonr(y_pred, y_true)
    ax.set_title(f"{label} — RF  r={r:+.3f}{sig(p)}", color="#1a4a8a", fontsize=14)
    ax.legend(fontsize=11, labelcolor="black", facecolor="white")
    ax.tick_params(colors="#666")
    for sp in ax.spines.values(): sp.set_edgecolor("black")

# Row 1: lag plots
ax_lag = fig.add_subplot(gs[1, :])
ax_lag.set_facecolor("#f5f5f5")
ax_lag.axhline(0, color="#444", lw=0.8)
lag_colors = {"co2":     {"valence":"#9c7bce","arousal":"#5cb8b0"},
              "ecg_mv":  {"valence":"#d46060","arousal":"#d4a060"},
              "voc_index":{"valence":"#c8c844","arousal":"#88c844"}}
for (sig_col, tgt), (rs, ps) in lag_results.items():
    c = lag_colors.get(sig_col, {}).get(tgt, "#aaa")
    ax_lag.plot(lags, rs, color=c, lw=1.8, marker="o", ms=3,
                label=f"{sig_col}→{tgt}")
ax_lag.set_xlabel("Lag (seconds)", color="black", fontsize=13)
ax_lag.set_ylabel("Pearson r", color="black", fontsize=13)
ax_lag.set_title("Lagged correlations: signals → emotion", color="#1a4a8a", fontsize=14)
ax_lag.legend(fontsize=8, labelcolor="black", facecolor="white", framealpha=0.9,
              ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0., columnspacing=1.0, handlelength=1.5)
ax_lag.tick_params(colors="#666")
for sp in ax_lag.spines.values(): sp.set_edgecolor("black")

# Row 2: feature importance — always from richest multi-feature bundle
for col, tgt in enumerate(["valence","arousal"]):
    rich = results_rich.get(tgt) or (None, None, None)
    rf_mod, fcols, bundle = rich if rich[0] is not None else results_best.get(tgt, (None,)*6)[2:5] + (None,)
    if rf_mod is None or fcols is None: continue
    imp = rf_mod.feature_importances_
    ax = fig.add_subplot(gs[2, col])
    ax.set_facecolor("#f5f5f5")
    sorted_pairs = sorted(zip(fcols, imp), key=lambda x: x[1])
    fs, is_ = zip(*sorted_pairs)
    clrs = [COLORS.get(f.replace("_d",""), "#aaa") for f in fs]
    ax.barh(range(len(fs)), is_, color=clrs)
    ax.set_yticks(range(len(fs)))
    ax.set_yticklabels(fs, color="black", fontsize=11)
    bname = bundle or "unknown"
    title = f"RF Importance → {tgt.capitalize()}"
    if bname and bname != "unknown":
        title += "\n" + "\n".join(textwrap.wrap(bname, width=36))
    ax.set_title(title, color="#1a4a8a", fontsize=10)
    ax.tick_params(colors="#666")
    for sp in ax.spines.values(): sp.set_edgecolor("black")

fig.suptitle("Predicting Emotion from CO₂ + Plant + VOC + HRV",
             color="black", fontsize=16, y=0.99)
out = args.file.replace(".csv", "_models.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out}")
plt.show()
