"""
Correlation analysis for co2_emotion_log.csv
─────────────────────────────────────────────
Computes Pearson + Spearman correlations between:
  co2, temperature, humidity, ecg_mv, valence, arousal

Usage:
  python analyze_correlations.py
  python analyze_correlations.py --file my_other_log.csv
  python analyze_correlations.py --lag 60   # also test time-lagged correlations
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
SIGNALS = ["co2", "temperature", "humidity", "ecg_mv", "ecg2_mv",
           "valence", "arousal", "valence_hrv", "arousal_hrv",
           "rmssd", "hr_bpm", "voc_index"]
PRETTY  = {"co2": "CO₂ (ppm)", "temperature": "Temp (°C)", "humidity": "Humidity (%)",
           "ecg_mv": "Plant1 (mV)", "ecg2_mv": "Plant2 (mV)",
           "valence": "Valence", "arousal": "Arousal",
           "valence_hrv": "Valence HRV", "arousal_hrv": "Arousal HRV",
           "rmssd": "RMSSD (ms)", "hr_bpm": "HR (bpm)", "voc_index": "VOC index"}
# Note: voc_raw excluded — it's raw SGP40 resistance, heavily confounded by
# temperature (r=-0.77) and humidity. Use voc_index (compensated) instead.

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file", default="co2_emotion_log.csv")
parser.add_argument("--lag",  type=int, default=0,
                    help="Max lag in rows for lagged correlation (0 = skip)")
parser.add_argument("--min-face", type=float, default=0.0,
                    help="Only include rows where abs(valence)>threshold (face present)")
args = parser.parse_args()

# ── Load ─────────────────────────────────────────────────────────
df = pd.read_csv(args.file, on_bad_lines='skip')
print(f"Columns found: {df.columns.tolist()}")

# Rename columns if they look like data (no header)
if df.columns[0] not in ['timestamp', 'ts']:
    df = pd.read_csv(args.file, header=None, on_bad_lines='skip')
    if df.shape[1] == 9:
        df.columns = ["ts","co2","temperature","humidity","ecg_mv","leads_on","emotion","valence","arousal"]
    elif df.shape[1] == 7:
        df.columns = ["ts","co2","temperature","humidity","emotion","valence","arousal"]
    elif df.shape[1] == 8:
        df.columns = ["ts","co2","temperature","humidity","ecg_mv","emotion","valence","arousal"]

ts_col = next((c for c in df.columns if "time" in c.lower() or c == "ts"), None)
if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)
    print(f"Date range: {df[ts_col].min()}  →  {df[ts_col].max()}")

# Drop rows with all-zero emotion (no face) if requested
if args.min_face > 0:
    df = df[df["valence"].abs() > args.min_face]
    print(f"Rows after face filter: {len(df)}")

# Drop unplugged sensor rows (AD8232 at rail = 3300 mV means no sensor connected)
if "ecg_mv" in df.columns:
    df["ecg_mv"] = pd.to_numeric(df["ecg_mv"], errors="coerce")
    n_before = len(df)
    df = df[df["ecg_mv"].isna() | (df["ecg_mv"] < 3290)]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with ECG≥3290 mV (unplugged sensor)")

# Drop physiologically implausible RMSSD values (artifacts from BLE drops)
if "rmssd" in df.columns:
    df["rmssd"] = pd.to_numeric(df["rmssd"], errors="coerce")
    n_before = len(df)
    df = df[df["rmssd"].isna() | (df["rmssd"].between(1, 200))]
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with RMSSD outside 1–200 ms (BLE artifact)")

# ── HRV-derived emotion ───────────────────────────────────────────
if "rmssd" in df.columns and "hr_bpm" in df.columns:
    df["rmssd"]   = pd.to_numeric(df["rmssd"],   errors="coerce")
    df["hr_bpm"]  = pd.to_numeric(df["hr_bpm"],  errors="coerce")
    WINDOW_HRV = 600
    df["valence_hrv"] = (df["rmssd"]
        .rolling(WINDOW_HRV, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))
    df["arousal_hrv"] = (df["hr_bpm"]
        .rolling(WINDOW_HRV, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))

# ── FER quality filter ────────────────────────────────────────────
if "valence" in df.columns:
    n_before = len(df)
    jump = df["valence"].diff().abs()
    df = df[(df["valence"].abs() <= 0.95) & (jump.isna() | (jump < 0.6))]
    print(f"FER quality filter: removed {n_before-len(df):,} likely false detections")

# ── ECG2 temporal alignment ───────────────────────────────────────
# ECG1 is logged as a 5-second mean (SCD41 interval); ECG2 updates every 500ms.
# Raw correlation between them is near-zero by construction (different time windows).
# Fix: replace ecg2_mv with its 5-second rolling mean to match ECG1's effective window.
if "ecg2_mv" in df.columns:
    df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()
    print("ECG2 resampled: 5-second rolling mean applied to match ECG1 update rate")

# Keep only numeric signal columns that exist — do NOT dropna globally
cols = [c for c in SIGNALS if c in df.columns]
data = df[cols].apply(pd.to_numeric, errors="coerce")

# Print per-column n so user can see coverage
print(f"\nLoaded: {args.file}")
print("Per-column valid counts:")
for c in cols:
    print(f"  {PRETTY.get(c,c):15} n={data[c].notna().sum():,}")
print()

# ── Pearson & Spearman — pairwise complete observations ──────────
# Each pair uses only rows where BOTH columns are non-null
pearson_r  = pd.DataFrame(index=cols, columns=cols, dtype=float)
pearson_p  = pd.DataFrame(index=cols, columns=cols, dtype=float)
pearson_n  = pd.DataFrame(index=cols, columns=cols, dtype=int)
spearman_r = pd.DataFrame(index=cols, columns=cols, dtype=float)
spearman_p = pd.DataFrame(index=cols, columns=cols, dtype=float)

for a in cols:
    for b in cols:
        pair = data[[a, b]].dropna()
        n_pair = len(pair)
        pearson_n.loc[a,b] = n_pair
        if a == b:
            pearson_r.loc[a,b] = 1.0; pearson_p.loc[a,b] = 0.0
            spearman_r.loc[a,b] = 1.0; spearman_p.loc[a,b] = 0.0
            continue
        if n_pair < 4:
            pearson_r.loc[a,b] = np.nan;  pearson_p.loc[a,b] = np.nan
            spearman_r.loc[a,b] = np.nan; spearman_p.loc[a,b] = np.nan
            continue
        x = pair[a].to_numpy(dtype=float)
        y = pair[b].to_numpy(dtype=float)
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        pearson_r.loc[a,b]  = float(np.atleast_1d(pr)[0])
        pearson_p.loc[a,b]  = float(np.atleast_1d(pp)[0])
        spearman_r.loc[a,b] = float(np.atleast_1d(sr)[0])
        spearman_p.loc[a,b] = float(np.atleast_1d(sp)[0])

# ── Print table ───────────────────────────────────────────────────
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "

print("═"*70)
print("PEARSON correlations  (* p<.05  ** p<.01  *** p<.001)")
print("─"*70)
header = f"{'':12}" + "".join(f"{PRETTY.get(c,c):>12}" for c in cols)
print(header)
for a in cols:
    row = f"{PRETTY.get(a,a):12}"
    for b in cols:
        r = pearson_r.loc[a,b]
        p = pearson_p.loc[a,b]
        cell = f"{r:+.3f}{sig(p)}"
        row += f"{cell:>12}"
    print(row)

print("\n" + "═"*70)
print("SPEARMAN correlations")
print("─"*70)
print(header)
for a in cols:
    row = f"{PRETTY.get(a,a):12}"
    for b in cols:
        r = spearman_r.loc[a,b]
        p = spearman_p.loc[a,b]
        cell = f"{r:+.3f}{sig(p)}"
        row += f"{cell:>12}"
    print(row)

# ── Highlight strongest off-diagonal pairs ────────────────────────
print("\n" + "═"*70)
print("TOP 5 STRONGEST PEARSON PAIRS (off-diagonal)")
print("─"*70)
pairs = []
for i, a in enumerate(cols):
    for b in cols[i+1:]:
        pairs.append((abs(pearson_r.loc[a,b]), a, b,
                      pearson_r.loc[a,b], pearson_p.loc[a,b]))
pairs.sort(reverse=True)
for _, a, b, r, p in pairs[:5]:
    n_pair = pearson_n.loc[a,b]
    print(f"  {PRETTY.get(a,a):18} ↔  {PRETTY.get(b,b):18}  r={r:+.3f}  p={p:.4f} {sig(p)}  n={n_pair:,}")

# ── Optional: lagged correlations ─────────────────────────────────
if args.lag > 0:
    print(f"\n{'═'*70}")
    print(f"LAGGED PEARSON  co2 → [valence, arousal, ecg_mv, ecg2_mv]  (lags 0..{args.lag} rows)")
    print("─"*70)
    targets = [c for c in ["valence", "arousal", "ecg_mv", "ecg2_mv"] if c in cols]
    for target in targets:
        pair = data[["co2", target]].dropna()
        best_r, best_lag, best_p = 0, 0, 1.0
        for lag in range(args.lag + 1):
            a = pair["co2"].iloc[:len(pair)-lag].values
            b = pair[target].iloc[lag:].values
            r, p = stats.pearsonr(a, b)
            if abs(r) > abs(best_r):
                best_r, best_lag, best_p = r, lag, p
        print(f"  co2 → {PRETTY.get(target,target):18}  best r={best_r:+.3f} "
              f"at lag={best_lag} rows  p={best_p:.4f} {sig(best_p)}")

# ── Plot heatmaps ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("white")

def heatmap(ax, matrix, title):
    ax.set_facecolor("#f5f5f5")
    labels = [PRETTY.get(c, c) for c in cols]
    m = matrix.values.astype(float)
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=35, ha="right",
                                                         color="black", fontsize=10)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels, color="black", fontsize=10)
    ax.set_title(title, color="#1a4a8a", fontsize=13, pad=10)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = m[i, j]
            # annotation removed — colour encodes value

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

heatmap(axes[0], pearson_r,  "Pearson r")
heatmap(axes[1], spearman_r, "Spearman ρ")

max_n = int(pearson_n.values.max())
min_n = int(np.nanmin([pearson_n.loc[a,b] for a in cols for b in cols if a!=b]))
plt.suptitle(f"CO₂ + Emotion Correlations  (pairwise n: {min_n:,}–{max_n:,})", color="black",
             fontsize=15, y=1.01)
plt.tight_layout()
out = args.file.replace(".csv", "_correlations.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nHeatmap saved → {out}")
plt.show()
