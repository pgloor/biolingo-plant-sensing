"""
Lagged Mediation Analysis: Emotion(t) → CO2(t+Δa) → Plant voltage(t+Δa+Δb)
─────────────────────────────────────────────────────────────────────────────
Tests whether CO2 mediates the emotion → plant voltage relationship.

Grid search over:
  Δa = 0..28 s  (emotion → CO2 lag)
  Δb = 0..28 s  (CO2 → plant voltage lag)
  total Δa+Δb ≤ 28 s  (within the known human→plant response window)

For each (Δa, Δb) combination:
  - Path a:  emotion → CO2           (Pearson r)
  - Path b:  CO2 → plant voltage     (Pearson r)
  - Path c:  emotion → plant voltage (total effect, no mediator)
  - Path c': emotion → plant voltage (direct, controlling CO2)
  - Indirect a×b: mediated effect

The (Δa, Δb) that maximises |indirect effect| tells us how the
5–28 s human→plant delay splits between the two paths.

Usage:
  python mediation_lagged.py
  python mediation_lagged.py --emotion arousal --max-lag 30
"""

import argparse, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file",    default="co2_emotion_log.csv")
parser.add_argument("--emotion", default="valence",
                    choices=["valence","arousal","valence_hrv","arousal_hrv",
                             "rmssd","voc_raw","voc_index","hr_bpm"],
                    help="X variable (emotion or physiological signal)")
parser.add_argument("--mediator", default="co2",
                    choices=["co2","voc_index","rmssd","hr_bpm","ecg2_mv"],
                    help="Mediator M (default: co2)")
parser.add_argument("--outcome",  default="ecg_mv",
                    choices=["ecg_mv","ecg2_mv","valence","arousal",
                             "valence_hrv","arousal_hrv","co2","voc_index"],
                    help="Outcome Y (default: ecg_mv = plant voltage sensor 1)")
parser.add_argument("--max-lag", type=int, default=28,
                    help="Max lag in rows (≈seconds at 1Hz FER)")
parser.add_argument("--window-closed-min", type=float, default=None,
                    help="Only use rows where CO2 >= this value (window closed)")
parser.add_argument("--window-closed-max", type=float, default=None,
                    help="Only use rows where CO2 <= this value")
parser.add_argument("--reverse", action="store_true",
                    help="Test reverse direction: outcome(t) → mediator(t+Δa) → emotion(t+Δa+Δb)")
parser.add_argument("--max-hour", type=int, default=18,
                    help="Exclude rows after this hour (default 18 = 6PM). Use 24 for all data.")
parser.add_argument("--exclude-transition", type=float, default=30,
                    help="Drop rows within this many seconds of a CO2 regime change")
args = parser.parse_args()

# ── Load ─────────────────────────────────────────────────────────
df = pd.read_csv(args.file)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.apply(lambda c: pd.to_numeric(c, errors="coerce")
              if c.name != "timestamp" else c)

# ── HRV-derived emotion — compute BEFORE dropna so available as X/M/Y ──
if "rmssd" in df.columns and "hr_bpm" in df.columns:
    WINDOW_HRV = 600
    df["valence_hrv"] = (df["rmssd"]
        .rolling(WINDOW_HRV, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))
    df["arousal_hrv"] = (df["hr_bpm"]
        .rolling(WINDOW_HRV, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))

MED  = args.mediator
OUT  = args.outcome
EMO  = args.emotion
required = list({MED, OUT, EMO} - {"timestamp"})
df = df.dropna(subset=required)
print(f"Rows with all required columns ({', '.join(required)}): {len(df):,}")

# Drop unplugged sensor rows (AD8232 rail value = 3300 mV)
n_before = len(df)
df = df[df["ecg_mv"] < 3290].reset_index(drop=True)
if len(df) < n_before:
    print(f"Dropped {n_before - len(df)} rows with ECG≥3290 mV (unplugged sensor)")
if "ecg2_mv" in df.columns:
    n_before = len(df)
    df = df[df["ecg2_mv"].isna() | (df["ecg2_mv"] < 3290)].reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with ECG2≥3290 mV")

# Drop physiologically implausible RMSSD (BLE artifacts)
if "rmssd" in df.columns:
    df["rmssd"] = pd.to_numeric(df["rmssd"], errors="coerce")
    n_before = len(df)
    df = df[df["rmssd"].isna() | df["rmssd"].between(1, 200)].reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with RMSSD outside 1–200 ms (BLE artifact)")

# Numeric conversion for VOC columns if present
for col in ["voc_raw", "voc_index"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── ECG2 temporal alignment ───────────────────────────────────────
if "ecg2_mv" in df.columns:
    df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()

# ── FER quality filter ────────────────────────────────────────────
if "valence" in df.columns:
    n_before = len(df)
    jump = df["valence"].diff().abs()
    df = df[(df["valence"].abs() <= 0.95) & (jump.isna() | (jump < 0.6))].reset_index(drop=True)
    print(f"FER quality filter: removed {n_before-len(df):,} likely false detections")

# Hour-of-day filter — exclude evening data (different plant metabolism, room conditions)
if args.max_hour < 24:
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    n_before = len(df)
    df = df[hour <= args.max_hour].reset_index(drop=True)
    print(f"Hour filter: kept hour ≤ {args.max_hour}:00  "
          f"({n_before - len(df):,} evening rows excluded)")

# Face-present only (skip for non-FER emotion variables)
NO_FACE_VARS = {"rmssd", "voc_raw", "voc_index"}
if args.emotion not in NO_FACE_VARS:
    df = df[df["valence"] != 0.0].reset_index(drop=True)

# Require valid values for the chosen emotion variable
df = df.dropna(subset=[args.emotion]).reset_index(drop=True)

# ── Window-closed filter ──────────────────────────────────────────
if args.window_closed_min is not None or args.window_closed_max is not None:
    lo = args.window_closed_min or 0
    hi = args.window_closed_max or 9999
    before = len(df)

    # Mark transition rows: CO2 crossed the threshold within exclude_transition rows
    co2_in  = df["co2"] >= lo
    co2_out = df["co2"] < lo
    regime_change = (co2_in != co2_in.shift(1)).fillna(False)
    # expand exclusion window around each change
    excl_rows = int(args.exclude_transition)
    transition_mask = regime_change.rolling(
        window=excl_rows*2+1, center=True, min_periods=1).max().astype(bool)

    df = df[(df["co2"] >= lo) & (df["co2"] <= hi) & ~transition_mask].reset_index(drop=True)
    print(f"Window filter: CO2 {lo:.0f}–{hi:.0f} ppm  "
          f"(excluded {excl_rows}s transition buffer)")
    print(f"Rows after filter: {before} → {len(df)}\n")

n  = len(df)
print(f"Rows (face-present): {n}")
if "co2" in df.columns:
    print(f"CO2:   {df['co2'].mean():.0f} ± {df['co2'].std():.0f} ppm")
print(f"{OUT}:  {df[OUT].mean():.0f} ± {df[OUT].std():.0f}")
print(f"X={EMO}  M={MED}  Y={OUT}\n")

MAX   = args.max_lag
REVERSE = args.reverse

if REVERSE:
    print(f"Direction: {OUT}(t) → {MED}(t+Δa) → {EMO}(t+Δa+Δb)  [REVERSE]\n")
else:
    print(f"Direction: {EMO}(t) → {MED}(t+Δa) → {OUT}(t+Δa+Δb)  [FORWARD]\n")

# ── Path computation ──────────────────────────────────────────────
def pearson(a, b):
    """Return (r, p) handling edge cases."""
    if len(a) < 10: return (0.0, 1.0)
    try:    return stats.pearsonr(a, b)
    except: return (0.0, 1.0)

def partial_r(x, y, z):
    """Pearson r between x and y after regressing out z."""
    def resid(a, b):
        b2 = b.reshape(-1,1)
        return a - LinearRegression().fit(b2, a).predict(b2)
    return pearson(resid(x, z), resid(y, z))

# Grid: all (da, db) where da>=0, db>=0, da+db <= MAX
results = []
lags_a  = range(0, MAX+1)
lags_b  = range(0, MAX+1)

for da, db in itertools.product(lags_a, lags_b):
    if da + db > MAX: continue

    total = da + db
    if total == 0:
        if REVERSE:
            X = df[OUT].values
            M = df[MED].values
            Y = df[EMO].values
        else:
            X = df[EMO].values
            M = df[MED].values
            Y = df[OUT].values
    else:
        end = n - total if total > 0 else n
        if REVERSE:
            X = df[OUT].iloc[:end].values
            M = df[MED].iloc[da: da+end].values
            Y = df[EMO].iloc[total: total+end].values
        else:
            X = df[EMO].iloc[:end].values
            M = df[MED].iloc[da: da+end].values
            Y = df[OUT].iloc[total: total+end].values

    if len(X) < 20: continue

    # Standardise
    X_s = (X - X.mean()) / (X.std() + 1e-9)
    M_s = (M - M.mean()) / (M.std() + 1e-9)
    Y_s = (Y - Y.mean()) / (Y.std() + 1e-9)

    # Paths
    ra, pa   = pearson(X_s, M_s)          # a: emotion → CO2
    rb, pb   = pearson(M_s, Y_s)          # b: CO2 → plant
    rc, pc   = pearson(X_s, Y_s)          # c: total emotion → plant
    rc2, pc2 = partial_r(Y_s, X_s, M_s)  # c': direct (CO2 controlled)

    indirect = ra * rb                     # a×b
    # proportion mediated (only meaningful when c and indirect same sign)
    prop_med = (indirect / rc) if abs(rc) > 0.01 else np.nan

    results.append({
        "da": da, "db": db, "total_lag": da+db,
        "r_a": ra, "p_a": pa,
        "r_b": rb, "p_b": pb,
        "r_c": rc, "p_c": pc,
        "r_c_prime": rc2, "p_c_prime": pc2,
        "indirect": indirect,
        "prop_mediated": prop_med,
        "n": len(X)
    })

res = pd.DataFrame(results)

# ── Best lag combination ──────────────────────────────────────────
best = res.loc[res["indirect"].abs().idxmax()]
X_label = OUT if REVERSE else EMO
Y_label = EMO if REVERSE else OUT

print("═"*65)
print(f"BEST LAG COMBINATION  (maximises |indirect effect a×b|)")
print("─"*65)
print(f"  Δa ({X_label:10s} → {MED}):        {int(best.da):2d} s")
print(f"  Δb ({MED} → {Y_label:12s}):  {int(best.db):2d} s")
print(f"  Total lag:                   {int(best.total_lag):2d} s")
print(f"  n at this lag:               {int(best.n)}")
print()
print(f"  Path a  ({X_label} → {MED}):       r={best.r_a:+.3f}  p={best.p_a:.4f}")
print(f"  Path b  ({MED} → {Y_label}):   r={best.r_b:+.3f}  p={best.p_b:.4f}")
print(f"  Path c  (total effect):      r={best.r_c:+.3f}  p={best.p_c:.4f}")
print(f"  Path c' (direct, {MED} ctrl):  r={best.r_c_prime:+.3f}  p={best.p_c_prime:.4f}")
print(f"  Indirect a×b:                {best.indirect:+.4f}")
if not np.isnan(best.prop_mediated):
    print(f"  Proportion mediated:         {best.prop_mediated*100:.1f}%")
print()

interp_x = X_label; interp_y = Y_label
if best.p_a < 0.05 and best.p_b < 0.05:
    if best.p_c_prime > 0.05:
        print(f"  → FULL mediation: {MED} fully explains {interp_x}→{interp_y} link")
    elif best.p_c_prime < 0.05 and abs(best.r_c_prime) < abs(best.r_c):
        print(f"  → PARTIAL mediation: {MED} partially explains {interp_x}→{interp_y} link")
    else:
        print(f"  → {MED} is significant on both paths but does not reduce direct effect")
else:
    print("  → Mediation not supported: at least one path is non-significant")

# ── Top 10 combinations by indirect effect ───────────────────────
print("\n" + "═"*65)
print("TOP 10 LAG COMBINATIONS by |indirect effect|")
print("─"*65)
top = res.reindex(res["indirect"].abs().nlargest(10).index)
print(f"  {'Δa':>3} {'Δb':>3} {'total':>5}  {'a':>7} {'b':>7} {'c':>7} {'c′':>7}  {'a×b':>8}  {'%med':>6}")
for _, row in top.iterrows():
    pm = f"{row.prop_mediated*100:.0f}%" if not np.isnan(row.prop_mediated) else "  —"
    print(f"  {int(row.da):3d} {int(row.db):3d} {int(row.total_lag):5d}"
          f"  {row.r_a:+.3f} {row.r_b:+.3f} {row.r_c:+.3f} {row.r_c_prime:+.3f}"
          f"  {row.indirect:+.5f}  {pm:>6}")

# ── Plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="white")

def setup_ax(ax, title):
    ax.set_facecolor("#f5f5f5")
    ax.set_title(title, color="#1a4a8a", fontsize=15)
    ax.tick_params(colors="black", labelsize=12)
    for sp in ax.spines.values(): sp.set_edgecolor("black")

# Helper: pivot to grid
def make_grid(col):
    pivot = res.pivot_table(index="da", columns="db", values=col, aggfunc="mean")
    return pivot

def heatmap_ax(ax, pivot, title, cmap, vmin=None, vmax=None):
    setup_ax(ax, title)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   origin="lower", vmin=vmin, vmax=vmax,
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(),   pivot.index.max()])
    xlab = f"Δb: {MED} → {Y_label} (s)"
    ylab = f"Δa: {X_label} → {MED} (s)"
    ax.set_xlabel(xlab, color="black", fontsize=13)
    ax.set_ylabel(ylab, color="black", fontsize=13)
    ax.scatter(int(best.db), int(best.da), color="yellow", marker="*", zorder=5, s=200, label=f"best ({int(best.da)},{int(best.db)})")
    ax.legend(fontsize=12, labelcolor="black", facecolor="white")
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(labelsize=11)

heatmap_ax(axes[0,0], make_grid("indirect"),  "Indirect effect a×b",   "RdBu_r")
heatmap_ax(axes[0,1], make_grid("r_a"),       f"Path a: {X_label} → CO₂",    "RdBu_r")
heatmap_ax(axes[0,2], make_grid("r_b"),       f"Path b: CO₂ → {Y_label}",     "RdBu_r")
heatmap_ax(axes[1,0], make_grid("r_c"),       "Total effect c",                "RdBu_r")
heatmap_ax(axes[1,1], make_grid("r_c_prime"), "Direct effect c′ (CO₂ ctrl)",  "RdBu_r")
heatmap_ax(axes[1,2], make_grid("prop_mediated"), "Proportion mediated",       "RdYlGn", 0, 1)

direction_str = f"{X_label}(t) → CO₂(t+Δa) → {Y_label}(t+Δa+Δb)"
fig.suptitle(
    f"Lagged Mediation:  {direction_str}\n"
    f"best Δa={int(best.da)}s  Δb={int(best.db)}s  indirect={best.indirect:+.4f}",
    color="black", fontsize=17, y=1.01)

plt.tight_layout()
import os
suffix = f"_mediation_{'reverse_' if REVERSE else ''}{EMO}_via_{MED}.png"
out = os.path.join(os.path.dirname(os.path.abspath(args.file)),
                   os.path.basename(args.file).replace(".csv", suffix))
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out}")
plt.show()
