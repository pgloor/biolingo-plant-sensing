"""
mediation_spectral.py
=====================
Spectral mediation: emotion(t) → HR(t+Δa) → Plant2_band_power(t+Δa+Δb)

Instead of mean plant voltage, uses STFT band powers as the outcome Y.
Tests all 8 frequency bands — the key question is which band carries
the emotion→HR→plant coupling.

Usage:
  python mediation_spectral.py --file co2_emotion_log.csv
  python mediation_spectral.py --emotion valence --session 6
  python mediation_spectral.py --emotion valence --mediator hr_bpm --max-lag 60
"""

import argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import pearsonr, t as t_dist
warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file",    default="co2_emotion_log.csv")
parser.add_argument("--emotion", default="valence",
                    choices=["valence","arousal","valence_hrv","arousal_hrv"])
parser.add_argument("--mediator",default="hr_bpm",
                    choices=["hr_bpm","rmssd","co2","voc_index"])
parser.add_argument("--max-lag", type=int, default=60)
parser.add_argument("--stft-window", type=int, default=120)
parser.add_argument("--max-hour", type=int, default=18)
parser.add_argument("--session", type=int, default=None)
parser.add_argument("--window-closed-min", type=float, default=500)
args = parser.parse_args()

FS  = 1.0
WIN = args.stft_window

# ── Load & clean ─────────────────────────────────────────────────
print(f"Loading {args.file}...")
df = pd.read_csv(args.file, on_bad_lines="skip")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
for c in df.columns:
    if c != "timestamp":
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["ecg_mv"].isna() | (df["ecg_mv"] < 3290)]
if "ecg2_mv" in df.columns:
    df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()
if "rmssd" in df.columns:
    df = df[df["rmssd"].isna() | df["rmssd"].between(1, 200)]
if "valence" in df.columns:
    jump = df["valence"].diff().abs()
    df = df[(df["valence"].abs() <= 0.95) & (jump.isna() | (jump < 0.6))]

if "rmssd" in df.columns and "hr_bpm" in df.columns:
    W = 600
    df["valence_hrv"] = (df["rmssd"]
        .rolling(W, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))
    df["arousal_hrv"] = (df["hr_bpm"]
        .rolling(W, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))

df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df = df[df["hour"] <= args.max_hour].reset_index(drop=True)

if args.window_closed_min > 0 and "co2" in df.columns:
    df = df[df["co2"] >= args.window_closed_min].reset_index(drop=True)

gap = df["timestamp"].diff().dt.total_seconds().fillna(0)
df["session"] = (gap > 3600).cumsum()
sessions = df["session"].unique()
print(f"Sessions: {len(sessions)}")
for s in sessions:
    sdf = df[df["session"] == s]
    print(f"  Session {s}: {len(sdf):,} rows  "
          f"{sdf['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} → "
          f"{sdf['timestamp'].iloc[-1].strftime('%H:%M')}")

if args.session is not None:
    df = df[df["session"] == args.session].reset_index(drop=True)
    print(f"Restricted to session {args.session}: {len(df):,} rows")

df = df[df["valence"].notna() & (df["valence"] != 0.0)].reset_index(drop=True)
print(f"Face-present rows: {len(df):,}")

# ── STFT band powers ──────────────────────────────────────────────
N_BANDS = 8
f_min, f_max = 0.008, 0.5
edges = np.logspace(np.log10(f_min), np.log10(f_max), N_BANDS + 1)
band_labels = []
for i in range(N_BANDS):
    band_labels.append(f"{1/edges[i+1]:.0f}–{1/edges[i]:.0f}s")

freqs_full = np.fft.rfftfreq(WIN, d=1.0/FS)

print(f"\nComputing STFT band powers (window={WIN}s, step={WIN//4}s)...")
sig = df["ecg2_mv"].values
band_power = np.full((N_BANDS, len(sig)), np.nan)
STEP = WIN // 4

for frame_start in range(0, len(sig) - WIN, STEP):
    seg = sig[frame_start:frame_start + WIN].copy()
    if np.isnan(seg).mean() > 0.3:
        continue
    seg[np.isnan(seg)] = np.nanmean(seg)
    seg = seg - seg.mean()
    freqs, psd = sp_signal.periodogram(seg, fs=FS)
    centre = frame_start + WIN // 2
    for bi in range(N_BANDS):
        mask = (freqs >= edges[bi]) & (freqs < edges[bi+1])
        band_power[bi, centre] = np.log1p(psd[mask].sum())

for bi in range(N_BANDS):
    s = pd.Series(band_power[bi])
    band_power[bi] = s.interpolate(method="linear", limit=STEP*2).values

for i in range(N_BANDS):
    df[f"band{i+1}"] = band_power[i]

n_valid = df["band1"].notna().sum()
print(f"Band power computed: {n_valid:,} rows with valid features")

# ── Mediation grid search per band ───────────────────────────────
EMO = args.emotion
MED = args.mediator
MAX = args.max_lag

print(f"\n{'='*75}")
print(f"SPECTRAL MEDIATION: {EMO}(t) → {MED}(t+Δa) → Plant2_band(t+Δa+Δb)")
print(f"Δa,Δb grid: 0..{MAX}s each")
print(f"{'='*75}")

def pear(a, b):
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < 30: return 0.0, 1.0
    try:
        r, p = pearsonr(a[valid], b[valid])
        return float(r), float(p)
    except: return 0.0, 1.0

def sigs(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

results = []
for bi in range(N_BANDS):
    band_col = f"band{bi+1}"
    x = df[EMO].values
    m = df[MED].values
    y = df[band_col].values

    best = dict(da=0, db=0, ra=0, pa=1, rb=0, pb=1,
                rc=0, pc=1, rc2=0, pc2=1, indirect=0, n=0)

    for da in range(0, MAX+1):
        for db in range(0, MAX+1):
            if da + db > MAX: continue
            n = len(x) - da - db
            if n < 100: continue
            X_s = x[:n];  M_s = m[da:n+da];  Y_s = y[da+db:n+da+db]
            ra, pa = pear(X_s, M_s)
            rb, pb = pear(M_s, Y_s)
            rc, pc = pear(X_s, Y_s)
            denom = np.sqrt(max(1e-12,(1-ra**2)*(1-rb**2)))
            rc2 = (rc - ra*rb) / denom
            t_s = rc2 * np.sqrt(max(0,(n-2)/(1-rc2**2+1e-12)))
            pc2 = float(2 * t_dist.sf(abs(t_s), df=n-2))
            ind = ra * rb
            if abs(ind) > abs(best["indirect"]):
                best = dict(da=da,db=db,ra=ra,pa=pa,rb=rb,pb=pb,
                            rc=rc,pc=pc,rc2=rc2,pc2=pc2,indirect=ind,n=n)
    results.append(best)

# ── Print table ───────────────────────────────────────────────────
print(f"\n{'Band':>4} {'Period':>10} {'Δa':>5} {'Δb':>5}  "
      f"{'Path a':>12}  {'Path b':>12}  {'a×b':>9}  Verdict")
print("─"*85)
for bi, r in enumerate(results):
    ra,pa = r["ra"],r["pa"]
    rb,pb = r["rb"],r["pb"]
    rc,pc = r["rc"],r["pc"]
    rc2,pc2 = r["rc2"],r["pc2"]
    ind = r["indirect"]
    pm = ind/rc if abs(rc)>1e-6 else float("nan")

    if pa<0.05 and pb<0.05:
        if pc2>0.05:            verdict = "✓ FULL mediation"
        elif abs(rc2)<abs(rc):  verdict = f"✓ PARTIAL {pm*100:.0f}%"
        else:                   verdict = "both sig, no reduction"
    else:
        verdict = "—"

    print(f"  {bi+1:>2}  {band_labels[bi]:>10}  "
          f"{r['da']:>4}s {r['db']:>4}s  "
          f"r={ra:+.3f}{sigs(pa):4s}  "
          f"r={rb:+.3f}{sigs(pb):4s}  "
          f"{ind:>+9.4f}  {verdict}")

# ── Plot ──────────────────────────────────────────────────────────
DARK_BG="white"; AX_BG="#f5f5f5"
fig, axes = plt.subplots(1, 3, figsize=(16,6), facecolor=DARK_BG)
fig.suptitle(
    f"Spectral Mediation: {EMO} → {MED} → Plant2 band power\n"
    f"({'Session '+str(args.session) if args.session is not None else 'All sessions'}, "
    f"n={results[0]['n']:,})",
    color="black", fontsize=15)

def sax(ax, title, xl, yl):
    ax.set_facecolor(AX_BG); ax.set_title(title, color="#1a4a8a", fontsize=13)
    ax.set_xlabel(xl, color="black", fontsize=12); ax.set_ylabel(yl, color="black", fontsize=12)
    ax.tick_params(colors="black", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("black")
    ax.grid(color="#dddddd", lw=0.4, alpha=0.6)

x_pos = np.arange(N_BANDS)

# Path a
ax = axes[0]; sax(ax, f"Path a: {EMO} → {MED} (at best Δa per band)", "Band", "r")
ra_vals = [r["ra"] for r in results]
ca = ["#4fc3f7" if r["pa"]<0.05 else "#bbbbbb" for r in results]
bars = ax.bar(x_pos, ra_vals, color=ca, alpha=0.85)
for i,(b,r) in enumerate(zip(bars,results)):
    ax.text(i, r["ra"]+0.003*np.sign(r["ra"]), f"Δa={r['da']}s",
            ha="center", fontsize=10, color="black")
ax.axhline(0,color="#999999",lw=0.8); ax.set_ylim(-0.3,0.3)
ax.set_xticks(x_pos); ax.set_xticklabels(band_labels, rotation=35, ha="right", fontsize=11)

# Path b
ax = axes[1]; sax(ax, f"Path b: {MED} → Plant2 band (at best Δb per band)", "Band", "r")
rb_vals = [r["rb"] for r in results]
cb = ["#81c784" if r["pb"]<0.05 else "#bbbbbb" for r in results]
bars = ax.bar(x_pos, rb_vals, color=cb, alpha=0.85)
for i,(b,r) in enumerate(zip(bars,results)):
    ax.text(i, r["rb"]+0.003*np.sign(r["rb"]), f"Δb={r['db']}s",
            ha="center", fontsize=10, color="black")
ax.axhline(0,color="#999999",lw=0.8); ax.set_ylim(-0.3,0.3)
ax.set_xticks(x_pos); ax.set_xticklabels(band_labels, rotation=35, ha="right", fontsize=11)

# Indirect effect
ax = axes[2]; sax(ax, "Indirect effect a×b (mediation strength per band)", "Band", "a×b")
ind_vals = [r["indirect"] for r in results]
mediated = ["✓" in r2["verdict"] for r2 in [
    dict(verdict=("✓ FULL mediation" if results[bi]["pa"]<0.05 and results[bi]["pb"]<0.05
                  and results[bi]["pc2"]>0.05
                  else "✓ PARTIAL" if results[bi]["pa"]<0.05 and results[bi]["pb"]<0.05
                  and abs(results[bi]["rc2"])<abs(results[bi]["rc"])
                  else "—"))
    for bi in range(N_BANDS)]]
ci = ["#ffd54f" if m else "#aaaaaa" for m in mediated]
ax.bar(x_pos, ind_vals, color=ci, alpha=0.85)
ax.axhline(0,color="#999999",lw=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(band_labels, rotation=35, ha="right", fontsize=11)

# Mark the mechanosensory band (27-75s = Bands 2-3)
for ax in axes:
    ax.axvspan(0.5, 2.5, alpha=0.08, color="#ffd54f", zorder=0,
               label="Mechanosensory range (27-75s)")

plt.tight_layout()
suffix = f"_s{args.session}" if args.session is not None else ""
fname = f"mediation_spectral_{EMO}_via_{MED}{suffix}.png"
fig.savefig(fname, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"\nPlot saved → {fname}")
