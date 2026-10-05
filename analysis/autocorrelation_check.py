"""Autocorrelation of the Study 1 signals at selected lags (seconds), after the standard preprocessing
of mediation_lagged.py. Motivates the effective-sample-size and block-bootstrap inference in block_bootstrap.py."""
import numpy as np, pandas as pd, sys
from scipy import signal as sp
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/co2_emotion_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce"); df = df.sort_values("timestamp")
for c in df.columns:
    if c != "timestamp": df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["ecg_mv"].isna() | (df["ecg_mv"] < 3290)]; df = df[df["rmssd"].isna() | df["rmssd"].between(1, 200)]
df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()
j = df["valence"].diff().abs(); df = df[(df["valence"].abs() <= 0.95) & (j.isna() | (j < 0.6))]
h = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60; df = df[h <= 18]
df = df[(df["valence"] != 0) & (df["co2"] >= 500)].reset_index(drop=True)
sig = df["ecg2_mv"].values; WIN, STEP = 120, 30; edges = np.logspace(np.log10(.008), np.log10(.5), 9); bp = np.full(len(sig), np.nan)
for s in range(0, len(sig) - WIN, STEP):
    seg = sig[s:s + WIN].copy()
    if np.isnan(seg).mean() > .3: continue
    seg[np.isnan(seg)] = np.nanmean(seg); seg -= seg.mean(); f, p = sp.periodogram(seg, fs=1.0)
    m = (f >= edges[1]) & (f < edges[2]); bp[s + WIN // 2] = np.log1p(p[m].sum())
df["band2"] = pd.Series(bp).interpolate(limit=60).values
def acf(x, lags):
    x = pd.Series(x).dropna().values; x = x - x.mean(); v = x.var()
    return [round(float(np.dot(x[:-k], x[k:]) / (len(x) * v)), 3) for k in lags]
lags = [1, 5, 10, 30, 60, 120, 300, 600]
print("lag (s):", lags)
for c in ("valence", "hr_bpm", "rmssd", "co2", "voc_index", "ecg_mv", "ecg2_mv", "band2"):
    print(f"{c:10s}", acf(df[c].values, lags))
