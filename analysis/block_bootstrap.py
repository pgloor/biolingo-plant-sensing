"""Session-aware moving-block bootstrap and autocorrelation-adjusted inference for the key Study 1 correlations.
Reproduces the preprocessing of analysis/mediation_lagged.py and analysis/mediation_spectral.py, then
(1) applies lags only within contiguous 1-s runs of the same session (no cross-gap, no cross-session pairs),
(2) computes Bartlett effective sample sizes, and
(3) bootstraps blocks of 300 s within session (B=1000) for 95% CIs."""
import numpy as np, pandas as pd, json, sys
from scipy import signal as sp_signal

B, BLOCK = 1000, 300
rng = np.random.default_rng(42)
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/co2_emotion_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce"); df = df.sort_values("timestamp").reset_index(drop=True)
for c in df.columns:
    if c != "timestamp": df[c] = pd.to_numeric(df[c], errors="coerce")
raw_n = len(df)
df = df[df["ecg_mv"].isna() | (df["ecg_mv"] < 3290)]
df = df[df["ecg2_mv"].isna() | (df["ecg2_mv"] < 3290)]
df = df[df["rmssd"].isna() | df["rmssd"].between(1, 200)]
df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()
jump = df["valence"].diff().abs()
df = df[(df["valence"].abs() <= 0.95) & (jump.isna() | (jump < 0.6))]
hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df = df[hour <= 18]
df = df[(df["valence"] != 0.0) & (df["co2"] >= 500)].reset_index(drop=True)
df["session"] = df["timestamp"].dt.date
print("raw rows", raw_n, "analysis rows", len(df), "sessions", df["session"].nunique())

# STFT band 2 (44–75 s) power of Plant2, as in mediation_spectral.py (WIN=120, STEP=30, periodogram, log1p)
WIN, STEP = 120, 30
edges = np.logspace(np.log10(0.008), np.log10(0.5), 9)
sig = df["ecg2_mv"].values; bp = np.full(len(sig), np.nan)
for fs_ in range(0, len(sig) - WIN, STEP):
    seg = sig[fs_:fs_ + WIN].copy()
    if np.isnan(seg).mean() > 0.3: continue
    seg[np.isnan(seg)] = np.nanmean(seg); seg -= seg.mean()
    f, psd = sp_signal.periodogram(seg, fs=1.0)
    m = (f >= edges[1]) & (f < edges[2]); bp[fs_ + WIN // 2] = np.log1p(psd[m].sum())
df["band2"] = pd.Series(bp).interpolate(limit=2 * STEP).values
df["ecg_d"] = df["ecg_mv"] - df["ecg_mv"].rolling(600, min_periods=30, center=True).mean()
df["val_d"] = df["valence"] - df["valence"].rolling(600, min_periods=30, center=True).mean()

# lag by TIMESTAMP within session: y at t+lag seconds, if that second exists (no cross-session pairs)
ts = df["timestamp"]
sess = df["session"].values
def lagged_pairs(xc, yc, lag):
    y = df.set_index("timestamp")[yc]; y = y[~y.index.duplicated()]
    ys = y.reindex(ts + pd.Timedelta(seconds=lag)).values
    ysess = pd.Series(sess, index=ts)[~pd.Series(sess, index=ts).index.duplicated()].reindex(ts + pd.Timedelta(seconds=lag)).values
    xs = df[xc].values
    m = ~(np.isnan(xs) | np.isnan(ys)) & (ysess == sess)
    return xs[m], ys[m], ts.values[m], sess[m]

def r_of(a, b): return np.corrcoef(a, b)[0, 1]
def n_eff(a, b, maxlag=600):
    a = a - a.mean(); b = b - b.mean(); n = len(a); s = 0.0
    for k in range(1, min(maxlag, n // 2)):
        ra = np.dot(a[:-k], a[k:]) / (n * a.var()); rb = np.dot(b[:-k], b[k:]) / (n * b.var())
        if ra * rb < 0 and k > 30: break
        s += ra * rb
    return n / (1 + 2 * s)

def block_boot(xs, ys, t, se):
    # 300-s calendar blocks within session; resample blocks with replacement, matching count per session
    tsec = (t.astype("datetime64[s]").astype(np.int64))
    blk = pd.Series(list(zip(se, tsec // BLOCK)))
    codes, uniq = pd.factorize(blk)
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    sess_of = np.array([u[0] for u in uniq])
    out = []
    for _ in range(B):
        p = []
        for s0 in np.unique(sess_of):
            g = np.where(sess_of == s0)[0]
            for gi in rng.choice(g, size=len(g), replace=True): p.append(groups[gi])
        p = np.concatenate(p); out.append(r_of(xs[p], ys[p]))
    return np.percentile(out, [2.5, 97.5])

tests = {
    "path_a_valence_to_HR_lag8": ("valence", "hr_bpm", 8),
    "path_b_HR_to_Plant2_band2_lag34": ("hr_bpm", "band2", 34),
    "Plant1_detrended_to_valence_detrended_lag25": ("ecg_d", "val_d", 25),
    "Plant1_CO2_lag0": ("ecg_mv", "co2", 0),
    "VOC_to_valence_lag35": ("voc_index", "valence", 35),
}
res = {}
for name, (xc, yc, lag) in tests.items():
    xs, ys, tt, se = lagged_pairs(xc, yc, lag)
    r = r_of(xs, ys); ne = n_eff(xs, ys); lo, hi = block_boot(xs, ys, tt, se)
    from scipy import stats
    t = r * np.sqrt((ne - 2) / max(1e-9, 1 - r ** 2)); p = 2 * stats.t.sf(abs(t), ne - 2)
    res[name] = dict(n_pairs=int(len(xs)), r=round(float(r), 3), n_eff=int(ne), p_neff=float(f"{p:.2g}"),
                     boot95=[round(float(lo), 3), round(float(hi), 3)])
    print(name, res[name], flush=True)
json.dump(res, open("block_bootstrap_results.json", "w"), indent=1)
