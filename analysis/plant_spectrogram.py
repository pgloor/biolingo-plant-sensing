"""
plant_spectrogram.py
====================
Spectral analysis of plant bioelectric signals (Plant1, Plant2) from co2_emotion_log.csv.

Computes:
  1. Spectrogram (STFT) of plant voltage — how does spectral content evolve over time?
  2. MFCC-style features adapted for plant frequencies (0.001–0.5 Hz)
  3. Correlation of spectral band power with emotion/HRV over time
  4. Session-averaged power spectra: emotional vs neutral periods

Usage:
  python plant_spectrogram.py --file co2_emotion_log.csv --sensor ecg2_mv
  python plant_spectrogram.py --file co2_emotion_log.csv --sensor ecg_mv
  python plant_spectrogram.py --file co2_emotion_log.csv --sensor both
  python plant_spectrogram.py --file co2_emotion_log.csv --sensor ecg2_mv --session 4
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal as sp_signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--file",    default="co2_emotion_log.csv")
parser.add_argument("--sensor",  default="ecg2_mv",
                    choices=["ecg_mv", "ecg2_mv", "both"],
                    help="Plant sensor to analyse")
parser.add_argument("--session", type=int, default=None,
                    help="Restrict to a single session (0-indexed)")
parser.add_argument("--window",  type=int, default=120,
                    help="STFT window size in seconds (default 120)")
parser.add_argument("--overlap", type=float, default=0.75,
                    help="STFT overlap fraction (default 0.75)")
parser.add_argument("--emotion", default="valence_hrv",
                    choices=["valence","arousal","valence_hrv","arousal_hrv","all"],
                    help="Emotion target for band correlation (default: valence_hrv)")
parser.add_argument("--lag",     type=int, default=35,
                    help="Seconds to lag emotion behind plant spectral features (default: 35)")
parser.add_argument("--max-hour", type=int, default=18)
args = parser.parse_args()

FS = 1.0  # 1 Hz sampling rate (1 row per second in CSV)

# ── Load & clean ─────────────────────────────────────────────────
print(f"Loading {args.file}...")
df = pd.read_csv(args.file, on_bad_lines="skip")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)
for c in df.columns:
    if c != "timestamp":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Artifact filters
df = df[df["ecg_mv"].isna() | (df["ecg_mv"] < 3290)]
if "ecg2_mv" in df.columns:
    df["ecg2_mv"] = df["ecg2_mv"].rolling(5, min_periods=1).mean()
if "rmssd" in df.columns:
    df = df[df["rmssd"].isna() | df["rmssd"].between(1, 200)]

# FER quality filter
if "valence" in df.columns:
    jump = df["valence"].diff().abs()
    df = df[(df["valence"].abs() <= 0.95) & (jump.isna() | (jump < 0.6))]

# Hour filter
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df = df[df["hour"] <= args.max_hour].reset_index(drop=True)

# HRV-derived emotion
if "rmssd" in df.columns and "hr_bpm" in df.columns:
    W = 600
    df["valence_hrv"] = (df["rmssd"]
        .rolling(W, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))
    df["arousal_hrv"] = (df["hr_bpm"]
        .rolling(W, min_periods=30)
        .apply(lambda x: np.clip((x.iloc[-1]-x.mean())/(x.std()+1e-6),-3,3)/3, raw=False))

# Session detection
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
    print(f"\nRestricted to session {args.session}: {len(df):,} rows")

# ── STFT parameters ──────────────────────────────────────────────
WIN     = args.window           # seconds = samples at 1Hz
OVERLAP = int(WIN * args.overlap)
STEP    = WIN - OVERLAP
NOVERLAP = OVERLAP

# Frequency axis for WIN-point FFT at 1Hz
freqs_stft = np.fft.rfftfreq(WIN, d=1.0/FS)  # 0 to 0.5 Hz

# Plant-adapted MFCC filterbank (logarithmically spaced in plant-frequency domain)
# 8 bands covering 0.008–0.5 Hz (periods 2–120s)
N_MFCC = 8
f_min, f_max = 0.008, 0.5
# Log-spaced band edges
edges = np.logspace(np.log10(f_min), np.log10(f_max), N_MFCC + 1)
band_labels = []
for i in range(N_MFCC):
    period_lo = 1/edges[i+1] if edges[i+1] > 0 else np.inf
    period_hi = 1/edges[i]   if edges[i]   > 0 else np.inf
    band_labels.append(f"{period_lo:.0f}–{period_hi:.0f}s")

def make_filterbank(freqs, edges):
    """Triangular filterbank for MFCC-style features."""
    n_freqs = len(freqs)
    fb = np.zeros((len(edges)-1, n_freqs))
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        mid = (lo + hi) / 2
        for j, f in enumerate(freqs):
            if lo <= f < mid:
                fb[i, j] = (f - lo) / (mid - lo + 1e-12)
            elif mid <= f <= hi:
                fb[i, j] = (hi - f) / (hi - mid + 1e-12)
    return fb

filterbank = make_filterbank(freqs_stft, edges)

def compute_stft_features(sig, win=WIN, noverlap=NOVERLAP):
    """
    Returns:
      times     : centre times of each frame (seconds from start)
      freqs     : frequency axis (Hz)
      Sxx       : power spectrogram (n_freqs × n_frames)
      mfcc_log  : log band energies (n_bands × n_frames)
    """
    sig = np.array(sig, dtype=float)
    # Replace NaN with local mean
    nan_mask = np.isnan(sig)
    if nan_mask.all():
        return None, None, None, None
    sig[nan_mask] = np.nanmean(sig)

    freqs, times, Sxx = sp_signal.spectrogram(
        sig, fs=FS, window="hann", nperseg=win,
        noverlap=noverlap, scaling="density")

    # Log filterbank energies (MFCC-style)
    # Sxx shape: (n_freqs, n_times)
    mfcc_log = np.log1p(filterbank @ Sxx)  # (n_bands, n_times)

    return times, freqs, Sxx, mfcc_log


# ── Plot helper ──────────────────────────────────────────────────
DARK_BG  = "white"
AX_BG    = "#f5f5f5"
GRID_COL = "#cccccc"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="#1a4a8a", fontsize=15, pad=6)
    ax.set_xlabel(xlabel, color="black", fontsize=12)
    ax.set_ylabel(ylabel, color="black", fontsize=13)
    ax.tick_params(colors="black", labelsize=11)
    for sp in ax.spines.values(): sp.set_edgecolor("black")
    ax.grid(color=GRID_COL, linewidth=0.4, alpha=0.6)


def plot_sensor(sensor_col, df_use, out_suffix=""):
    sig = df_use[sensor_col].values
    t_abs = df_use["timestamp"].values

    print(f"\n[{sensor_col}] Computing STFT ({len(sig):,} samples)...")
    times, freqs, Sxx, mfcc_log = compute_stft_features(sig)
    if times is None:
        print(f"  Insufficient data in {sensor_col}")
        return

    # Frame timestamps (absolute)
    t0 = pd.Timestamp(t_abs[0])
    frame_ts = [t0 + pd.Timedelta(seconds=float(t)) for t in times]
    frame_min = np.array([t.total_seconds()/60 for t in
                          [pd.Timestamp(ft) - t0 for ft in frame_ts]])

    # ── Figure layout ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle(f"Plant Spectrogram & MFCC — {sensor_col}  "
                 f"({'Session '+str(args.session) if args.session is not None else 'all sessions'})",
                 color="black", fontsize=17, y=0.98)

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3,
                          left=0.07, right=0.97, top=0.94, bottom=0.06)

    # ── 1. Raw signal ─────────────────────────────────────────────
    ax_raw = fig.add_subplot(gs[0, :])
    style_ax(ax_raw, f"Raw {sensor_col} (1Hz)", "Time (min)", "mV")
    t_min = np.arange(len(sig)) / 60
    ax_raw.plot(t_min, sig, lw=0.5, color="#4fc3f7", alpha=0.8)
    ax_raw.set_xlim(0, t_min[-1])

    # Overlay valence if available
    if "valence" in df_use.columns:
        val = df_use["valence"].values
        ax2 = ax_raw.twinx()
        ax2.plot(t_min, val, lw=0.6, color="#ce93d8", alpha=0.5, label="Valence")
        ax2.set_ylabel("Valence", color="#ce93d8", fontsize=12)
        ax2.tick_params(colors="#ce93d8", labelsize=11)
        ax2.set_ylim(-1.5, 1.5)

    # ── 2. Power spectrogram ──────────────────────────────────────
    ax_spec = fig.add_subplot(gs[1, :])
    style_ax(ax_spec, "Power Spectrogram (STFT)", "Time (min)", "Frequency (Hz)")
    # Only show 0–0.2Hz (periods > 5s) where plant signals live
    f_mask = freqs <= 0.20
    Sxx_db = 10 * np.log10(Sxx[f_mask, :] + 1e-12)
    vmin, vmax = np.percentile(Sxx_db, [5, 95])
    im = ax_spec.imshow(Sxx_db, aspect="auto", origin="lower",
                        extent=[frame_min[0], frame_min[-1],
                                freqs[f_mask][0], freqs[f_mask][-1]],
                        cmap="viridis", vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(im, ax=ax_spec, label="Power (dB)", shrink=0.8)
    cb1.ax.tick_params(labelsize=11)
    # Mark the 0.029Hz coupling frequency (35s period)
    ax_spec.axhline(1/35, color="#ff6b6b", lw=1.0, ls="--", alpha=0.8,
                    label="0.029 Hz (35s coupling)")
    ax_spec.axhline(1/9,  color="#ffd166", lw=0.8, ls=":",  alpha=0.7,
                    label="0.111 Hz (9s coupling)")
    ax_spec.legend(fontsize=11, loc="upper right",
                   facecolor=AX_BG, edgecolor="black", labelcolor="black")

    # ── 3. MFCC log band energies ─────────────────────────────────
    ax_mfcc = fig.add_subplot(gs[2, :])
    style_ax(ax_mfcc, "Log Band Energies (MFCC-style)", "Time (min)", "Band")
    im2 = ax_mfcc.imshow(mfcc_log, aspect="auto", origin="lower",
                          extent=[frame_min[0], frame_min[-1], 0, N_MFCC],
                          cmap="magma")
    cb2 = plt.colorbar(im2, ax=ax_mfcc, label="log(1 + energy)", shrink=0.8)
    cb2.ax.tick_params(labelsize=11)
    ax_mfcc.set_yticks(np.arange(N_MFCC) + 0.5)
    ax_mfcc.set_yticklabels(band_labels, fontsize=11)

    # ── 4a. Mean power spectrum ───────────────────────────────────
    ax_psd = fig.add_subplot(gs[3, 0])
    style_ax(ax_psd, "Mean Power Spectrum", "Frequency (Hz)", "Power (dB)")
    mean_psd = 10 * np.log10(Sxx.mean(axis=1) + 1e-12)
    ax_psd.plot(freqs[1:], mean_psd[1:], color="#4fc3f7", lw=1.2)
    ax_psd.axvline(1/35, color="#ff6b6b", lw=1.0, ls="--", label="35s")
    ax_psd.axvline(1/9,  color="#ffd166", lw=0.8, ls=":",  label="9s")
    ax_psd.set_xlim(0, 0.2)
    ax_psd.legend(fontsize=11, facecolor=AX_BG, labelcolor="black")

    # ── 4b. Band-emotion correlations ────────────────────────────
    ax_corr = fig.add_subplot(gs[3, 1])
    style_ax(ax_corr, f"Band Power × Emotion (lag={args.lag}s)", "Band", "Pearson r")

    if args.emotion == "all":
        emotion_cols = [c for c in ["valence_hrv","arousal_hrv","valence","arousal"]
                        if c in df_use.columns]
    else:
        emotion_cols = [args.emotion] if args.emotion in df_use.columns else []

    colors_emo = {"valence":"#ce93d8","arousal":"#80cbc4",
                  "valence_hrv":"#2980b9","arousal_hrv":"#8e44ad"}

    x = np.arange(N_MFCC)
    width = 0.8 / max(len(emotion_cols), 1)
    for ei, ecol in enumerate(emotion_cols):
        # Lag emotion by args.lag seconds after plant spectral feature
        emo_raw = df_use[ecol].values
        emo_lagged = np.roll(emo_raw, -args.lag)  # shift emotion forward in time
        emo_lagged[-args.lag:] = np.nan

        corrs = []
        for bi in range(N_MFCC):
            band_ts = mfcc_log[bi, :]
            band_interp = np.interp(
                np.arange(len(emo_lagged)),
                np.linspace(0, len(emo_lagged)-1, len(band_ts)),
                band_ts)
            valid = ~np.isnan(emo_lagged) & ~np.isnan(band_interp)
            if valid.sum() < 30:
                corrs.append(0.0)
            else:
                r, _ = pearsonr(band_interp[valid], emo_lagged[valid])
                corrs.append(r)
        offset = (ei - len(emotion_cols)/2 + 0.5) * width
        ax_corr.bar(x + offset, corrs, width=width*0.9,
                    color=colors_emo.get(ecol, "#aaa"),
                    label=ecol, alpha=0.85)

    ax_corr.axhline(0, color="#999999", lw=0.8)
    ax_corr.set_xticks(x)
    ax_corr.set_xticklabels(band_labels, rotation=30, ha="right", fontsize=11)
    ax_corr.legend(fontsize=11, facecolor=AX_BG, labelcolor="black",
                   ncol=2, loc="upper right")
    ax_corr.set_ylim(-0.3, 0.3)

    # ── Save ─────────────────────────────────────────────────────
    fname = f"plant_spectrogram_{sensor_col}{out_suffix}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {fname}")

    # Print top correlating bands
    emo_col = (args.emotion if args.emotion != "all" else "valence_hrv")
    if emo_col not in df_use.columns:
        emo_col = "valence" if "valence" in df_use.columns else None
    print(f"\n  Band–Emotion correlations ({emo_col}, lag={args.lag}s):")
    if emo_col:
        emo_raw    = df_use[emo_col].values
        emo_lagged = np.roll(emo_raw, -args.lag)
        emo_lagged[-args.lag:] = np.nan
        for bi in range(N_MFCC):
            band_ts = mfcc_log[bi, :]
            band_interp = np.interp(
                np.arange(len(emo_lagged)),
                np.linspace(0, len(emo_lagged)-1, len(band_ts)),
                band_ts)
            valid = ~np.isnan(emo_lagged) & ~np.isnan(band_interp)
            if valid.sum() > 30:
                r, p = pearsonr(band_interp[valid], emo_lagged[valid])
                sig_str = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
                print(f"    Band {bi+1} ({band_labels[bi]:>12s}): r={r:+.3f} {sig_str}")


# ── Run analysis ─────────────────────────────────────────────────
suffix = f"_s{args.session}" if args.session is not None else ""

sensors = (["ecg_mv", "ecg2_mv"] if args.sensor == "both"
           else [args.sensor])

for s in sensors:
    if s not in df.columns or df[s].notna().sum() < 200:
        print(f"Skipping {s} — insufficient data")
        continue
    df_use = df[df[s].notna()].copy()
    plot_sensor(s, df_use, out_suffix=suffix)

print("\nDone.")
