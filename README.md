# Biolingo Plant Sensing — Mechanosensory Coupling Analysis

Analysis code accompanying the study:

> **Mechanosensory Coupling Between Human Cardiac Activity and Plant Bioelectric Potentials: A Naturalistic Self-Study with Spectral Validation**
> Peter A. Gloor (2026). *Manuscript under review.*

This repository contains the Python analysis pipeline used to test whether human
emotional states couple to the bioelectric potentials of a co-located
*Kalanchoe daigremontiana* plant, and through which physical pathway. It is one
component of a multi-study research program on plant bioelectric sensing of human
activity using the Biolingo sensor platform.

The central finding the code reproduces: heart rate **fully mediates** the
valence→plant coupling specifically in the 44–75 second oscillation band
(Δb ≈ 34 s), with the valence→HR step (Δa = 8 s) invariant across all eight plant
frequency bands — a frequency specificity that argues against broadband coupling
and points to a frequency-selective transduction mechanism.

---

## Repository structure

```
biolingo-plant-sensing/
├── analysis/
│   ├── analyze_correlations.py   # zero-lag & lagged correlation matrices
│   ├── mediation_lagged.py       # lagged mediation (mean voltage outcome)
│   ├── mediation_spectral.py     # spectral mediation across 8 STFT bands  ← key analysis
│   ├── plant_spectrogram.py      # STFT spectrogram + MFCC-style band energy
│   └── predict_models.py         # Random Forest / XGBoost emotion prediction
├── data/
│   └── README.md                 # expected CSV schema and data availability
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Which script produces which result

| Script | Reproduces | Manuscript |
|--------|------------|------------|
| `analyze_correlations.py` | Pearson/Spearman matrices; lagged correlations | Fig. 2; §4.2–4.3 |
| `mediation_lagged.py`     | Lagged mediation heatmaps (forward & reverse) | Figs. 3–4; §4.4 |
| `mediation_spectral.py`   | Spectral mediation across 8 bands             | Fig. 5, Table 2; §4.5 |
| `plant_spectrogram.py`    | STFT spectrogram + MFCC band-energy matrix    | Fig. 6; §4.6 |
| `predict_models.py`       | RF/XGBoost prediction, feature importance     | Fig. 7; §4.7 |

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/pgloor/biolingo-plant-sensing.git
cd biolingo-plant-sensing
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Data

All scripts read a single CSV (default name `co2_emotion_log.csv`) containing the
1 Hz synchronised sensor streams. The expected column schema and data-availability
details are documented in [`data/README.md`](data/README.md). Place the CSV in the
`data/` directory and pass it with `--file`, e.g. `--file data/co2_emotion_log.csv`.

A few derived columns (`valence_hrv`, `arousal_hrv`, STFT band powers) are computed
on the fly by the scripts and need not be present in the raw CSV.

## Usage

Each script is a standalone command-line tool. Representative invocations:

```bash
# Correlation matrices (Fig. 2) + lagged correlations (§4.3)
python analysis/analyze_correlations.py --file data/co2_emotion_log.csv --lag 60

# Lagged mediation, emotion → CO2 → plant voltage (Figs. 3–4)
python analysis/mediation_lagged.py --file data/co2_emotion_log.csv --emotion valence
python analysis/mediation_lagged.py --file data/co2_emotion_log.csv --emotion valence --reverse

# Spectral mediation across 8 frequency bands (Fig. 5, Table 2) — the key analysis
python analysis/mediation_spectral.py --file data/co2_emotion_log.csv --emotion valence --mediator hr_bpm --max-lag 60

# Plant spectrogram + MFCC band energy (Fig. 6)
python analysis/plant_spectrogram.py --file data/co2_emotion_log.csv --sensor ecg2_mv

# Prediction models, RF/XGBoost (Fig. 7)
python analysis/predict_models.py --file data/co2_emotion_log.csv --lag 60
```

Run any script with `-h` / `--help` for the full set of options (session selection,
hour-of-day cutoff, window-closed CO₂ threshold, date exclusions, etc.). Each script
writes its figure(s) to the working directory as PNG.

## Hardware

Plant bioelectric signals were recorded with the **Biolingo** sensor platform
(galaxyadvisors AG): two independent AD8232 single-lead front-ends on ESP32
microcontrollers (Plant1: RC low-pass, slow-drift; Plant2: 0.1–20 Hz bandpass),
alongside a Sensirion SCD41 (CO₂/temperature/humidity), a Sensirion SGP40 (VOC
index), a Polar H10 heart-rate strap, and HSEmotion facial expression recognition.
This repository covers the analysis only; firmware is maintained separately.

## Reproducibility notes

- The lag grid search is run once on the full dataset; the reported lags
  (Δa = 8 s, Δb = 34 s) are data-derived and, as stated in the manuscript, warrant
  independent replication.
- Findings are from a single-subject design and should be read as exploratory and
  hypothesis-generating rather than confirmatory.

## Citation

If you use this code, please cite the manuscript (see [`CITATION.cff`](CITATION.cff)).
Once a DOI is assigned, it will be added here.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Acknowledgments

Supported by the Software AG Stiftung (Darmstadt) and the Hasler Stiftung (Bern;
Phänomena Plant Sentient Wall project). Thanks to Beat Hächler and Stefan Rentsch
for designing and building the Biolingo sensor hardware.
