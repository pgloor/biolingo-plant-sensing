# Data

All analysis scripts expect a single CSV of 1 Hz synchronised sensor readings.
The default filename is `co2_emotion_log.csv`; place it in this `data/` directory
and pass it to any script with `--file data/co2_emotion_log.csv`.

## Expected columns

| Column        | Unit / range        | Description |
|---------------|---------------------|-------------|
| `timestamp`   | ISO 8601 datetime   | 1 Hz sample time |
| `co2`         | ppm                 | Sensirion SCD41 CO₂ |
| `temperature` | °C                  | SCD41 temperature |
| `humidity`    | %                   | SCD41 relative humidity |
| `ecg_mv`      | mV                  | Plant1 bioelectric voltage (AD8232, RC low-pass; 5 s mean) |
| `ecg2_mv`     | mV                  | Plant2 bioelectric voltage (AD8232, 0.1–20 Hz bandpass) |
| `leads_on`    | 0/1                 | Electrode-contact flag (optional) |
| `emotion`     | label               | HSEmotion top categorical label |
| `valence`     | [−1, +1]            | FER valence |
| `arousal`     | [−1, +1]            | FER arousal |
| `rmssd`       | ms                  | HRV RMSSD (60 s window) from Polar H10 |
| `hr_bpm`      | bpm                 | Heart rate from Polar H10 |
| `voc_index`   | 0–500               | Sensirion SGP40 VOC Index (compensated) |
| `voc_raw`     | ohms (optional)     | Raw SGP40 resistance — excluded from analysis (temperature/humidity confounded) |

Not every column is required by every script (e.g. the correlation script tolerates
missing columns and uses pairwise-complete observations). The derived columns
`valence_hrv`, `arousal_hrv`, and the STFT band powers are **computed by the scripts**
and should not be present in the raw CSV.

## Cleaning applied by the scripts

The scripts apply the same preprocessing described in the manuscript, including:
removal of disconnected-electrode rows (`ecg_mv` ≥ 3290 mV), implausible RMSSD
(outside 1–200 ms), an FER quality filter (|valence| > 0.95 or jump > 0.6/s),
an hour-of-day cutoff (default ≤ 18:00, to exclude CAM-photosynthesis effects),
and an optional window-closed CO₂ threshold.

## Availability

The raw dataset for the manuscript is archived separately. For a citable, versioned
release, deposit the CSV on a DOI-issuing repository (e.g. Zenodo or Dryad) and link
it here:

- **Dataset DOI:** _to be added_

For peer review, the data should be made available at submission rather than upon
acceptance.
