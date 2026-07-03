# Human Activity Recognition with Hidden Markov Models

Recognizing human activities (standing, walking, jumping, still) from
smartphone accelerometer and gyroscope data using a Gaussian Hidden Markov
Model implemented from scratch in numpy.

## Overview

- Data collected with the Sensor Logger app at ~100 Hz, resampled onto a
  uniform 50 Hz grid so all recordings are comparable
- 1-second sliding windows with 50% overlap; 7 features per window
  (5 time-domain, 2 frequency-domain derived from the FFT)
- Gaussian HMM (diagonal covariance) implemented from scratch:
  Forward, Backward, Baum-Welch (EM) training with a log-likelihood
  convergence check, and Viterbi decoding, all in log space
- Evaluated on 15 unseen test files: 88.6% overall accuracy, with 97.3%
  on the normal-intensity mixed session and a documented failure case on
  deliberately gentle jumps (distribution shift analysis in the report)

## Repository contents

| Path | Description |
|---|---|
| `hmm_activity_recognition.ipynb` | Full pipeline with saved outputs |
| `data/train/` | Labelled 5-second training CSVs + mixed training session |
| `data/test/` | Unseen test CSVs, incl. two labelled mixed sessions (`*_labels.csv`) |
| `report/` | Project report (PDF) and filled formative template (PDF) |
| `figures/` | Result figures (convergence, matrices, decoded sequences) |

Each data CSV has columns: `time, ax, ay, az, gx, gy, gz`
(accelerometer and gyroscope, merged onto shared timestamps).

## How to run

### Option A: run with the processed data included in this repo (fastest)

The `data/` folder already contains the sliced, labelled CSV files, so no
raw recordings or Google Drive setup are needed.

1. Open `hmm_activity_recognition.ipynb` in Google Colab.
2. Upload the `data/` folder from this repo into the Colab session so it
   sits at `/content/data` (i.e. `/content/data/train` and
   `/content/data/test`).
3. Run Cell 1 (imports and configuration). When the Google Drive
   authorization prompt appears you can approve it; Drive is not actually
   used in this option.
4. Skip Cell 2 (data preparation; it rebuilds the dataset from raw zips).
5. Run Cell 3 and everything after it.

### Option B: reproduce everything from raw recordings

1. Record each activity with the Sensor Logger app (accelerometer and
   gyroscope enabled) and export each recording as a zip. Each zip must
   contain `Accelerometer.csv` and `Gyroscope.csv`, which is Sensor
   Logger's default export format.
2. Name each single-activity zip starting with its activity:
   `standing...`, `walking...`, `jumping...`, `still...`. Recordings that
   contain several activities can have any other name (for example
   `all_activities...`) and are treated as mixed sessions.
3. Put all zips in a Google Drive folder and set `RAW_ZIPS` in Cell 1 to
   that folder's path (default: `/content/drive/MyDrive/hmm_data`).
4. Mixed recordings that should become labelled test files must be listed
   in the `MIXED_TESTS` dictionary at the top of Cell 2, with the time (in
   seconds, measured after the 3-second edge trim) at which each activity
   ends. All other mixed recordings go entirely to training, where
   Baum-Welch uses them without labels to learn realistic activity
   transitions.
5. Runtime, then Run all. Cell 2 unzips every export, merges the two
   sensors onto shared timestamps, trims 3 seconds from each end to remove
   start/stop artifacts, slices single-activity recordings into labelled
   5-second files (holding out every 6th slice as unseen test data), and
   writes `data/train` and `data/test` automatically.

## Method summary

1. **Preprocessing:** merge sensors, trim edges, resample to 50 Hz,
   slice into labelled 5 s files.
2. **Features:** per 1 s window: mean acceleration magnitude, log std,
   log RMS, log SMA, log gyroscope std, dominant frequency (FFT), log
   spectral energy (FFT). Z-score normalization fitted on training data
   only.
3. **Model:** hidden states = 4 activities; observations = 7-d feature
   vectors; emissions = diagonal Gaussians; transitions = 4x4 matrix.
   Informed initialization from labelled files, then joint refinement
   with Baum-Welch (stops when the change in total log-likelihood falls
   below 1e-4). Hidden states are mapped to activity names by majority
   vote on decoded training files.
4. **Evaluation:** Viterbi decoding of unseen files; per-class
   sensitivity and specificity, overall accuracy, confusion matrix, and
   decoded-sequence plots against ground truth.

## Results

| State (Activity) | Samples | Sensitivity | Specificity |
|---|---|---|---|
| Standing | 93 | 0.882 | 0.987 |
| Walking | 223 | 0.978 | 0.804 |
| Jumping | 121 | 0.686 | 1.000 |
| Still | 36 | 1.000 | 1.000 |

Overall accuracy: **0.886** across all unseen test data. On the
normal-intensity mixed session the model reaches **0.973** with jumping
sensitivity 0.966; the lower combined jumping score comes entirely from a
second mixed session kept deliberately as a robustness probe, in which
gentle jumps overlap the walking intensity distribution. Full analysis in
`report/`.
