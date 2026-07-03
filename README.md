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
