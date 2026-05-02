# audio-text-clap

A study on how audio distortions affect cross-modal retrieval performance using the [LAION-CLAP](https://github.com/LAION-AI/CLAP) model.

## Overview

We downloaded 500 audio clips from AudioSet, applied different types of signal distortions, extracted CLAP embeddings, and measured how much retrieval performance drops under each condition.

## Project Structure

```
.
├── src/                    # preprocessing scripts
│   ├── download.py         # download 10s clips from YouTube (AudioSet)
│   ├── add_noise.py        # add Gaussian white noise (configurable SNR)
│   ├── high_pass.py        # high-pass filter (default cutoff: 1000 Hz)
│   ├── low_pass.py         # low-pass filter (default cutoff: 4000 Hz)
│   └── pitch_shift.py      # pitch shifting
├── notebooks/
│   ├── 01_extract_embeddings.ipynb        # extract CLAP audio & text embeddings
│   ├── 02_similarity_analysis.ipynb       # similarity matrix & distribution analysis
│   └── 03_processed_audio_analysis.ipynb  # compare results across distortion conditions
├── results/                # saved embeddings, retrieval metrics, and plots
├── CLAP/                   # LAION-CLAP source code
├── env/                    # conda environment file
└── data/                   # audio files & metadata (local only, not tracked)
```

## Pipeline

```
Download audio → Apply distortions → Extract CLAP embeddings → Compute retrieval metrics → Analyze
```

**Distortion conditions**

| Condition | Description |
|-----------|-------------|
| original | clean audio (baseline) |
| highpass_1000Hz | high-pass filter at 1000 Hz, removes low-frequency content |
| lowpass_4000Hz | low-pass filter at 4000 Hz, removes high-frequency content |
| noisy_snr10dB | Gaussian white noise added at SNR = 10 dB |
| pitch_shift | pitch-shifted audio |

## Results

Retrieval metrics (R@1 / R@5 / R@10 / MedR) evaluated on 500 samples:

| Condition | Direction | R@1 | R@5 | R@10 | MedR |
|-----------|-----------|-----|-----|------|------|
| original | A→T | 0.264 | 0.668 | 0.803 | 3 |
| original | T→A | 0.245 | 0.690 | 0.827 | 3 |
| highpass_1000Hz | A→T | 0.171 | 0.443 | 0.606 | 7 |
| highpass_1000Hz | T→A | 0.165 | 0.495 | 0.646 | 6 |
| lowpass_4000Hz | A→T | 0.264 | 0.650 | 0.797 | 3 |
| lowpass_4000Hz | T→A | 0.219 | 0.660 | 0.803 | 3 |
| noisy_snr10dB | A→T | 0.219 | 0.586 | 0.728 | 4 |
| noisy_snr10dB | T→A | 0.266 | 0.622 | 0.787 | 4 |
| pitch_shift | A→T | 0.167 | 0.445 | 0.584 | 7 |
| pitch_shift | T→A | 0.159 | 0.469 | 0.644 | 6 |

**Key finding:** Low-pass filtering barely hurts performance, while high-pass filtering and pitch shifting cause a significant drop (R@1 down ~35%, MedR goes from 3 to 7). This suggests CLAP relies more on high-frequency and pitch information than low-frequency content for audio-text matching.

## Getting Started

### Environment Setup

```bash
conda env create -f env/environment.yml
conda activate dl4m-clap
pip install -e ./CLAP
```

### Download Data

```bash
python src/download.py
```

### Apply Distortions

```bash
python src/add_noise.py --snr 10
python src/high_pass.py --cutoff 1000
python src/low_pass.py --cutoff 4000
python src/pitch_shift.py
```

### Run Analysis

Run the three notebooks in `notebooks/` in order.
