# Audio Distortion Effects on CLAP Cross-Modal Retrieval

## Project Idea and Goals

Cross-modal audio-text retrieval systems like [LAION-CLAP](https://github.com/LAION-AI/CLAP) are trained on clean audio, but real-world audio is rarely pristine — it may be filtered, noisy, or pitch-shifted. This project investigates how common signal distortions degrade CLAP's ability to match audio clips to their text descriptions.

Our goals:
- Quantify how four types of distortion (high-pass filter, low-pass filter, additive noise, pitch shift) affect retrieval performance
- Identify which frequency or timbral properties CLAP relies on most for audio-text matching
- Provide a reproducible evaluation pipeline on the AudioCaps benchmark dataset

## Data

We used **[AudioCaps](https://audiocaps.github.io/)**, a large-scale dataset of YouTube audio clips paired with human-written text captions, built as a subset of AudioSet.

- **500 clips** sampled from AudioCaps, each trimmed to 10 seconds
- Downloaded directly from YouTube using the AudioCaps metadata (video IDs + timestamps)
- Each clip comes with a human-written text caption used as the retrieval query
- The raw audio files are not tracked in this repository (local only); use `src/download.py` to reproduce the dataset

We applied four distortion conditions to every clip, producing five versions per audio file (original + 4 distorted):

| Condition | Description |
|-----------|-------------|
| original | clean audio (baseline) |
| highpass_1000Hz | high-pass filter at 1000 Hz — removes low-frequency content |
| lowpass_4000Hz | low-pass filter at 4000 Hz — removes high-frequency content |
| noisy_snr10dB | Gaussian white noise at SNR = 10 dB |
| pitch_shift | pitch-shifted audio (−3 semitones via librosa) |

## Code Structure

```
.
├── src/                    # preprocessing scripts
│   ├── download.py         # download 10s clips from YouTube (AudioCaps)
│   ├── add_noise.py        # add Gaussian white noise (configurable SNR)
│   ├── high_pass.py        # high-pass filter (default cutoff: 1000 Hz)
│   ├── low_pass.py         # low-pass filter (default cutoff: 4000 Hz)
│   └── pitch_shift.py      # pitch shifting
├── notebooks/
│   ├── 01_extract_embeddings.ipynb        # extract CLAP audio & text embeddings
│   ├── 02_similarity_analysis.ipynb       # similarity matrix & distribution analysis
│   └── 03_processed_audio_analysis.ipynb  # compare retrieval metrics across distortion conditions
├── outputs/
│   ├── embeddings/         # saved .npy embedding files (intermediate outputs)
│   ├── figures/            # plots and visualizations (.png)
│   └── metrics/            # retrieval metrics (.csv)
├── CLAP/                   # LAION-CLAP source code
├── env/                    # conda environment file
└── data/                   # audio files & metadata (local only, not tracked)
```

**Pipeline:**
```
Download audio → Apply distortions → Extract CLAP embeddings → Compute retrieval metrics → Analyze
```

## Results and Key Findings

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

**Key findings:**

- **Low-pass filtering (≤4000 Hz) causes almost no degradation** — R@1 and MedR remain nearly identical to the clean baseline. CLAP appears to encode enough discriminative information in low-to-mid frequencies.
- **High-pass filtering and pitch shifting both cause large drops** — R@1 falls ~35% and MedR doubles from 3 to 7. This suggests CLAP's audio encoder is sensitive to the loss of low-frequency energy and to changes in pitch.
- **Additive noise at SNR=10 dB causes a moderate drop** — performance degrades but does not collapse, indicating some robustness to mild noise.
- Overall, the model's cross-modal matching relies more heavily on **high-frequency content and pitch** than on low-frequency energy.

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

## Team & Contributions

**Group 6**

| Member | Contributions |
|--------|--------------|
| Maxzavier Guo | `src/download.py`, `notebooks/01_extract_embeddings.ipynb` |
| Enqi Lian | `notebooks/02_similarity_analysis.ipynb`, `notebooks/03_processed_audio_analysis.ipynb` |
| Dustin Chen | `src/add_noise.py` |
| Yiming Zhao | `src/high_pass.py`, `src/low_pass.py` |
| Haosen Sun | `src/pitch_shift.py` |
