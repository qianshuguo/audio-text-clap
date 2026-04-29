"""
Add Gaussian white noise to all audio files in data/audio/.

Usage:
    python src/add_noise.py --snr 10
"""

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CLEAN_AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
METADATA_CSV    = os.path.join(PROJECT_DIR, "data", "metadata_500.csv")


def add_gaussian_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = signal.astype(np.float32)

    signal_power = np.mean(signal ** 2)
    if signal_power <= 1e-12:
        return signal

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std   = np.sqrt(noise_power)

    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape).astype(np.float32)
    noisy = signal + noise

    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy


def add_noise_to_dataset(
    clean_dir: str,
    out_dir: str,
    snr_db: float,
    seed: int = 42,
    metadata_csv: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    if metadata_csv is not None and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        filenames = sorted(df["audio_filename"].tolist())
    else:
        filenames = sorted(f for f in os.listdir(clean_dir) if f.lower().endswith(".wav"))

    rng = np.random.default_rng(seed)

    n_ok, n_skip = 0, 0
    skipped = []

    for fname in tqdm(filenames, desc=f"Adding noise @ SNR={snr_db}dB"):
        in_path  = os.path.join(clean_dir, fname)
        out_path = os.path.join(out_dir, fname)

        if not os.path.exists(in_path):
            skipped.append((fname, "missing")); n_skip += 1; continue

        try:
            signal, sr = sf.read(in_path, always_2d=False)
        except Exception as e:
            skipped.append((fname, f"read_error: {e}")); n_skip += 1; continue

        if signal.size == 0:
            skipped.append((fname, "empty")); n_skip += 1; continue

        noisy = add_gaussian_noise(signal, snr_db=snr_db, rng=rng)
        sf.write(out_path, noisy, samplerate=sr, subtype="PCM_16")
        n_ok += 1

    print(f"\nDone. saved={n_ok}, skipped={n_skip}, out_dir={out_dir}")
    if skipped:
        print("Skipped files:")
        for f, why in skipped:
            print(f"  - {f}: {why}")


def main():
    parser = argparse.ArgumentParser(description="Add Gaussian white noise to all audio files.")
    parser.add_argument("--snr",          type=float, default=10.0,          help="Target SNR in dB (e.g. 20, 10, 5, 0, -5).")
    parser.add_argument("--seed",         type=int,   default=42,            help="Random seed.")
    parser.add_argument("--clean-dir",    type=str,   default=CLEAN_AUDIO_DIR)
    parser.add_argument("--out-dir",      type=str,   default=None)
    parser.add_argument("--metadata-csv", type=str,   default=METADATA_CSV)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(PROJECT_DIR, "data", f"audio_noisy_snr{int(args.snr)}dB")

    add_noise_to_dataset(
        clean_dir    = args.clean_dir,
        out_dir      = out_dir,
        snr_db       = args.snr,
        seed         = args.seed,
        metadata_csv = args.metadata_csv,
    )


if __name__ == "__main__":
    main()
