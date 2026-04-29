"""
Apply high-pass filtering to all audio files in data/audio/.

Usage:
    python src/high_pass.py --cutoff 1000
"""

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CLEAN_AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
METADATA_CSV    = os.path.join(PROJECT_DIR, "data", "metadata_500.csv")


def high_pass_filter(signal: np.ndarray, sr: int, cutoff_hz: float, order: int = 5) -> np.ndarray:
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be greater than 0")
    if cutoff_hz >= sr / 2:
        raise ValueError(f"cutoff_hz must be lower than Nyquist frequency ({sr / 2:.1f} Hz)")

    signal = signal.astype(np.float32)
    sos    = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")

    # padtype='constant' handles very short files without falling back to causal filtering
    filtered = sosfiltfilt(sos, signal, axis=0, padtype="constant")
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)


def high_pass_dataset(
    clean_dir: str,
    out_dir: str,
    cutoff_hz: float,
    order: int = 5,
    metadata_csv: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    if metadata_csv is not None and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        filenames = df["audio_filename"].tolist()
    else:
        filenames = sorted(f for f in os.listdir(clean_dir) if f.lower().endswith(".wav"))

    n_ok, n_skip = 0, 0
    skipped = []

    for fname in tqdm(filenames, desc=f"High-pass filtering @ cutoff={cutoff_hz:g}Hz"):
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

        try:
            filtered = high_pass_filter(signal, sr=sr, cutoff_hz=cutoff_hz, order=order)
        except Exception as e:
            skipped.append((fname, f"filter_error: {e}")); n_skip += 1; continue

        sf.write(out_path, filtered, samplerate=sr, subtype="PCM_16")
        n_ok += 1

    print(f"\nDone. saved={n_ok}, skipped={n_skip}, out_dir={out_dir}")
    if skipped:
        print("Skipped files:")
        for f, why in skipped:
            print(f"  - {f}: {why}")


def main():
    parser = argparse.ArgumentParser(description="Apply high-pass filtering to all audio files.")
    parser.add_argument("--cutoff",       type=float, default=1000.0, help="High-pass cutoff frequency in Hz (e.g. 250, 500, 1000, 2000).")
    parser.add_argument("--order",        type=int,   default=5,      help="Butterworth filter order.")
    parser.add_argument("--clean-dir",    type=str,   default=CLEAN_AUDIO_DIR)
    parser.add_argument("--out-dir",      type=str,   default=None)
    parser.add_argument("--metadata-csv", type=str,   default=METADATA_CSV)
    args = parser.parse_args()

    cutoff_label = f"{args.cutoff:g}"
    out_dir = args.out_dir or os.path.join(PROJECT_DIR, "data", f"audio_highpass_{cutoff_label}Hz")

    high_pass_dataset(
        clean_dir    = args.clean_dir,
        out_dir      = out_dir,
        cutoff_hz    = args.cutoff,
        order        = args.order,
        metadata_csv = args.metadata_csv,
    )


if __name__ == "__main__":
    main()
