"""
给 data/audio/ 下的所有音频做高通滤波，输出到 data/audio_highpass_{XXXX}Hz/

特点
----
- 用 cutoff (Hz) 控制高通截止频率，保留高于该频率的成分
- 文件名 / 采样率 / 长度保持不变，可以直接接到 CLAP pipeline 里
- 自动跳过空文件 / 损坏文件（如 34.wav, 196.wav, 247.wav）
- 默认使用 Butterworth 滤波器，适合做稳定的 perturbation 实验

使用方式
--------
1) 单个 cutoff（最常用）：
       python src/high_pass.py --cutoff 1000

   输出到: data/audio_highpass_1000Hz/

2) 扫多个 cutoff 做 robustness 实验：
       for cutoff in 250 500 1000 2000; do
           python src/high_pass.py --cutoff $cutoff
       done

3) 在 notebook 里改 AUDIO_DIR 即可使用：
       AUDIO_DIR = os.path.join(PROJECT_DIR, 'data', 'audio_highpass_1000Hz')

参数说明
--------
--cutoff        高通截止频率（Hz）。常用: 250, 500, 1000, 2000。数值越大低频损失越强。
--order         Butterworth 滤波器阶数，默认 5。
--clean-dir     干净音频目录，默认 data/audio/。
--out-dir       输出目录，默认 data/audio_highpass_{cutoff}Hz/。
--metadata-csv  可选 CSV，按 audio_filename 列决定处理哪些文件；
                默认 data/metadata_500.csv，没有则处理 clean-dir 下所有 wav。
"""

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


# =========================
# 1. 路径设置（和 download.py / add_noise.py 保持一致）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_DIR = os.path.dirname(BASE_DIR)                  # 项目根目录

CLEAN_AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
METADATA_CSV = os.path.join(PROJECT_DIR, "data", "metadata_500.csv")


# =========================
# 2. 核心函数：高通滤波
# =========================
def high_pass_filter(signal: np.ndarray, sr: int, cutoff_hz: float, order: int = 5) -> np.ndarray:
    """
    对 signal 做高通滤波，保留高于 cutoff_hz 的频率成分。

    cutoff_hz 必须满足 0 < cutoff_hz < sr / 2。
    """
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be greater than 0")
    if cutoff_hz >= sr / 2:
        raise ValueError(f"cutoff_hz must be lower than Nyquist frequency ({sr / 2:.1f} Hz)")

    signal = signal.astype(np.float32)
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")

    # 对多声道音频也适用：axis=0 表示沿时间维滤波。
    # padtype='constant' 避免极短音频触发 filtfilt padding 异常，保证所有文件行为一致。
    filtered = sosfiltfilt(sos, signal, axis=0, padtype="constant")

    filtered = np.clip(filtered, -1.0, 1.0).astype(np.float32)
    return filtered


# =========================
# 3. 批处理：遍历所有音频
# =========================
def high_pass_dataset(
    clean_dir: str,
    out_dir: str,
    cutoff_hz: float,
    order: int = 5,
    metadata_csv: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # 决定要处理哪些文件
    if metadata_csv is not None and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        filenames = df["audio_filename"].tolist()
    else:
        filenames = sorted(f for f in os.listdir(clean_dir) if f.lower().endswith(".wav"))

    n_ok, n_skip = 0, 0
    skipped = []

    for fname in tqdm(filenames, desc=f"High-pass filtering @ cutoff={cutoff_hz:g}Hz"):
        in_path = os.path.join(clean_dir, fname)
        out_path = os.path.join(out_dir, fname)

        if not os.path.exists(in_path):
            skipped.append((fname, "missing"))
            n_skip += 1
            continue

        try:
            signal, sr = sf.read(in_path, always_2d=False)
        except Exception as e:
            skipped.append((fname, f"read_error: {e}"))
            n_skip += 1
            continue

        if signal.size == 0:
            skipped.append((fname, "empty"))
            n_skip += 1
            continue

        try:
            filtered = high_pass_filter(signal, sr=sr, cutoff_hz=cutoff_hz, order=order)
        except Exception as e:
            skipped.append((fname, f"filter_error: {e}"))
            n_skip += 1
            continue

        # 用 PCM_16 保存，和 add_noise.py 保持一致；想保留更高精度可以用 'FLOAT'
        sf.write(out_path, filtered, samplerate=sr, subtype="PCM_16")
        n_ok += 1

    print(f"\nDone. saved={n_ok}, skipped={n_skip}, out_dir={out_dir}")
    if skipped:
        print("Skipped files:")
        for f, why in skipped:
            print(f"  - {f}: {why}")


# =========================
# 4. CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Apply high-pass filtering to all audio files.")
    parser.add_argument("--cutoff", type=float, default=1000.0,
                        help="高通截止频率 (Hz)。常用值: 250, 500, 1000, 2000。")
    parser.add_argument("--order", type=int, default=5,
                        help="Butterworth 滤波器阶数，默认 5。")
    parser.add_argument("--clean-dir", type=str, default=CLEAN_AUDIO_DIR,
                        help="干净音频文件夹。")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出文件夹，默认 data/audio_highpass_{cutoff}Hz。")
    parser.add_argument("--metadata-csv", type=str, default=METADATA_CSV,
                        help="可选：metadata CSV，用 audio_filename 列决定处理哪些文件。")
    args = parser.parse_args()

    cutoff_label = f"{args.cutoff:g}"
    out_dir = args.out_dir or os.path.join(
        PROJECT_DIR, "data", f"audio_highpass_{cutoff_label}Hz"
    )

    high_pass_dataset(
        clean_dir=args.clean_dir,
        out_dir=out_dir,
        cutoff_hz=args.cutoff,
        order=args.order,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
