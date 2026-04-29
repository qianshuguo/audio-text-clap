"""
给 data/audio/ 下的所有音频加高斯白噪声，输出到 data/audio_noisy_snr{XX}dB/

特点
----
- 用 SNR (dB) 控制噪音相对强度（不同响度的音频被污染的程度一致）
- 文件名 / 采样率 / 长度保持不变，可以直接接到 CLAP pipeline 里
- 固定随机种子，多人/多次复跑结果一致
- 自动跳过空文件 / 损坏文件（如 34.wav, 196.wav, 247.wav）

使用方式
--------
1) 单个 SNR（最常用）：
       python src/add_noise.py --snr 10

   输出到: data/audio_noisy_snr10dB/

2) 扫多个 SNR 做 robustness 实验：
       for snr in 20 10 5 0 -5; do
           python src/add_noise.py --snr $snr
       done

3) 在 notebook 里改 AUDIO_DIR 即可使用：
       AUDIO_DIR = os.path.join(PROJECT_DIR, 'data', 'audio_noisy_snr10dB')

参数说明
--------
--snr           目标信噪比（dB）。常用: 20, 10, 5, 0, -5。数值越小噪音越强。
--seed          随机种子，默认 42。
--clean-dir     干净音频目录，默认 data/audio/。
--out-dir       输出目录，默认 data/audio_noisy_snr{SNR}dB/。
--metadata-csv  可选 CSV，按 audio_filename 列决定处理哪些文件；
                默认 data/metadata_500.csv，没有则处理 clean-dir 下所有 wav。
"""

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# =========================
# 1. 路径设置（和 download.py 保持一致）
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_DIR = os.path.dirname(BASE_DIR)                  # 项目根目录

CLEAN_AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
METADATA_CSV = os.path.join(PROJECT_DIR, "data", "metadata_500.csv")


# =========================
# 2. 核心函数：加高斯白噪声
# =========================
def add_gaussian_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    在 signal 上叠加高斯白噪声，使得 SNR = snr_db。

    SNR_dB = 10 * log10(P_signal / P_noise)
    => P_noise = P_signal / 10^(SNR_dB / 10)
    """
    signal = signal.astype(np.float32)

    # 信号功率（均方值）。对多声道也适用：在所有样本上取均值。
    signal_power = np.mean(signal ** 2)

    # 极端情况：纯静音文件，没法定义 SNR，直接返回原信号
    if signal_power <= 1e-12:
        return signal

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape).astype(np.float32)
    noisy = signal + noise

    # 防止保存成 wav 时被裁剪（float32 wav 理论上没问题，但保险起见 clip 到 [-1, 1]）
    noisy = np.clip(noisy, -1.0, 1.0)
    return noisy


# =========================
# 3. 批处理：遍历所有音频
# =========================
def add_noise_to_dataset(
    clean_dir: str,
    out_dir: str,
    snr_db: float,
    seed: int = 42,
    metadata_csv: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # 决定要处理哪些文件
    if metadata_csv is not None and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        filenames = df["audio_filename"].tolist()
    else:
        filenames = sorted(f for f in os.listdir(clean_dir) if f.lower().endswith(".wav"))

    rng = np.random.default_rng(seed)

    n_ok, n_skip = 0, 0
    skipped = []

    for fname in tqdm(filenames, desc=f"Adding noise @ SNR={snr_db}dB"):
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

        noisy = add_gaussian_noise(signal, snr_db=snr_db, rng=rng)

        # 用 PCM_16 保存，和原文件保持一致的常见格式；想保留更高精度可以用 'FLOAT'
        sf.write(out_path, noisy, samplerate=sr, subtype="PCM_16")
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
    parser = argparse.ArgumentParser(description="Add Gaussian white noise to all audio files.")
    parser.add_argument("--snr", type=float, default=10.0,
                        help="目标信噪比 (dB)。常用值: 20, 10, 5, 0, -5。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现。")
    parser.add_argument("--clean-dir", type=str, default=CLEAN_AUDIO_DIR,
                        help="干净音频文件夹。")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出文件夹，默认 data/audio_noisy_snr{SNR}dB。")
    parser.add_argument("--metadata-csv", type=str, default=METADATA_CSV,
                        help="可选：metadata CSV，用 audio_filename 列决定处理哪些文件。")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        PROJECT_DIR, "data", f"audio_noisy_snr{int(args.snr)}dB"
    )

    add_noise_to_dataset(
        clean_dir=args.clean_dir,
        out_dir=out_dir,
        snr_db=args.snr,
        seed=args.seed,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
