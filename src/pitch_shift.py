#!/usr/bin/env python3
"""Create a pitch-shifted copy of the raw audio dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_input_dir = project_root / "data" / "audio"
    default_metadata_csv = project_root / "data" / "metadata_500.csv"
    default_output_dir = project_root / "data" / "audio_pitch_shift"
    default_output_metadata = project_root / "data" / "metadata_500_pitch_shift.csv"

    parser = argparse.ArgumentParser(
        description="Apply pitch shifting to every WAV file in the dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--metadata-csv", type=Path, default=default_metadata_csv)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--output-metadata-csv",
        type=Path,
        default=default_output_metadata,
    )
    parser.add_argument(
        "--failures-csv",
        type=Path,
        default=None,
        help="Optional CSV for files that could not be processed.",
    )
    parser.add_argument(
        "--audio-filename-column",
        default="audio_filename",
        help="Column containing filenames relative to --input-dir.",
    )
    parser.add_argument(
        "--audio-path-column",
        default="audio_path",
        help="Fallback column containing absolute or relative source paths.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*.wav",
        help="Pattern used when no metadata CSV is available.",
    )
    parser.add_argument(
        "--n-steps",
        type=float,
        default=2.0,
        help="Pitch-shift amount in semitones. Positive raises pitch, negative lowers it.",
    )
    parser.add_argument(
        "--bins-per-octave",
        type=int,
        default=12,
        help="Number of bins per octave used by librosa.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.bins_per_octave <= 0:
        raise ValueError("--bins-per-octave must be positive.")


def load_metadata_or_files(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    if args.metadata_csv.exists():
        metadata_df = pd.read_csv(args.metadata_csv)
        rows = metadata_df.to_dict(orient="records")
        source = f"metadata file: {args.metadata_csv}"
        return rows, [source] * len(rows)

    audio_files = sorted(args.input_dir.glob(args.glob_pattern))
    rows = [{"audio_filename": audio_file.name} for audio_file in audio_files]
    source = f"directory scan: {args.input_dir / args.glob_pattern}"
    return rows, [source] * len(rows)


def resolve_input_path(
    row: dict,
    input_dir: Path,
    audio_filename_column: str,
    audio_path_column: str,
) -> Path:
    filename_value = row.get(audio_filename_column)
    if pd.notna(filename_value) and str(filename_value).strip():
        candidate = input_dir / str(filename_value)
        if candidate.exists():
            return candidate

    path_value = row.get(audio_path_column)
    if pd.notna(path_value) and str(path_value).strip():
        candidate = Path(str(path_value))
        if candidate.exists():
            return candidate
        fallback = input_dir / candidate.name
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Could not resolve input audio from columns "
        f"'{audio_filename_column}' or '{audio_path_column}'."
    )


def apply_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: float,
    bins_per_octave: int,
) -> np.ndarray:
    if audio.ndim == 1:
        shifted = librosa.effects.pitch_shift(
            y=audio,
            sr=sample_rate,
            n_steps=n_steps,
            bins_per_octave=bins_per_octave,
        )
    else:
        shifted = np.stack(
            [
                librosa.effects.pitch_shift(
                    y=channel,
                    sr=sample_rate,
                    n_steps=n_steps,
                    bins_per_octave=bins_per_octave,
                )
                for channel in audio
            ],
            axis=0,
        )
    return np.clip(shifted, -1.0, 1.0).astype(np.float32)


def to_soundfile_layout(audio: np.ndarray) -> np.ndarray:
    return audio.T if audio.ndim == 2 else audio


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sources = load_metadata_or_files(args)

    if args.failures_csv is None:
        args.failures_csv = args.output_metadata_csv.with_name(
            f"{args.output_metadata_csv.stem}_failures.csv"
        )

    processed_rows: list[dict] = []
    failed_rows: list[dict] = []

    for row, row_source in tqdm(
        zip(rows, sources),
        total=len(rows),
        desc="Applying pitch shift",
    ):
        row_copy = dict(row)
        try:
            input_path = resolve_input_path(
                row_copy,
                args.input_dir,
                args.audio_filename_column,
                args.audio_path_column,
            )
            output_path = args.output_dir / input_path.name

            source_info = sf.info(input_path)
            audio, sample_rate = librosa.load(input_path, sr=None, mono=False)
            audio = np.asarray(audio, dtype=np.float32)

            shifted_audio = apply_pitch_shift(
                audio=audio,
                sample_rate=sample_rate,
                n_steps=args.n_steps,
                bins_per_octave=args.bins_per_octave,
            )

            sf.write(
                output_path,
                to_soundfile_layout(shifted_audio),
                sample_rate,
                subtype=source_info.subtype,
            )

            row_copy["source_audio_path"] = str(input_path)
            row_copy["audio_filename"] = output_path.name
            row_copy["audio_path"] = str(output_path)
            row_copy["augmentation"] = "pitch_shift"
            row_copy["pitch_shift_steps"] = args.n_steps
            row_copy["pitch_shift_bins_per_octave"] = args.bins_per_octave
            processed_rows.append(row_copy)
        except Exception as exc:  # noqa: BLE001
            row_copy["failure_reason"] = str(exc)
            row_copy["record_source"] = row_source
            failed_rows.append(row_copy)

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(args.output_metadata_csv, index=False)

    failed_df = pd.DataFrame(failed_rows)
    failed_df.to_csv(args.failures_csv, index=False)

    print(f"Processed files: {len(processed_rows)}")
    print(f"Failed files: {len(failed_rows)}")
    print(f"Pitch-shifted audio saved to: {args.output_dir}")
    print(f"Output metadata saved to: {args.output_metadata_csv}")
    print(f"Failure log saved to: {args.failures_csv}")


if __name__ == "__main__":
    main()
