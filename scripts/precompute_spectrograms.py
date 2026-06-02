"""
预计算训练数据的 mel 频谱并保存为 .npy 文件。

训练时每轮都需要从 OGG 计算频谱，非常耗时。预计算后直接从 .npy 加载，
I/O 速度提升 5-10x。

用法:
    python scripts/precompute_spectrograms.py --meta-dir data/processed --out-dir data/processed/spec_cache

预计算完成后，训练时加上 --use-precomputed 即可：
    python train.py ... --meta-dir data/processed --use-precomputed
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from birdclef2026.audio import LogMelExtractor, crop_or_pad, load_audio


def precompute_manifest(
    manifest_path: Path,
    out_dir: Path,
    sample_rate: int = 32000,
    duration: float = 10.0,
) -> Path:
    """对 manifest 中每条音频预计算 mel 频谱，返回新 manifest 路径。"""
    df = pd.read_csv(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = LogMelExtractor(sample_rate=sample_rate)
    num_samples_target = int(sample_rate * duration)

    spec_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing spectrograms"):
        audio_id = Path(row["path"]).stem
        row_num_samples = num_samples_target
        if "num_samples" in row and not pd.isna(row.get("num_samples")):
            row_num_samples = int(row["num_samples"])

        audio = load_audio(row["path"], sample_rate=sample_rate, mono=True)

        if "start_sample" in row and not pd.isna(row.get("start_sample")):
            start = int(row["start_sample"])
            audio = audio[start : start + row_num_samples]
        else:
            audio = audio[:row_num_samples]

        # 固定长度 pad，与训练时一致
        audio = crop_or_pad(audio, num_samples_target, random_crop=False)

        # 计算 mel 频谱
        import torch
        waveform = torch.from_numpy(audio.astype(np.float32))
        spec = extractor(waveform).numpy().astype(np.float16)  # fp16 节省磁盘空间

        # 文件名: {audio_id}.npy
        spec_file = out_dir / f"{audio_id}.npy"
        np.save(spec_file, spec)
        spec_paths.append(str(spec_file))

    new_manifest = manifest_path.parent / f"{manifest_path.stem}_precomputed.csv"
    new_df = df.copy()
    new_df["spec_path"] = spec_paths
    new_df.to_csv(new_manifest, index=False)
    print(f"Saved {len(new_df)} spectrograms to {out_dir}")
    print(f"New manifest: {new_manifest}")
    return new_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="预计算 mel 频谱加速训练")
    parser.add_argument("--meta-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/spec_cache"))
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="音频统一长度（秒），需与训练 --duration 一致")
    args = parser.parse_args()

    # 预计算 train_manifest
    train_manifest = args.meta_dir / "train_manifest.csv"
    if train_manifest.exists():
        precompute_manifest(train_manifest, args.out_dir, args.sample_rate, args.duration)

    # 预计算 soundscape_manifest
    soundscape_manifest = args.meta_dir / "soundscape_manifest.csv"
    if soundscape_manifest.exists():
        precompute_manifest(soundscape_manifest, args.out_dir, args.sample_rate, args.duration)

    print("\n预计算完成！训练时加上 --use-precomputed 即可使用缓存频谱。")


if __name__ == "__main__":
    main()
