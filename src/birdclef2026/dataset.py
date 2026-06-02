from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import LogMelExtractor, crop_or_pad, load_audio


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        sample_rate: int = 32000,
        duration: float = 10.0,
        train: bool = True,
        use_precomputed: bool = False,
        spec_mode: str = "logmel",
        spec_augment_p: float = 0.0,
        time_masks: int = 2,
        freq_masks: int = 2,
        time_mask_width: int = 48,
        freq_mask_width: int = 16,
    ) -> None:
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.train = train
        self.use_precomputed = use_precomputed
        self.spec_mode = spec_mode
        if not use_precomputed:
            self.extractor = LogMelExtractor(sample_rate=sample_rate, mode=spec_mode)
        self.spec_augment_p = spec_augment_p
        self.time_masks = time_masks
        self.freq_masks = freq_masks
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.target_cols = [c for c in self.df.columns if c.startswith("target_")]
        if not self.target_cols:
            raise ValueError("Manifest must contain target_0 ... target_N columns.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rng = np.random.default_rng()

        if self.use_precomputed and "spec_path" in row and not pd.isna(row["spec_path"]):
            # 直接从预计算的 .npy 加载频谱（大幅加速）
            spec = torch.from_numpy(np.load(row["spec_path"]).astype(np.float32))
            if self.train and self.spec_augment_p > 0 and rng.random() < self.spec_augment_p:
                spec = self._spec_augment(spec, rng)
            target = torch.tensor(row[self.target_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
            return spec, target

        # --- 实时计算模式（原始逻辑）---
        fixed_num_samples = self.num_samples
        row_num_samples = fixed_num_samples
        if "num_samples" in row and not pd.isna(row.get("num_samples")):
            row_num_samples = int(row["num_samples"])

        audio = load_audio(row["path"], sample_rate=self.sample_rate)

        if "start_sample" in row and not pd.isna(row.get("start_sample")):
            start = int(row["start_sample"])
            audio = audio[start : start + row_num_samples]
        else:
            audio = audio[:row_num_samples]

        audio = crop_or_pad(audio, fixed_num_samples, random_crop=self.train, rng=rng)
        if self.train:
            gain = float(rng.uniform(0.7, 1.3))
            audio = (audio * gain).clip(-1.0, 1.0)

        waveform = torch.from_numpy(audio)
        spec = self.extractor(waveform)
        if self.train and self.spec_augment_p > 0 and rng.random() < self.spec_augment_p:
            spec = self._spec_augment(spec, rng)
        target = torch.tensor(row[self.target_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        return spec, target

    def _spec_augment(self, spec: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        for _ in range(self.freq_masks):
            width = int(rng.integers(0, min(self.freq_mask_width, n_mels) + 1))
            if width > 0:
                start = int(rng.integers(0, n_mels - width + 1))
                spec[:, start : start + width, :] = 0.0
        for _ in range(self.time_masks):
            width = int(rng.integers(0, min(self.time_mask_width, n_frames) + 1))
            if width > 0:
                start = int(rng.integers(0, n_frames - width + 1))
                spec[:, :, start : start + width] = 0.0
        return spec
