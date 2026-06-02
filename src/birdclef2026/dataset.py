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
    ) -> None:
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.train = train
        self.extractor = LogMelExtractor(sample_rate=sample_rate)
        self.target_cols = [c for c in self.df.columns if c.startswith("target_")]
        if not self.target_cols:
            raise ValueError("Manifest must contain target_0 ... target_N columns.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rng = np.random.default_rng()
        num_samples = self.num_samples
        if "num_samples" in row and not pd.isna(row.get("num_samples")):
            num_samples = int(row["num_samples"])
        audio = load_audio(row["path"], sample_rate=self.sample_rate)

        if "start_sample" in row and not pd.isna(row.get("start_sample")):
            start = int(row["start_sample"])
            audio = audio[start : start + num_samples]

        audio = crop_or_pad(audio, num_samples, random_crop=self.train, rng=rng)
        if self.train:
            gain = float(rng.uniform(0.7, 1.3))
            audio = (audio * gain).clip(-1.0, 1.0)

        waveform = torch.from_numpy(audio)
        spec = self.extractor(waveform)
        target = torch.tensor(row[self.target_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        return spec, target
