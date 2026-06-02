from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly


def load_audio(path: str | Path, sample_rate: int = 32000, mono: bool = True) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2 and mono:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def crop_or_pad(
    audio: np.ndarray,
    length: int,
    random_crop: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if len(audio) == length:
        return audio
    if len(audio) < length:
        reps = int(np.ceil(length / max(1, len(audio))))
        audio = np.tile(audio, reps)
        return audio[:length].astype(np.float32)
    if random_crop:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, len(audio) - length + 1))
    else:
        start = max(0, (len(audio) - length) // 2)
    return audio[start : start + length].astype(np.float32)


class LogMelExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: int = 20,
        f_max: int = 16000,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.mel(waveform)
        spec = self.amplitude_to_db(spec)
        spec = (spec + 80.0) / 80.0
        return spec.clamp(0.0, 1.0)

