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
        mode: str = "logmel",
    ) -> None:
        super().__init__()
        if mode not in {"logmel", "pcen", "logmel_pcen"}:
            raise ValueError(f"Unknown spectrogram mode: {mode}")
        self.mode = mode
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
        batched = waveform.ndim == 2
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        power = self.mel(waveform)
        logmel = self.amplitude_to_db(power)
        logmel = ((logmel + 80.0) / 80.0).clamp(0.0, 1.0)
        if batched:
            logmel = logmel.unsqueeze(1)
        if self.mode == "logmel":
            return logmel

        pcen = self._pcen(power)
        if batched:
            pcen = pcen.unsqueeze(1)
        if self.mode == "pcen":
            return pcen
        return torch.cat([logmel, pcen], dim=1 if batched else 0)

    @staticmethod
    def _pcen(
        power: torch.Tensor,
        gain: float = 0.98,
        bias: float = 2.0,
        power_exp: float = 0.5,
        smooth: float = 0.025,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        # PCEN emphasizes local acoustic events and is a common BirdCLEF sidecar feature.
        smoother = torch.zeros_like(power)
        smoother[..., 0] = power[..., 0]
        for t in range(1, power.shape[-1]):
            smoother[..., t] = (1.0 - smooth) * smoother[..., t - 1] + smooth * power[..., t]
        pcen = (power / (eps + smoother).pow(gain) + bias).pow(power_exp) - bias**power_exp
        pcen_min = pcen.amin(dim=(-2, -1), keepdim=True)
        pcen_max = pcen.amax(dim=(-2, -1), keepdim=True)
        return ((pcen - pcen_min) / (pcen_max - pcen_min + eps)).clamp(0.0, 1.0)
