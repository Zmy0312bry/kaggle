from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import resample_poly
from torch import nn
from tqdm import tqdm


# Paste this whole file into one Kaggle Notebook cell, or run it as a script.
# If CHECKPOINT_PATHS is empty, the script searches /kaggle/input for fold*_best.pt,
# then falls back to every .pt/.pth/.ckpt file it can find.
DATA_DIR = Path("/kaggle/input/birdclef-2026")
OUT_PATH = Path("/kaggle/working/submission.csv")
CHECKPOINT_PATHS: list[str] = []
SAMPLE_RATE = 32000
WINDOW_SECONDS = 5
BATCH_SIZE = 32
TTA = 0
NUM_THREADS = max(1, min(4, os.cpu_count() or 2))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_species_list(data_dir: str | Path, sample_submission: str | Path | None = None) -> list[str]:
    data_dir = Path(data_dir)
    if sample_submission is not None and Path(sample_submission).exists():
        sample = pd.read_csv(sample_submission, nrows=1)
        return [c for c in sample.columns if c != "row_id"]
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, nrows=1)
        return [c for c in sample.columns if c != "row_id"]
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    return taxonomy["primary_label"].astype(str).tolist()


def load_audio(path: str | Path, sample_rate: int = 32000, mono: bool = True) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2 and mono:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def crop_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio.astype(np.float32)
    if len(audio) < length:
        reps = int(np.ceil(length / max(1, len(audio))))
        return np.tile(audio, reps)[:length].astype(np.float32)
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


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), kernel_size=x.shape[-2:]).pow(1.0 / self.p).flatten(1)
        if x.ndim == 3:
            return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return x


class FeaturePool(nn.Module):
    def __init__(self, pooling: str = "avg") -> None:
        super().__init__()
        self.pooling = pooling
        self.gem = GeM()

    @property
    def feature_multiplier(self) -> int:
        return 2 if self.pooling == "avgmax" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return x
        if self.pooling == "gem":
            return self.gem(x)
        if x.ndim == 4:
            avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
            if self.pooling == "max":
                return F.adaptive_max_pool2d(x, 1).flatten(1)
            if self.pooling == "avgmax":
                return torch.cat([avg, F.adaptive_max_pool2d(x, 1).flatten(1)], dim=1)
            return avg
        avg = x.mean(dim=1)
        if self.pooling == "max":
            return x.max(dim=1).values
        if self.pooling == "avgmax":
            return torch.cat([avg, x.max(dim=1).values], dim=1)
        return avg


class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.2,
        pooling: str = "avg",
        head_hidden: int = 0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("This notebook needs timm. Add a Kaggle Dataset containing timm wheels, or use Kaggle's installed timm.") from exc
        model_kwargs = {}
        if drop_path_rate > 0:
            model_kwargs["drop_path_rate"] = drop_path_rate
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                global_pool="",
                **model_kwargs,
            )
        except TypeError:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                global_pool="",
            )
        self.pool = FeaturePool(pooling)
        head_in = self.backbone.num_features * self.pool.feature_multiplier
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(head_in, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(head_in, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x)))


def parse_row_id(row_id: str) -> tuple[str, int]:
    audio_id, end_s = row_id.rsplit("_", 1)
    return audio_id, int(float(end_s))


def discover_checkpoints() -> list[Path]:
    if CHECKPOINT_PATHS:
        return [Path(p) for p in CHECKPOINT_PATHS]
    root = Path("/kaggle/input")
    best = sorted(root.rglob("fold*_best.pt"))
    if best:
        return best
    candidates: list[Path] = []
    for pattern in ["*.pt", "*.pth", "*.ckpt"]:
        candidates.extend(root.rglob(pattern))
    return sorted(candidates)


def load_models(checkpoint_paths: Iterable[Path], num_classes: int, device: torch.device) -> list[nn.Module]:
    models = []
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model_name = ckpt.get("model_name", "tf_efficientnet_b0_ns") if isinstance(ckpt, dict) else "tf_efficientnet_b0_ns"
        model = BirdCLEFModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=float(ckpt.get("dropout", 0.2)) if isinstance(ckpt, dict) else 0.2,
            pooling=str(ckpt.get("pooling", "avg")) if isinstance(ckpt, dict) else "avg",
            head_hidden=int(ckpt.get("head_hidden", 0)) if isinstance(ckpt, dict) else 0,
            drop_path_rate=float(ckpt.get("drop_path", 0.0)) if isinstance(ckpt, dict) else 0.0,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
        print(f"Loaded {ckpt_path.name}: model={model_name}")
    if not models:
        raise FileNotFoundError("No checkpoint found. Set CHECKPOINT_PATHS near the top of this cell.")
    return models


def make_variants(clips: np.ndarray, tta: int) -> list[np.ndarray]:
    variants = [clips]
    if tta >= 1:
        variants.append(clips[:, ::-1].copy())
    if tta >= 2:
        variants.append(np.clip(clips * 1.15, -1.0, 1.0))
    return variants


@torch.no_grad()
def predict_clips(
    models: list[nn.Module],
    extractor: LogMelExtractor,
    clips: np.ndarray,
    device: torch.device,
    batch_size: int,
    tta: int,
) -> np.ndarray:
    all_logits = []
    for model in models:
        model_logits = []
        for variant in make_variants(clips, tta):
            logits = []
            for start in range(0, len(variant), batch_size):
                batch_np = np.ascontiguousarray(variant[start : start + batch_size])
                waveform = torch.from_numpy(batch_np).to(device)
                spec = extractor(waveform).to(device)
                logits.append(model(spec).cpu().numpy())
            model_logits.append(np.concatenate(logits, axis=0))
        all_logits.append(np.mean(model_logits, axis=0))
    return sigmoid(np.mean(all_logits, axis=0)).astype(np.float32)


def build_audio_clips(audio: np.ndarray, end_seconds: list[int], sample_rate: int) -> np.ndarray:
    clip_len = WINDOW_SECONDS * sample_rate
    clips = []
    for end_s in end_seconds:
        start = max(0, int((end_s - WINDOW_SECONDS) * sample_rate))
        clip = audio[start : start + clip_len]
        clips.append(crop_or_pad(clip, clip_len))
    return np.stack(clips).astype(np.float32)


def main() -> None:
    torch.set_num_threads(NUM_THREADS)
    sample_path = DATA_DIR / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    species = read_species_list(DATA_DIR, sample_path)
    device = torch.device("cpu")
    checkpoints = discover_checkpoints()
    print("Checkpoints:")
    for path in checkpoints:
        print(f"  {path}")
    models = load_models(checkpoints, num_classes=len(species), device=device)

    extractor = LogMelExtractor(sample_rate=SAMPLE_RATE).to(device).eval()
    grouped: dict[str, list[tuple[str, int]]] = {}
    for row_id in sample["row_id"].astype(str):
        audio_id, end_s = parse_row_id(row_id)
        grouped.setdefault(audio_id, []).append((row_id, end_s))

    test_dir = DATA_DIR / "test_soundscapes"
    fallback_dir = DATA_DIR / "train_soundscapes"
    predictions: dict[str, np.ndarray] = {}
    for audio_id, rows in tqdm(grouped.items(), desc="soundscapes"):
        path = test_dir / f"{audio_id}.ogg"
        if not path.exists():
            path = fallback_dir / f"{audio_id}.ogg"
        if not path.exists():
            for row_id, _ in rows:
                predictions[row_id] = np.zeros(len(species), dtype=np.float32)
            continue

        audio = load_audio(path, sample_rate=SAMPLE_RATE)
        row_ids = [row_id for row_id, _ in rows]
        end_seconds = [end_s for _, end_s in rows]
        clips = build_audio_clips(audio, end_seconds, SAMPLE_RATE)
        probs = predict_clips(models, extractor, clips, device=device, batch_size=BATCH_SIZE, tta=TTA)
        for row_id, prob in zip(row_ids, probs):
            predictions[row_id] = prob

    pred_mat = np.vstack([predictions.get(str(row_id), np.zeros(len(species), dtype=np.float32)) for row_id in sample["row_id"]])
    pred_df = pd.DataFrame(pred_mat, columns=species)
    sub = pd.concat([sample[["row_id"]].copy(), pred_df], axis=1)
    sub = sub[sample.columns]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} shape={sub.shape}")
    print(sub.head())


if __name__ == "__main__":
    main()
