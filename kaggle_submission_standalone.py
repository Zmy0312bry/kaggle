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


DATA_DIR = Path("/kaggle/input/birdclef-2026")
OUT_PATH = Path("/kaggle/working/submission.csv")
CHECKPOINT_PATHS: list[str] = []
SIDECAR_CSV_PATHS: list[str] = []
SIDECAR_WEIGHTS: list[float] = []

SAMPLE_RATE = 32000
WINDOW_SECONDS = 5
BATCH_SIZE = 24
TTA = 0
NUM_THREADS = max(1, min(4, os.cpu_count() or 2))

SIDECAR_RANK_BLEND = True
SIDECAR_TOPK = 48
SIDECAR_BUDGET = 0.006
TAX_GENUS_ALPHA = 0.15
TAX_CLASS_ALPHA = 0.05
TEMPORAL_SMOOTH_ALPHA = 0.15


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
        return torch.cat([logmel, pcen], dim=1)

    @staticmethod
    def _pcen(
        power: torch.Tensor,
        gain: float = 0.98,
        bias: float = 2.0,
        power_exp: float = 0.5,
        smooth: float = 0.025,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        smoother = torch.zeros_like(power)
        smoother[..., 0] = power[..., 0]
        for t in range(1, power.shape[-1]):
            smoother[..., t] = (1.0 - smooth) * smoother[..., t - 1] + smooth * power[..., t]
        pcen = (power / (eps + smoother).pow(gain) + bias).pow(power_exp) - bias**power_exp
        pcen_min = pcen.amin(dim=(-2, -1), keepdim=True)
        pcen_max = pcen.amax(dim=(-2, -1), keepdim=True)
        return ((pcen - pcen_min) / (pcen_max - pcen_min + eps)).clamp(0.0, 1.0)


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
        if pooling not in {"avg", "max", "avgmax", "gem", "attn"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.gem = GeM()

    @property
    def feature_multiplier(self) -> int:
        return 2 if self.pooling in {"avgmax", "attn"} else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return x
        if self.pooling == "gem":
            return self.gem(x)
        if x.ndim == 4:
            avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
            if self.pooling == "attn":
                b, _, h, w = x.shape
                attn = torch.softmax(x.mean(dim=1).flatten(1), dim=1).view(b, 1, h, w)
                weighted = (x * attn).sum(dim=(-2, -1))
                peak = F.adaptive_max_pool2d(x, 1).flatten(1)
                return torch.cat([weighted, peak], dim=1)
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
        in_chans: int = 1,
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
                in_chans=in_chans,
                num_classes=0,
                global_pool="",
                **model_kwargs,
            )
        except TypeError:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
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
            self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(head_in, num_classes))

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


def load_models(checkpoint_paths: Iterable[Path], num_classes: int, device: torch.device) -> tuple[list[nn.Module], str]:
    models = []
    spec_modes = []
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model_name = ckpt.get("model_name", "tf_efficientnet_b0_ns") if isinstance(ckpt, dict) else "tf_efficientnet_b0_ns"
        spec_mode = str(ckpt.get("spec_mode", "logmel")) if isinstance(ckpt, dict) else "logmel"
        in_chans = int(ckpt.get("in_chans", 2 if spec_mode == "logmel_pcen" else 1)) if isinstance(ckpt, dict) else 1
        model = BirdCLEFModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=float(ckpt.get("dropout", 0.2)) if isinstance(ckpt, dict) else 0.2,
            pooling=str(ckpt.get("pooling", "avg")) if isinstance(ckpt, dict) else "avg",
            head_hidden=int(ckpt.get("head_hidden", 0)) if isinstance(ckpt, dict) else 0,
            drop_path_rate=float(ckpt.get("drop_path", 0.0)) if isinstance(ckpt, dict) else 0.0,
            in_chans=in_chans,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
        spec_modes.append(spec_mode)
        print(f"Loaded {ckpt_path.name}: model={model_name} spec_mode={spec_mode} pooling={model.pool.pooling}")
    if not models:
        raise FileNotFoundError("No checkpoint found. Set CHECKPOINT_PATHS near the top of this cell.")
    if len(set(spec_modes)) > 1:
        raise ValueError(f"Checkpoint ensemble uses mixed spec modes: {sorted(set(spec_modes))}")
    return models, spec_modes[0]


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


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0)
    ranks = np.empty_like(values, dtype=np.float32)
    scale = max(1, values.shape[0] - 1)
    for col in range(values.shape[1]):
        ranks[order[:, col], col] = np.arange(values.shape[0], dtype=np.float32) / scale
    return ranks


def blend_sidecar_csv(anchor: pd.DataFrame, sidecar_path: Path, species: list[str], weight: float) -> pd.DataFrame:
    side = pd.read_csv(sidecar_path).set_index("row_id").reindex(anchor["row_id"].astype(str)).reset_index()
    missing = [c for c in species if c not in side.columns]
    if missing:
        raise ValueError(f"{sidecar_path} missing species columns, first={missing[:5]}")
    base = anchor[species].to_numpy(dtype=np.float32)
    extra = side[species].fillna(0.0).to_numpy(dtype=np.float32)
    base_blend = percentile_rank(base) if SIDECAR_RANK_BLEND else base
    extra_blend = percentile_rank(extra) if SIDECAR_RANK_BLEND else extra
    mask = np.ones_like(base_blend, dtype=bool)
    if 0 < SIDECAR_TOPK < len(species):
        mask = np.zeros_like(base_blend, dtype=bool)
        rows = np.arange(base_blend.shape[0])[:, None]
        mask[rows, np.argpartition(base_blend, -SIDECAR_TOPK, axis=1)[:, -SIDECAR_TOPK:]] = True
        mask[rows, np.argpartition(extra_blend, -SIDECAR_TOPK, axis=1)[:, -SIDECAR_TOPK:]] = True
    delta = np.where(mask, extra_blend - base_blend, 0.0)
    movement = float(np.mean(np.abs(weight * delta)))
    if SIDECAR_BUDGET > 0 and movement > SIDECAR_BUDGET:
        weight *= SIDECAR_BUDGET / max(movement, 1e-8)
        print(f"Shrunk sidecar {sidecar_path.name}: movement={movement:.6f} new_weight={weight:.5f}")
    out = anchor.copy()
    out[species] = np.clip(base_blend + weight * delta, 0.0, 1.0)
    return out


def apply_taxonomy_smoothing(df: pd.DataFrame, taxonomy_path: Path, species: list[str]) -> pd.DataFrame:
    if TAX_GENUS_ALPHA <= 0 and TAX_CLASS_ALPHA <= 0:
        return df
    if not taxonomy_path.exists():
        print("taxonomy.csv not found; smoothing skipped.")
        return df
    taxonomy = pd.read_csv(taxonomy_path)
    if "primary_label" not in taxonomy.columns:
        return df
    taxonomy = taxonomy.set_index("primary_label").reindex(species).reset_index()
    genus_col = next((c for c in ["genus", "genus_name"] if c in taxonomy.columns), None)
    class_col = next((c for c in ["class", "class_name", "category", "taxon_class"] if c in taxonomy.columns), None)
    if genus_col is None and "scientific_name" in taxonomy.columns:
        taxonomy["__genus"] = taxonomy["scientific_name"].astype(str).str.split().str[0]
        genus_col = "__genus"
    values = df[species].to_numpy(dtype=np.float32)
    for col, alpha in [(genus_col, TAX_GENUS_ALPHA), (class_col, TAX_CLASS_ALPHA)]:
        if col is None or alpha <= 0:
            continue
        groups = taxonomy[col].fillna("").astype(str).to_numpy()
        shared = values.copy()
        for group in sorted(set(groups)):
            idx = np.where(groups == group)[0]
            if group and len(idx) > 1:
                shared[:, idx] = values[:, idx].mean(axis=1, keepdims=True)
        values = (1.0 - alpha) * values + alpha * shared
    out = df.copy()
    out[species] = np.clip(values, 0.0, 1.0)
    return out


def apply_temporal_smoothing(df: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    if TEMPORAL_SMOOTH_ALPHA <= 0:
        return df
    out = df.copy()
    values = out[species].to_numpy(dtype=np.float32)
    audio_ids = out["row_id"].astype(str).map(lambda x: x.rsplit("_", 1)[0])
    for _, idx in out.groupby(audio_ids, sort=False).groups.items():
        order = list(idx)
        if len(order) <= 1:
            continue
        series = values[order].copy()
        smooth = series.copy()
        for i in range(1, len(order)):
            smooth[i] = (1.0 - TEMPORAL_SMOOTH_ALPHA) * series[i] + TEMPORAL_SMOOTH_ALPHA * smooth[i - 1]
        values[order] = smooth
    out[species] = np.clip(values, 0.0, 1.0)
    return out


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
    models, spec_mode = load_models(checkpoints, num_classes=len(species), device=device)
    extractor = LogMelExtractor(sample_rate=SAMPLE_RATE, mode=spec_mode).to(device).eval()

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
    sub = pd.concat([sample[["row_id"]].copy(), pd.DataFrame(pred_mat, columns=species)], axis=1)
    sub = sub[sample.columns]
    for i, sidecar in enumerate(SIDECAR_CSV_PATHS):
        weight = SIDECAR_WEIGHTS[i] if i < len(SIDECAR_WEIGHTS) else 0.03
        sub = blend_sidecar_csv(sub, Path(sidecar), species=species, weight=weight)
    sub = apply_taxonomy_smoothing(sub, DATA_DIR / "taxonomy.csv", species)
    sub = apply_temporal_smoothing(sub, species)
    sub = sub[sample.columns]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} shape={sub.shape}")
    print(sub.head())


if __name__ == "__main__":
    main()
