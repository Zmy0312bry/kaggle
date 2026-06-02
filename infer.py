from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd()

for src_path in [
    ROOT / "src",
    ROOT / "birdclef2026" / "src",
    Path("/kaggle/working/birdclef2026/src"),
]:
    if src_path.exists():
        sys.path.append(str(src_path))
        break

from birdclef2026.audio import LogMelExtractor, crop_or_pad, load_audio
from birdclef2026.model import BirdCLEFModel, load_checkpoint
from birdclef2026.utils import read_species_list


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_row_id(row_id: str) -> tuple[str, int]:
    audio_id, end_s = row_id.rsplit("_", 1)
    return audio_id, int(float(end_s))


def predict_clip(model, extractor, audio, sample_rate, device, tta: int = 0) -> np.ndarray:
    variants = [audio]
    if tta >= 1:
        variants.append(audio[::-1].copy())
    if tta >= 2:
        variants.append(np.clip(audio * 1.15, -1.0, 1.0))
    logits = []
    with torch.no_grad():
        for variant in variants:
            waveform = torch.from_numpy(variant.astype(np.float32)).to(device)
            spec = extractor(waveform).unsqueeze(0).to(device)
            logits.append(model(spec).cpu().numpy()[0])
    return np.mean(logits, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/birdclef-2026"))
    parser.add_argument("--checkpoint", type=Path, required=True, action="append", help="One or more .pt checkpoints.")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--tta", type=int, default=0)
    args = parser.parse_args()

    sample_path = args.data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    species = read_species_list(args.data_dir, sample_path)
    device = torch.device("cpu")

    models = []
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = args.model or ckpt.get("model_name", "tf_efficientnet_b0_ns")
        model = BirdCLEFModel(model_name, num_classes=len(species), pretrained=False, dropout=float(ckpt.get("dropout", 0.2))).to(device)
        load_checkpoint(model, str(ckpt_path), device)
        model.eval()
        models.append(model)

    extractor = LogMelExtractor(sample_rate=args.sample_rate).to(device)
    extractor.eval()

    predictions: dict[str, np.ndarray] = {}
    grouped: dict[str, list[tuple[str, int]]] = {}
    for row_id in sample["row_id"].astype(str):
        audio_id, end_s = parse_row_id(row_id)
        grouped.setdefault(audio_id, []).append((row_id, end_s))

    test_dir = args.data_dir / "test_soundscapes"
    fallback_dir = args.data_dir / "train_soundscapes"
    for audio_id, rows in tqdm(grouped.items(), desc="soundscapes"):
        path = test_dir / f"{audio_id}.ogg"
        if not path.exists():
            path = fallback_dir / f"{audio_id}.ogg"
        if not path.exists():
            for row_id, _ in rows:
                predictions[row_id] = np.zeros(len(species), dtype=np.float32)
            continue

        audio = load_audio(path, sample_rate=args.sample_rate)
        for row_id, end_s in rows:
            start = max(0, int((end_s - 5) * args.sample_rate))
            clip = audio[start : start + 5 * args.sample_rate]
            clip = crop_or_pad(clip, 5 * args.sample_rate, random_crop=False)
            model_logits = [predict_clip(m, extractor, clip, args.sample_rate, device, args.tta) for m in models]
            predictions[row_id] = sigmoid(np.mean(model_logits, axis=0)).astype(np.float32)

    sub = sample[["row_id"]].copy()
    pred_mat = np.vstack([predictions.get(str(r), np.zeros(len(species), dtype=np.float32)) for r in sample["row_id"]])
    pred_df = pd.DataFrame(pred_mat, columns=species)
    sub = pd.concat([sub, pred_df], axis=1)
    sub = sub[sample.columns]
    sub.to_csv(args.out, index=False)
    print(f"Saved {args.out} shape={sub.shape}")


if __name__ == "__main__":
    main()
