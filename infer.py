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


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0)
    ranks = np.empty_like(values, dtype=np.float32)
    scale = max(1, values.shape[0] - 1)
    for col in range(values.shape[1]):
        ranks[order[:, col], col] = np.arange(values.shape[0], dtype=np.float32) / scale
    return ranks


def blend_sidecar_csv(
    anchor: pd.DataFrame,
    sidecar_path: Path,
    species: list[str],
    weight: float,
    rank_space: bool,
    topk: int,
    budget: float,
) -> pd.DataFrame:
    side = pd.read_csv(sidecar_path)
    side = side.set_index("row_id").reindex(anchor["row_id"].astype(str)).reset_index()
    missing_cols = [c for c in species if c not in side.columns]
    if missing_cols:
        raise ValueError(f"{sidecar_path} is missing {len(missing_cols)} species columns, first={missing_cols[:5]}")

    base = anchor[species].to_numpy(dtype=np.float32)
    extra = side[species].fillna(0.0).to_numpy(dtype=np.float32)
    base_blend = percentile_rank(base) if rank_space else base
    extra_blend = percentile_rank(extra) if rank_space else extra
    mask = np.ones_like(base_blend, dtype=bool)
    if topk > 0 and topk < len(species):
        mask = np.zeros_like(base_blend, dtype=bool)
        base_top = np.argpartition(base_blend, -topk, axis=1)[:, -topk:]
        side_top = np.argpartition(extra_blend, -topk, axis=1)[:, -topk:]
        rows = np.arange(base_blend.shape[0])[:, None]
        mask[rows, base_top] = True
        mask[rows, side_top] = True
    delta = np.where(mask, extra_blend - base_blend, 0.0)
    movement = float(np.mean(np.abs(weight * delta)))
    if budget > 0 and movement > budget:
        weight *= budget / max(movement, 1e-8)
        print(f"Shrunk sidecar weight for {sidecar_path.name}: movement={movement:.6f} budget={budget:.6f} new_weight={weight:.5f}")
    blended = np.clip(base_blend + weight * delta, 0.0, 1.0)
    out = anchor.copy()
    out[species] = blended
    return out


def _taxonomy_group_columns(taxonomy: pd.DataFrame) -> tuple[str | None, str | None]:
    genus_col = next((c for c in ["genus", "genus_name"] if c in taxonomy.columns), None)
    class_col = next((c for c in ["class", "class_name", "category", "taxon_class"] if c in taxonomy.columns), None)
    return genus_col, class_col


def apply_taxonomy_smoothing(df: pd.DataFrame, taxonomy_path: Path, species: list[str], genus_alpha: float, class_alpha: float) -> pd.DataFrame:
    if genus_alpha <= 0 and class_alpha <= 0:
        return df
    if not taxonomy_path.exists():
        print(f"taxonomy.csv not found at {taxonomy_path}; taxonomy smoothing skipped.")
        return df
    taxonomy = pd.read_csv(taxonomy_path)
    if "primary_label" not in taxonomy.columns:
        return df
    taxonomy = taxonomy.set_index("primary_label").reindex(species).reset_index()
    genus_col, class_col = _taxonomy_group_columns(taxonomy)
    if genus_col is None and "scientific_name" in taxonomy.columns:
        taxonomy["__genus"] = taxonomy["scientific_name"].astype(str).str.split().str[0]
        genus_col = "__genus"

    values = df[species].to_numpy(dtype=np.float32)
    for col, alpha in [(genus_col, genus_alpha), (class_col, class_alpha)]:
        if col is None or alpha <= 0:
            continue
        smoothed = values.copy()
        groups = taxonomy[col].fillna("").astype(str).to_numpy()
        for group in sorted(set(groups)):
            idx = np.where(groups == group)[0]
            if not group or len(idx) <= 1:
                continue
            smoothed[:, idx] = values[:, idx].mean(axis=1, keepdims=True)
        values = (1.0 - alpha) * values + alpha * smoothed
    out = df.copy()
    out[species] = np.clip(values, 0.0, 1.0)
    return out


def apply_temporal_smoothing(df: pd.DataFrame, species: list[str], alpha: float) -> pd.DataFrame:
    if alpha <= 0:
        return df
    out = df.copy()
    values = out[species].to_numpy(dtype=np.float32)
    for _, idx in out.groupby(out["row_id"].astype(str).map(lambda x: x.rsplit("_", 1)[0]), sort=False).groups.items():
        order = list(idx)
        if len(order) <= 1:
            continue
        series = values[order].copy()
        smooth = series.copy()
        for i in range(1, len(order)):
            smooth[i] = (1.0 - alpha) * series[i] + alpha * smooth[i - 1]
        values[order] = smooth
    out[species] = np.clip(values, 0.0, 1.0)
    return out


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
    parser.add_argument("--sidecar-csv", type=Path, action="append", default=[],
                        help="Optional Perch/BirdNET/other submission CSV to blend after anchor inference.")
    parser.add_argument("--sidecar-weight", type=float, action="append", default=[],
                        help="Weight for each --sidecar-csv. Defaults to 0.03 for missing entries.")
    parser.add_argument("--sidecar-rank-blend", action="store_true",
                        help="Blend sidecars in class-wise percentile-rank space.")
    parser.add_argument("--sidecar-topk", type=int, default=48,
                        help="Only allow sidecars to move classes in anchor/sidecar top-k per row. Set 0 to disable.")
    parser.add_argument("--sidecar-budget", type=float, default=0.006,
                        help="Max mean absolute rank/probability movement per sidecar. Set 0 to disable.")
    parser.add_argument("--tax-genus-alpha", type=float, default=0.0)
    parser.add_argument("--tax-class-alpha", type=float, default=0.0)
    parser.add_argument("--temporal-smooth-alpha", type=float, default=0.0)
    args = parser.parse_args()

    sample_path = args.data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    species = read_species_list(args.data_dir, sample_path)
    device = torch.device("cpu")

    models = []
    spec_modes = []
    for ckpt_path in args.checkpoint:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = args.model or ckpt.get("model_name", "tf_efficientnet_b0_ns")
        spec_mode = str(ckpt.get("spec_mode", "logmel"))
        spec_modes.append(spec_mode)
        model = BirdCLEFModel(
            model_name,
            num_classes=len(species),
            pretrained=False,
            dropout=float(ckpt.get("dropout", 0.2)),
            pooling=str(ckpt.get("pooling", "avg")),
            head_hidden=int(ckpt.get("head_hidden", 0)),
            drop_path_rate=float(ckpt.get("drop_path", 0.0)),
            in_chans=int(ckpt.get("in_chans", 2 if spec_mode == "logmel_pcen" else 1)),
        ).to(device)
        load_checkpoint(model, str(ckpt_path), device)
        model.eval()
        models.append(model)

    if len(set(spec_modes)) > 1:
        raise ValueError(f"All checkpoints in one ensemble must use the same spec_mode, got {sorted(set(spec_modes))}")
    spec_mode = spec_modes[0] if spec_modes else "logmel"
    extractor = LogMelExtractor(sample_rate=args.sample_rate, mode=spec_mode).to(device)
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
    for i, sidecar_path in enumerate(args.sidecar_csv):
        weight = args.sidecar_weight[i] if i < len(args.sidecar_weight) else 0.03
        sub = blend_sidecar_csv(
            sub,
            sidecar_path=sidecar_path,
            species=species,
            weight=weight,
            rank_space=args.sidecar_rank_blend,
            topk=args.sidecar_topk,
            budget=args.sidecar_budget,
        )
    sub = apply_taxonomy_smoothing(
        sub,
        taxonomy_path=args.data_dir / "taxonomy.csv",
        species=species,
        genus_alpha=args.tax_genus_alpha,
        class_alpha=args.tax_class_alpha,
    )
    sub = apply_temporal_smoothing(sub, species=species, alpha=args.temporal_smooth_alpha)
    sub = sub[sample.columns]
    sub.to_csv(args.out, index=False)
    print(f"Saved {args.out} shape={sub.shape}")


if __name__ == "__main__":
    main()
