from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from birdclef2026.utils import multi_hot, parse_label_list, read_species_list, save_json


def find_audio_path(data_dir: Path, row: pd.Series) -> Path:
    candidates = []
    for col in ["filename", "file_name", "filepath", "path", "audio_path"]:
        if col in row and pd.notna(row[col]):
            candidates.append(str(row[col]))
    if "primary_label" in row and "filename" in row:
        candidates.append(f"{row['primary_label']}/{row['filename']}")
    for candidate in candidates:
        p = Path(candidate)
        if p.is_absolute() and p.exists():
            return p
        for base in [data_dir, data_dir / "train_audio"]:
            full = base / candidate
            if full.exists():
                return full
    if "primary_label" in row and "filename" in row:
        return data_dir / "train_audio" / str(row["primary_label"]) / str(row["filename"])
    raise ValueError(f"Cannot infer audio path for row: {row.to_dict()}")


def build_train_manifest(data_dir: Path, out_dir: Path, species: list[str], secondary_weight: float) -> pd.DataFrame:
    train_csv = data_dir / "train.csv"
    train = pd.read_csv(train_csv)
    species_to_idx = {s: i for i, s in enumerate(species)}
    rows = []
    for _, row in train.iterrows():
        labels = []
        if "primary_label" in row:
            labels.append(str(row["primary_label"]))
        target = multi_hot(labels, species_to_idx, 1.0)
        for col in ["secondary_labels", "secondary_label", "background_labels"]:
            if col in row:
                target = np.maximum(target, multi_hot(parse_label_list(row[col]), species_to_idx, secondary_weight))
        item = {
            "path": str(find_audio_path(data_dir, row)),
            "primary_label": str(row.get("primary_label", "")),
            "source": "train_audio",
        }
        for i, v in enumerate(target):
            item[f"target_{i}"] = float(v)
        rows.append(item)
    manifest = pd.DataFrame(rows)
    n_splits = 5
    manifest["fold"] = np.arange(len(manifest)) % n_splits
    y = manifest["primary_label"].fillna("missing").astype(str)
    counts = y.value_counts()
    common_mask = y.map(counts) >= n_splits
    if common_mask.sum() >= n_splits and y[common_mask].nunique() > 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        common_idx = manifest.index[common_mask].to_numpy()
        for fold, (_, val_pos) in enumerate(skf.split(manifest.loc[common_idx], y.loc[common_idx])):
            manifest.loc[common_idx[val_pos], "fold"] = fold
    manifest.to_csv(out_dir / "train_manifest.csv", index=False)
    return manifest


def build_soundscape_manifest(data_dir: Path, out_dir: Path, species: list[str]) -> pd.DataFrame | None:
    labels_path = data_dir / "train_soundscapes_labels.csv"
    sound_dir = data_dir / "train_soundscapes"
    if not labels_path.exists() or not sound_dir.exists():
        return None
    labels_df = pd.read_csv(labels_path)
    species_to_idx = {s: i for i, s in enumerate(species)}
    class_cols = [c for c in species if c in labels_df.columns]
    rows = []
    for _, row in labels_df.iterrows():
        if "row_id" in row and pd.notna(row["row_id"]):
            row_id = str(row["row_id"])
            parts = row_id.rsplit("_", 1)
            audio_id = parts[0]
            end_second = float(parts[1]) if len(parts) == 2 and parts[1].replace(".", "", 1).isdigit() else float(row.get("seconds", 5))
        else:
            audio_id = str(row.get("audio_id", row.get("filename", ""))).replace(".ogg", "")
            end_second = float(row.get("seconds", row.get("end_second", 5)))
            row_id = f"{audio_id}_{int(end_second)}"

        audio_path = sound_dir / f"{audio_id}.ogg"
        if class_cols:
            target = row[class_cols].astype(float).to_numpy(dtype=np.float32)
            full = np.zeros(len(species), dtype=np.float32)
            for col, val in zip(class_cols, target):
                full[species_to_idx[col]] = val
            target = full
        else:
            label_values = []
            for col in ["birds", "labels", "primary_label", "species"]:
                if col in row:
                    label_values.extend(parse_label_list(row[col]))
            target = multi_hot(label_values, species_to_idx, 1.0)

        item = {
            "path": str(audio_path),
            "row_id": row_id,
            "primary_label": "",
            "source": "train_soundscapes",
            "start_sample": int(max(0, end_second - 5) * 32000),
        }
        for i, v in enumerate(target):
            item[f"target_{i}"] = float(v)
        rows.append(item)
    manifest = pd.DataFrame(rows)
    manifest["fold"] = -1
    manifest.to_csv(out_dir / "soundscape_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--secondary-weight", type=float, default=0.35)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    species = read_species_list(args.data_dir)
    save_json(species, args.out_dir / "species_list.json")
    train_manifest = build_train_manifest(args.data_dir, args.out_dir, species, args.secondary_weight)
    sound_manifest = build_soundscape_manifest(args.data_dir, args.out_dir, species)

    print(f"Species: {len(species)}")
    print(f"Train manifest: {train_manifest.shape}")
    if sound_manifest is not None:
        print(f"Soundscape manifest: {sound_manifest.shape}")
    else:
        print("No soundscape labels found.")


if __name__ == "__main__":
    main()
