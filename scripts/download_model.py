from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "anchor_v2_strong": {
        "timm": ["convnext_base.fb_in22k_ft_in1k"],
        "kaggle": ["google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu"],
        "note": "Best default for the current logmel_pcen + attn anchor. Perch is downloaded as a sidecar/teacher asset.",
    },
    "anchor_v2_fast": {
        "timm": ["convnext_tiny.fb_in22k_ft_in1k"],
        "kaggle": ["google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu"],
        "note": "Faster anchor for smaller GPUs or faster CPU inference tests.",
    },
    "anchor_v2_small": {
        "timm": ["tf_efficientnet_b0_ns"],
        "kaggle": [],
        "note": "Small fallback anchor. Useful for smoke tests and low VRAM.",
    },
    "sidecars": {
        "timm": [],
        "kaggle": [
            "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu",
            "shadiakiki1/birdnet-analyzer/tflite/birdnet_global_6k_v2.4_model_fp32-1",
        ],
        "note": "External public bioacoustic models only. Use their outputs through sidecar CSV blending or distillation.",
    },
}


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def configure_hf(endpoint: str | None) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_timm_backbone(model_name: str, out_dir: Path) -> Path:
    import timm
    import torch

    print(f"[timm] downloading pretrained backbone: {model_name}")
    print(f"[timm] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<default>')}")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(model_name)}.pth"
    torch.save(model.state_dict(), out_path)
    size_mb = out_path.stat().st_size / (1024**2)
    print(f"[timm] saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def copy_asset(src: Path, dst: Path, overwrite: bool) -> Path:
    if dst.exists():
        if not overwrite:
            print(f"[kaggle] exists, skip copy: {dst}")
            return dst
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def kaggle_handle_variants(handle: str) -> list[str]:
    variants = [handle]
    if "/tensorflow2/" in handle:
        variants.append(handle.replace("/tensorflow2/", "/TensorFlow2/"))
    if "/TensorFlow2/" in handle:
        variants.append(handle.replace("/TensorFlow2/", "/tensorflow2/"))
    return list(dict.fromkeys(variants))


def download_kaggle_model(handle: str, out_dir: Path, copy_to_out: bool, overwrite: bool) -> Path:
    import kagglehub

    last_error: Exception | None = None
    downloaded: Path | None = None
    used_handle = handle
    for candidate in kaggle_handle_variants(handle):
        try:
            print(f"[kaggle] downloading model: {candidate}")
            downloaded = Path(kagglehub.model_download(candidate))
            used_handle = candidate
            break
        except Exception as exc:
            last_error = exc
            print(f"[kaggle] failed: {candidate}: {exc}")
    if downloaded is None:
        raise RuntimeError(f"Could not download Kaggle model {handle}") from last_error

    print(f"[kaggle] cache path: {downloaded}")
    if not copy_to_out:
        return downloaded
    dst = out_dir / safe_name(used_handle)
    copied = copy_asset(downloaded, dst, overwrite=overwrite)
    print(f"[kaggle] copied: {copied}")
    return copied


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model assets for the BirdCLEF+ 2026 logmel_pcen + attn pipeline."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="anchor_v2_strong")
    parser.add_argument("--model", action="append", default=[], help="Extra timm model name. Can be repeated.")
    parser.add_argument("--kaggle-model", action="append", default=[], help="Extra Kaggle model handle. Can be repeated.")
    parser.add_argument("--skip-preset-timm", action="store_true")
    parser.add_argument("--skip-preset-kaggle", action="store_true")
    parser.add_argument("--skip-kaggle", action="store_true", help="Only download timm/HF pretrained backbones.")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com", help="Set empty string to use default HuggingFace.")
    parser.add_argument("--out-dir", type=Path, default=Path("models/pretrained"), help="Where timm .pth files are saved.")
    parser.add_argument("--kaggle-out-dir", type=Path, default=Path("models/kaggle"), help="Where Kaggle models are copied.")
    parser.add_argument("--manifest", type=Path, default=Path("models/model_assets.json"))
    parser.add_argument("--no-copy-kaggle", action="store_true", help="Keep Kaggle models in kagglehub cache only.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        for name, cfg in PRESETS.items():
            print(f"{name}: {cfg['note']}")
            print(f"  timm: {cfg['timm']}")
            print(f"  kaggle: {cfg['kaggle']}")
        return

    configure_hf(args.hf_endpoint or None)
    preset = PRESETS[args.preset]
    timm_models = [] if args.skip_preset_timm else list(preset["timm"])
    timm_models.extend(args.model)
    kaggle_models = [] if args.skip_preset_kaggle else list(preset["kaggle"])
    kaggle_models.extend(args.kaggle_model)

    manifest: dict[str, Any] = {
        "preset": args.preset,
        "note": preset["note"],
        "timm": {},
        "kaggle": {},
        "recommended_train_command": None,
        "perch_usage": (
            "Perch v2 CPU is a TensorFlow/Kaggle model. Use it as a teacher, embedding extractor, "
            "or sidecar CSV source; it is not a PyTorch backbone checkpoint for train.py."
        ),
    }

    for model_name in timm_models:
        path = download_timm_backbone(model_name, args.out_dir)
        manifest["timm"][model_name] = str(path)

    if not args.skip_kaggle:
        for handle in kaggle_models:
            path = download_kaggle_model(
                handle,
                out_dir=args.kaggle_out_dir,
                copy_to_out=not args.no_copy_kaggle,
                overwrite=args.overwrite,
            )
            manifest["kaggle"][handle] = str(path)

    strong_path = manifest["timm"].get("convnext_base.fb_in22k_ft_in1k")
    if strong_path:
        manifest["recommended_train_command"] = (
            "python train.py --data-dir data/birdclef-2026 --meta-dir data/processed "
            "--out-dir outputs/anchor_v2_convnext_base_fold0 "
            "--model convnext_base.fb_in22k_ft_in1k "
            f"--pretrained-path {strong_path} "
            "--epochs 20 --fold 0 --batch-size 4 --grad-accum 4 --duration 10 "
            "--channels-last --include-soundscapes --spec-mode logmel_pcen --pooling attn "
            "--head-hidden 768 --drop-path 0.2 --balanced-sampler --mixup-alpha 0.3 "
            "--mixup-p 0.5 --spec-augment-p 0.5 --scheduler cosine --lr 1e-4 --weight-decay 1e-4"
        )

    write_manifest(args.manifest, manifest)


if __name__ == "__main__":
    main()
