from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from birdclef2026.dataset import BirdCLEFDataset
from birdclef2026.losses import AsymmetricLoss
from birdclef2026.model import BirdCLEFModel
from birdclef2026.utils import load_json


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or p <= 0 or x.size(0) < 2 or np.random.random() > p:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    return x * lam + x[index] * (1.0 - lam), y * lam + y[index] * (1.0 - lam)


def evaluate(model, loader, device, use_amp: bool = False, target_threshold: float = 0.0):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="valid", leave=False):
            x = x.to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(x)
            preds.append(torch.sigmoid(logits).cpu())
            targets.append(y.cpu())
    pred = torch.cat(preds).numpy()
    target = (torch.cat(targets).numpy() > target_threshold).astype("int32")
    aucs = []
    for i in range(target.shape[1]):
        if target[:, i].min() != target[:, i].max():
            aucs.append(roc_auc_score(target[:, i], pred[:, i]))
    return float(sum(aucs) / max(1, len(aucs)))


def build_balanced_sampler(df: pd.DataFrame, target_cols: list[str]) -> WeightedRandomSampler:
    targets = df[target_cols].to_numpy(dtype=np.float32)
    positives = np.maximum(targets.sum(axis=0), 1.0)
    class_weight = 1.0 / positives
    sample_weight = (targets * class_weight).sum(axis=1)
    sample_weight = np.maximum(sample_weight, np.percentile(sample_weight[sample_weight > 0], 25) if np.any(sample_weight > 0) else 1.0)
    sample_weight = sample_weight / sample_weight.mean()
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/exp001"))
    parser.add_argument("--model", default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--pretrained-path", type=Path, default=None,
                        help="本地预训练 backbone 权重路径（避免从网络下载）")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps for small VRAM GPUs.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--spec-mode", choices=["logmel", "pcen", "logmel_pcen"], default="logmel",
                        help="Audio frontend. logmel_pcen creates a two-channel input inspired by PCEN sidecars.")
    parser.add_argument("--loss", choices=["bce", "asymmetric"], default="asymmetric")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", choices=["avg", "max", "avgmax", "gem", "attn"], default="attn")
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--mixup-p", type=float, default=0.5)
    parser.add_argument("--spec-augment-p", type=float, default=0.5)
    parser.add_argument("--time-mask-width", type=int, default=48)
    parser.add_argument("--freq-mask-width", type=int, default=16)
    parser.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--eval-target-threshold", type=float, default=0.0, help="Binarize soft labels before ROC-AUC.")
    parser.add_argument("--include-soundscapes", action="store_true", help="Add labeled 5-second soundscape rows to the training split.")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Sample rare positive classes more often. Useful for macro-AUC on long-tail taxa.")
    parser.add_argument("--use-precomputed", action="store_true",
                        help="使用预计算频谱 .npy 缓存（需先运行 scripts/precompute_spectrograms.py）")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch_factor，每个 worker 预取 batch 数")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last tensors on CUDA for speed/memory.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch>=2.0) for faster forward/backward.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    species = load_json(args.meta_dir / "species_list.json")

    # 预计算模式：自动使用带 spec_path 的 manifest
    if args.use_precomputed:
        precomputed_manifest = args.meta_dir / "train_manifest_precomputed.csv"
        if precomputed_manifest.exists():
            manifest = pd.read_csv(precomputed_manifest)
            print(f"Using precomputed manifest: {precomputed_manifest}")
        else:
            manifest = pd.read_csv(args.meta_dir / "train_manifest.csv")
            print("Warning: --use-precomputed but precomputed manifest not found, using original.")
    else:
        manifest = pd.read_csv(args.meta_dir / "train_manifest.csv")
    train_df = manifest[manifest["fold"] != args.fold].reset_index(drop=True)
    valid_df = manifest[manifest["fold"] == args.fold].reset_index(drop=True)
    soundscape_manifest = args.meta_dir / "soundscape_manifest.csv"
    if args.use_precomputed:
        precomputed_soundscape = args.meta_dir / "soundscape_manifest_precomputed.csv"
        if precomputed_soundscape.exists():
            soundscape_manifest = precomputed_soundscape
    if args.include_soundscapes and soundscape_manifest.exists():
        sound_df = pd.read_csv(soundscape_manifest)
        sound_df["num_samples"] = 5 * 32000
        train_df = pd.concat([train_df, sound_df], ignore_index=True)
        print(f"Added labeled soundscape rows to training: {len(sound_df)}")

    train_ds = BirdCLEFDataset(
        train_df,
        duration=args.duration,
        train=True,
        use_precomputed=args.use_precomputed,
        spec_mode=args.spec_mode,
        spec_augment_p=args.spec_augment_p,
        time_mask_width=args.time_mask_width,
        freq_mask_width=args.freq_mask_width,
    )
    valid_ds = BirdCLEFDataset(
        valid_df,
        duration=5.0,
        train=False,
        use_precomputed=args.use_precomputed,
        spec_mode=args.spec_mode,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    target_cols = [c for c in train_df.columns if c.startswith("target_")]
    sampler = build_balanced_sampler(train_df, target_cols) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = BirdCLEFModel(
        args.model,
        num_classes=len(species),
        pretrained=args.pretrained_path is None,
        pretrained_path=str(args.pretrained_path) if args.pretrained_path else None,
        dropout=args.dropout,
        pooling=args.pooling,
        head_hidden=args.head_hidden,
        drop_path_rate=args.drop_path,
        in_chans=2 if args.spec_mode == "logmel_pcen" else 1,
    ).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (reduce-overhead)")
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = AsymmetricLoss() if args.loss == "asymmetric" else torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_accum = max(1, args.grad_accum)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_optimizer_steps)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.min_lr)
    ema = None if args.no_ema else ModelEma(model, decay=args.ema_decay)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime visible to PyTorch: {torch.version.cuda}")
    else:
        print("CUDA is not available to PyTorch; training will run on CPU.")
    print(
        f"model={args.model} pooling={args.pooling} head_hidden={args.head_hidden} "
        f"batch_size={args.batch_size} grad_accum={grad_accum} effective_batch={args.batch_size * grad_accum} "
        f"amp={use_amp} mixup={args.mixup_alpha}/{args.mixup_p} specaug={args.spec_augment_p} "
        f"spec_mode={args.spec_mode} balanced_sampler={args.balanced_sampler} "
        f"scheduler={args.scheduler} ema={ema is not None}"
    )

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), start=1):
            x, y = x.to(device), y.to(device)
            if args.channels_last and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            x, y = mixup_batch(x, y, alpha=args.mixup_alpha, p=args.mixup_p)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
            total_loss += float(loss.detach().cpu()) * grad_accum

        eval_model = ema.module if ema is not None else model
        auc = evaluate(eval_model, valid_loader, device, use_amp=use_amp, target_threshold=args.eval_target_threshold)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch} train_loss={total_loss / max(1, len(train_loader)):.5f} val_macro_auc={auc:.5f} lr={current_lr:.2e}")
        ckpt = {
            "model": eval_model.state_dict(),
            "species": species,
            "model_name": args.model,
            "fold": args.fold,
            "auc": auc,
            "dropout": args.dropout,
            "pooling": args.pooling,
            "head_hidden": args.head_hidden,
            "drop_path": args.drop_path,
            "duration": args.duration,
            "spec_mode": args.spec_mode,
            "in_chans": 2 if args.spec_mode == "logmel_pcen" else 1,
        }
        torch.save(ckpt, args.out_dir / f"fold{args.fold}_last.pt")
        if auc > best_auc:
            best_auc = auc
            torch.save(ckpt, args.out_dir / f"fold{args.fold}_best.pt")


if __name__ == "__main__":
    main()
