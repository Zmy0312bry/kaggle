from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from birdclef2026.dataset import BirdCLEFDataset
from birdclef2026.losses import AsymmetricLoss
from birdclef2026.model import BirdCLEFModel
from birdclef2026.utils import load_json


def evaluate(model, loader, device, target_threshold: float = 0.0):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="valid", leave=False):
            x = x.to(device)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--meta-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/exp001"))
    parser.add_argument("--model", default="tf_efficientnet_b0_ns")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps for small VRAM GPUs.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--loss", choices=["bce", "asymmetric"], default="asymmetric")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-target-threshold", type=float, default=0.0, help="Binarize soft labels before ROC-AUC.")
    parser.add_argument("--include-soundscapes", action="store_true", help="Add labeled 5-second soundscape rows to the training split.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last tensors on CUDA for speed/memory.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    species = load_json(args.meta_dir / "species_list.json")
    manifest = pd.read_csv(args.meta_dir / "train_manifest.csv")
    train_df = manifest[manifest["fold"] != args.fold].reset_index(drop=True)
    valid_df = manifest[manifest["fold"] == args.fold].reset_index(drop=True)
    soundscape_manifest = args.meta_dir / "soundscape_manifest.csv"
    if args.include_soundscapes and soundscape_manifest.exists():
        sound_df = pd.read_csv(soundscape_manifest)
        sound_df["num_samples"] = 5 * 32000
        train_df = pd.concat([train_df, sound_df], ignore_index=True)
        print(f"Added labeled soundscape rows to training: {len(sound_df)}")

    train_ds = BirdCLEFDataset(train_df, duration=args.duration, train=True)
    valid_ds = BirdCLEFDataset(valid_df, duration=5.0, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    model = BirdCLEFModel(args.model, num_classes=len(species), pretrained=True, dropout=args.dropout).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = AsymmetricLoss() if args.loss == "asymmetric" else torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, args.grad_accum)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime visible to PyTorch: {torch.version.cuda}")
    else:
        print("CUDA is not available to PyTorch; training will run on CPU.")
    print(f"batch_size={args.batch_size} grad_accum={grad_accum} effective_batch={args.batch_size * grad_accum} amp={use_amp}")

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), start=1):
            x, y = x.to(device), y.to(device)
            if args.channels_last and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * grad_accum

        auc = evaluate(model, valid_loader, device, target_threshold=args.eval_target_threshold)
        print(f"epoch={epoch} train_loss={total_loss / max(1, len(train_loader)):.5f} val_macro_auc={auc:.5f}")
        ckpt = {
            "model": model.state_dict(),
            "species": species,
            "model_name": args.model,
            "fold": args.fold,
            "auc": auc,
            "dropout": args.dropout,
        }
        torch.save(ckpt, args.out_dir / f"fold{args.fold}_last.pt")
        if auc > best_auc:
            best_auc = auc
            torch.save(ckpt, args.out_dir / f"fold{args.fold}_best.pt")


if __name__ == "__main__":
    main()
