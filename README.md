# BirdCLEF+ 2026 Kaggle Starter

This folder contains a practical starter pipeline for the Kaggle BirdCLEF+ 2026 competition.

The competition is a 5-second-window, multi-label bioacoustic classification task. The hidden test set is populated only when the Kaggle submission notebook runs, so the final notebook must work offline and write `/kaggle/working/submission.csv`.

## Files

- `docs/strategy.md` - competition reading notes, research-backed scoring strategy, and ablation roadmap.
- `docs/ai_handoff.md` - compact instructions for another AI/engineer to continue the work.
- `docs/中文运行与提交指南.md` - Chinese local GPU training and Kaggle submission guide.
- `scripts/download_data.py` - standalone data download script using `kagglehub`.
- `scripts/check_gpu.py` - quick PyTorch CUDA check for local training.
- `scripts/prepare_metadata.py` - parses the Kaggle CSV files into train/soundscape manifests and folds.
- `train.py` - trains a mel-spectrogram image model with multi-label BCE or asymmetric loss.
- `infer.py` - CPU-friendly Kaggle inference script that builds `submission.csv`.
- `src/birdclef2026/` - reusable dataset, audio, model, and utility code.

## Local Quick Start

```bash
cd birdclef2026
python scripts/download_data.py --out data/birdclef-2026 --copy
python scripts/prepare_metadata.py --data-dir data/birdclef-2026 --out-dir data/processed
python scripts/check_gpu.py
python train.py --data-dir data/birdclef-2026 --meta-dir data/processed --out-dir outputs/exp001 --epochs 8 --fold 0 --batch-size 8 --grad-accum 2 --duration 8 --channels-last
python infer.py --data-dir data/birdclef-2026 --checkpoint outputs/exp001/fold0_best.pt --out submission.csv
```

## Stronger Local Training Recipe

For an RTX 3060 Laptop GPU, start with EfficientNetV2-S plus in-domain soundscapes, mixup, SpecAugment, EMA, and cosine LR scheduling:

```bash
python train.py --data-dir data/birdclef-2026 --meta-dir data/processed --out-dir outputs/effv2s_fold0 --model tf_efficientnetv2_s.in21k_ft_in1k --epochs 12 --fold 0 --batch-size 8 --grad-accum 2 --duration 10 --channels-last --include-soundscapes --pooling gem --head-hidden 512 --drop-path 0.1 --mixup-alpha 0.3 --mixup-p 0.5 --spec-augment-p 0.5 --scheduler cosine
```

If memory is still comfortable, try `--batch-size 12 --grad-accum 1` or `--batch-size 16 --grad-accum 1`. Keep the final Kaggle inference model small enough to finish CPU scoring within 90 minutes; one EfficientNetV2-S fold is a reasonable first stronger submission.

On Kaggle, attach the competition dataset and your trained weights dataset, then run `infer.py` from a notebook or paste its cells into a Kaggle Code notebook.
