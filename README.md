# BirdCLEF+ 2026 Kaggle Pipeline

This project is a stronger BirdCLEF+ 2026 training and submission pipeline. It now follows the public high-score pattern more closely: a solid PyTorch anchor model, PCEN/log-mel audio features, event-aware pooling, in-domain soundscape training, taxonomy smoothing, temporal smoothing, and optional Perch/BirdNET-style sidecar CSV blending.

## What Changed

- Audio frontend supports `logmel`, `pcen`, and `logmel_pcen`.
- `logmel_pcen` uses two input channels, inspired by PCEN/ConvNeXt sidecar notebooks.
- Model pooling now supports `attn`, which combines attention-weighted features with peak features for short acoustic events.
- Training supports rare-class `--balanced-sampler`.
- `infer.py` reads `spec_mode` and `in_chans` from checkpoints automatically.
- Inference supports sidecar submission CSVs, rank-space blending, taxonomy smoothing, and temporal smoothing.
- `kaggle_submission_standalone.py` was rebuilt so Kaggle CPU submission uses the same inference logic.

## Setup

```bash
cd birdclef2026
python scripts/download_data.py --out data/birdclef-2026 --copy
python scripts/prepare_metadata.py --data-dir data/birdclef-2026 --out-dir data/processed
python scripts/check_gpu.py
```

## Download Model Assets

For the strongest current anchor, download the ConvNeXt-Base timm backbone and the Google Perch v2 CPU Kaggle model:

```bash
python scripts/download_model.py --preset anchor_v2_strong
```

This creates:

- `models/pretrained/convnext_base.fb_in22k_ft_in1k.pth`
- `models/kaggle/...perch_v2_cpu...`
- `models/model_assets.json`

Perch is a TensorFlow/Kaggle model. Use it as a teacher, embedding model, or sidecar CSV source; do not pass it to `train.py --pretrained-path`.

## Strong Anchor Training

Start with one fold and validate the pipeline:

```bash
python train.py --data-dir data/birdclef-2026 --meta-dir data/processed --out-dir outputs/anchor_v2_convnext_base_fold0 --model convnext_base.fb_in22k_ft_in1k --pretrained-path models/pretrained/convnext_base.fb_in22k_ft_in1k.pth --epochs 20 --fold 0 --batch-size 4 --grad-accum 4 --duration 10 --channels-last --include-soundscapes --spec-mode logmel_pcen --pooling attn --head-hidden 768 --drop-path 0.2 --balanced-sampler --mixup-alpha 0.3 --mixup-p 0.5 --spec-augment-p 0.5 --scheduler cosine --lr 1e-4 --weight-decay 1e-4
```

If VRAM is tight, use EfficientNet-B0 and shorter clips:

```bash
python train.py --data-dir data/birdclef-2026 --meta-dir data/processed --out-dir outputs/anchor_v2_small_fold0 --model tf_efficientnet_b0_ns --epochs 10 --fold 0 --batch-size 8 --grad-accum 2 --duration 8 --channels-last --include-soundscapes --spec-mode logmel_pcen --pooling attn --head-hidden 256 --balanced-sampler
```

Train 2-5 folds only after a single fold produces a valid submission and fits the CPU budget.

## Local Inference

Plain anchor inference:

```bash
python infer.py --data-dir data/birdclef-2026 --checkpoint outputs/anchor_v2_fold0/fold0_best.pt --out submission.csv --tax-genus-alpha 0.15 --tax-class-alpha 0.05 --temporal-smooth-alpha 0.15
```

With a Perch/BirdNET/other sidecar CSV:

```bash
python infer.py --data-dir data/birdclef-2026 --checkpoint outputs/anchor_v2_fold0/fold0_best.pt --out submission.csv --sidecar-csv subm_birdnet_v24.csv --sidecar-weight 0.03 --sidecar-rank-blend --sidecar-topk 48 --sidecar-budget 0.006 --tax-genus-alpha 0.15 --tax-class-alpha 0.05 --temporal-smooth-alpha 0.15
```

The sidecar CSV must have `row_id` plus the same species columns as `sample_submission.csv`.

## Kaggle Submission

Upload these files to a Kaggle Dataset:

- `kaggle_submission_standalone.py`
- `fold*_best.pt`
- optional sidecar CSVs if you generated them

In the Kaggle notebook, attach the competition data and your weights dataset, turn internet off, use CPU, then run:

```python
!python /kaggle/input/birdclef2026-weights/kaggle_submission_standalone.py
```

Edit the constants at the top of `kaggle_submission_standalone.py` if the checkpoint paths or sidecar paths need to be fixed manually.
