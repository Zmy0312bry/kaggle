# BirdCLEF+ 2026 Strategy Notes

## Competition Read

BirdCLEF+ 2026 asks us to identify wildlife species in Brazilian Pantanal soundscapes. The task is multi-label classification over 234 target columns. Test audio is split into 5-second windows; each row in `submission.csv` is a `row_id` plus 234 probabilities.

Important constraints and facts verified from public competition/report sources:

- Target: 234 species/sonotypes across birds, amphibians, insects, mammals, and reptiles.
- Metric: macro ROC-AUC, so ranking quality matters more than hard thresholds.
- Test: hidden `test_soundscapes`, about 600 one-minute OGG files, populated only during Kaggle notebook scoring.
- Runtime: offline Kaggle Code submission, CPU-only, 90 minutes.
- Training data: `train_audio` focal recordings plus `train_soundscapes` with expert labels in `train_soundscapes_labels.csv`.
- The labeled soundscapes are unusually valuable because they are in-domain Pantanal recordings, and some hidden-test species may only appear there rather than in `train_audio`.

Key public references used:

- Kaggle competition pages: `https://www.kaggle.com/competitions/birdclef-2026/overview` and `https://www.kaggle.com/competitions/birdclef-2026/data`.
- Public BirdCLEF+ 2026 analysis report: `https://storage.googleapis.com/kaggle-forum-message-attachments/3420357/39213/BirdCLEF2026_Analysis_Report.html`.
- DS@GT BirdCLEF+ 2025 paper: `https://arxiv.org/abs/2507.08236`.
- BirdCLEF 2024 pseudo-label solution notes: `https://github.com/jfpuget/birdclef-2024` and `https://github.com/TheoViel/kaggle_birdclef2024`.
- BirdCLEF pseudo-labeling paper: `https://ceur-ws.org/Vol-3740/paper-199.pdf`.

## Scoring Principles

Because the metric is macro ROC-AUC:

1. Do not turn probabilities into binary labels.
2. Prefer raw sigmoid outputs or averaged logits.
3. Avoid aggressive probability transforms that change class-wise ranking.
4. Use multi-label loss, not softmax cross entropy.
5. Validate on in-domain soundscape labels, not only random train-audio folds.

## Data Parsing Plan

Expected files:

- `train.csv`: metadata for short focal recordings, including `primary_label`, secondary labels, rating, latitude/longitude, and filename/path-like columns.
- `taxonomy.csv`: canonical class order for all target labels.
- `train_audio/`: OGG training clips.
- `train_soundscapes/`: in-domain soundscape recordings.
- `train_soundscapes_labels.csv`: expert labels for 5-second windows.
- `sample_submission.csv`: exact row and column order for submission.
- `test_soundscapes/`: hidden public/private test audio at scoring time.

The starter parser does this:

- Uses `taxonomy.csv` or `sample_submission.csv` to define `species_list`.
- Converts focal recordings to a manifest with primary target = 1 and secondary target = 0.35 by default.
- Converts soundscape rows into multi-hot labels from either class columns, a `birds`/`labels` string column, or `primary_label`.
- Creates stratified folds on primary labels for focal recordings, with a group fallback if metadata columns exist.
- Leaves soundscape labels available as a separate in-domain fine-tuning manifest.

## Baseline Model

The included model is a Kaggle-friendly mel-spectrogram image classifier:

- Audio sample rate: 32 kHz.
- Training window: default 10 seconds for focal clips.
- Inference window: exact 5 seconds.
- Spectrogram: 128 mel bins, 20-16,000 Hz, log compression.
- Backbone: `timm` EfficientNet/ConvNeXt/Swin-style 2D model with `in_chans=1`.
- Head: 234 sigmoid logits.
- Loss: BCEWithLogits or AsymmetricLoss.

Good starting configs:

```bash
python train.py --model tf_efficientnet_b0_ns --epochs 10 --batch-size 32
python train.py --model tf_efficientnetv2_s.in21k_ft_in1k --epochs 12 --batch-size 16 --loss asymmetric
```

## High-Score Roadmap

1. Strong baseline:
   Train 5 folds with `tf_efficientnet_b0_ns` or `tf_efficientnetv2_s.in21k_ft_in1k`. Submit a single fold first to validate the pipeline.

2. Use labeled soundscapes:
   Fine-tune the final epochs with a higher sampling weight for `train_soundscapes_labels.csv`. These are in-domain and should be treated as gold.

3. Reduce domain shift:
   Use mixup of 2-3 clips, background noise injection, random crop/roll, gain changes, SpecAugment, and low SNR augmentation. The goal is to make focal clips behave like passive soundscapes.

4. Handle rare non-bird classes:
   Oversample amphibians/insects/mammals/reptile classes, use class-balanced positive weights, and monitor class AUC by taxonomy group. Rare classes matter heavily in macro AUC.

5. Pseudo-label soundscapes:
   Train teacher models, predict unlabeled train soundscapes, keep confident soft labels, and train a student. Average logits, not thresholded labels.

6. Ensemble within CPU budget:
   Start with 2 folds, profile CPU inference, then add folds/backbones only if still under 90 minutes. Logit averaging is the default.

7. Export for speed:
   ONNX Runtime or TFLite can be 2-10x faster depending on model. Keep pure PyTorch as the reliable fallback.

## Kaggle Submission Workflow

1. On Kaggle, join the competition and accept the rules before the June 3, 2026 final deadline.
2. Train models locally or in Kaggle notebooks with internet/GPU enabled if allowed.
3. Save model weights and `species_list.json` as a Kaggle Dataset.
4. Create a new Kaggle Code notebook for submission.
5. Add inputs:
   - Competition data: `birdclef-2026`.
   - Your weights dataset.
6. Turn internet off for the final submission notebook.
7. Use CPU inference, generate `/kaggle/working/submission.csv`.
8. Click **Save Version** -> choose **Save & Run All**.
9. After the run finishes, open the notebook version page and click **Submit to Competition**.
10. Check that Kaggle shows no timeout and that the submission shape matches `sample_submission.csv`.

Never upload a CSV directly for this competition unless Kaggle explicitly enables it. Code competitions usually score the saved notebook version.
