# AI Handoff Guide

## Objective

Build and iterate a Kaggle BirdCLEF+ 2026 solution that produces a valid `submission.csv` for hidden soundscape test files.

## Do First

1. Read `docs/strategy.md`.
2. Download data with `scripts/download_data.py`, or on Kaggle use `/kaggle/input/birdclef-2026`.
3. Run `scripts/prepare_metadata.py` and inspect `data/processed/species_list.json`.
4. Train one fold and verify `infer.py` can create a CSV aligned to `sample_submission.csv`.

## Non-Negotiables

- This is multi-label. Use sigmoid/BCE-style training.
- Do not use softmax for final probabilities.
- Do not threshold predictions for submission.
- Align columns exactly to `sample_submission.csv`.
- The final Kaggle submission must run offline on CPU in 90 minutes.
- Labeled soundscapes are in-domain and should be used, not ignored.

## Likely Next Improvements

- Add a second training stage that mixes focal train clips with soundscape-labeled 5-second clips.
- Add pseudo-label generation for unlabeled `train_soundscapes`.
- Export the best fold(s) to ONNX and compare CPU speed against PyTorch.
- Train 5 folds and average logits.
- Add per-taxonomy monitoring; non-bird taxa are scarce and important for macro AUC.

## Common Failure Modes

- `submission.csv` has wrong columns: always reorder by `sample_submission.csv`.
- Predictions missing for hidden rows: parse row IDs as `{audio_id}_{end_second}` and use sample rows as truth.
- Notebook times out: reduce ensemble size, model size, TTA count, or spectrogram resolution.
- Local score looks good but LB is poor: random focal-audio CV does not capture soundscape domain shift.

