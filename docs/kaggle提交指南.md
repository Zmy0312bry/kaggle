# BirdCLEF+ 2026 Kaggle 提交指南

最终目标是在 Kaggle Code Notebook 中生成：

```text
/kaggle/working/submission.csv
```

## 1. 要上传的文件

创建一个 Kaggle Dataset，例如 `birdclef2026-weights`，上传：

```text
kaggle_submission_standalone.py
fold0_best.pt
```

如果训练了多折，可以一起上传：

```text
fold0_best.pt
fold1_best.pt
fold2_best.pt
```

如果有 Perch、BirdNET 或其他公开模型生成的 sidecar submission，也可以上传：

```text
subm_birdnet_v24.csv
subm_perch.csv
```

如果你用 `scripts/download_model.py --preset anchor_v2_strong` 下载了 Google Perch v2 CPU，本地会有 `models/kaggle/...`。正式 Kaggle 提交时更推荐直接在 Notebook 的 Add Input 里添加 Kaggle Model：

```text
google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu
```

它不是 PyTorch checkpoint，不能替代 `fold*_best.pt`。它适合作为外部 teacher/sidecar 资产。

## 2. standalone 脚本包含什么

`kaggle_submission_standalone.py` 已经合并了新版核心逻辑：

- `logmel`、`pcen`、`logmel_pcen`
- 双通道 checkpoint 自动识别
- `attn` pooling
- 多 checkpoint logit ensemble
- sidecar CSV rank-space blending
- taxonomy smoothing
- temporal smoothing

所以 Kaggle 上不需要再复制 `src/` 目录。

## 3. 模型资产下载脚本

本地或服务器上可以先运行：

```bash
python scripts/download_model.py --preset anchor_v2_strong
```

这个 preset 会下载：

- `convnext_base.fb_in22k_ft_in1k` 的 timm 预训练 backbone
- Google Perch v2 CPU Kaggle Model

训练时使用的是 timm `.pth`：

```bash
python train.py --model convnext_base.fb_in22k_ft_in1k --pretrained-path models/pretrained/convnext_base.fb_in22k_ft_in1k.pth ...
```

Perch 的输出如果整理成 submission CSV，再通过 `--sidecar-csv` 或 standalone 顶部的 `SIDECAR_CSV_PATHS` 融合。

## 4. Kaggle Notebook 设置

新建 Notebook，Add Input：

- 官方比赛数据：`birdclef-2026`
- 你的权重 Dataset：例如 `birdclef2026-weights`

设置：

```text
Accelerator: None
Internet: Off
```

运行：

```python
!python /kaggle/input/birdclef2026-weights/kaggle_submission_standalone.py
```

## 5. 手动指定权重

如果自动搜索权重不符合预期，改脚本顶部：

```python
CHECKPOINT_PATHS = [
    "/kaggle/input/birdclef2026-weights/fold0_best.pt",
]
```

多折：

```python
CHECKPOINT_PATHS = [
    "/kaggle/input/birdclef2026-weights/fold0_best.pt",
    "/kaggle/input/birdclef2026-weights/fold1_best.pt",
]
```

注意：同一个 ensemble 里的 checkpoint 必须使用相同 `spec_mode`，例如都为 `logmel_pcen`。

## 6. 加 sidecar CSV

编辑脚本顶部：

```python
SIDECAR_CSV_PATHS = [
    "/kaggle/input/birdclef2026-weights/subm_birdnet_v24.csv",
]
SIDECAR_WEIGHTS = [0.03]
```

默认 sidecar 参数比较保守：

```python
SIDECAR_RANK_BLEND = True
SIDECAR_TOPK = 48
SIDECAR_BUDGET = 0.006
```

sidecar CSV 必须包含 `row_id` 和所有物种列，列名要与 `sample_submission.csv` 一致。

## 7. 后处理参数

脚本顶部默认：

```python
TAX_GENUS_ALPHA = 0.15
TAX_CLASS_ALPHA = 0.05
TEMPORAL_SMOOTH_ALPHA = 0.15
```

这些是弱修正。如果线上分数不稳定，可以先设为 `0.0` 做 anchor 对照，再逐个打开。

## 8. 检查 submission

运行完成后：

```python
import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
sub.head()
```

列数应为 `235`。脚本最后会强制按 `sample_submission.csv` 对齐列顺序。

## 9. 正式提交

点击：

```text
Save Version -> Save & Run All -> Submit to Competition
```

不要直接上传本地 CSV。这个比赛的隐藏 `test_soundscapes` 只会在 Kaggle 评分 Notebook 运行时挂载。
