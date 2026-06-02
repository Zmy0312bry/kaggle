# BirdCLEF+ 2026 Kaggle 提交指南

这份指南用于把本地训练好的模型提交到 Kaggle Code Competition。最终目标是在 Kaggle Notebook 里生成：

```text
/kaggle/working/submission.csv
```

## 1. 本地准备

先确认本地至少有一个训练好的权重，例如：

```text
D:\kaggle\birdclef2026\outputs\effv2s_fold0\fold0_best.pt
```

如果你还在用第一个 baseline，也可以是：

```text
D:\kaggle\birdclef2026\outputs\exp001\fold0_best.pt
```

同时确认这个单文件提交脚本存在：

```text
D:\kaggle\birdclef2026\kaggle_submission_standalone.py
```

这个脚本已经把 `audio.py`、`model.py`、`utils.py`、`infer.py` 的核心内容合并到一个文件里。Kaggle 上不需要再额外复制 `src/` 目录。

## 2. 创建 Kaggle Dataset

在 Kaggle 网页操作：

1. 点击右上角 `Create`
2. 选择 `New Dataset`
3. 上传以下文件：

```text
kaggle_submission_standalone.py
fold0_best.pt
```

如果你训练了多个 fold，可以一起上传：

```text
fold0_best.pt
fold1_best.pt
fold2_best.pt
...
```

Dataset 名字可以叫：

```text
birdclef2026-weights
```

## 3. 创建提交 Notebook

新建一个 Kaggle Notebook，然后添加两个 Input：

1. 官方比赛数据集：`birdclef-2026`
2. 你的权重数据集：`birdclef2026-weights`

Notebook 设置里建议：

```text
Accelerator: None
Internet: Off
```

比赛最终评分是 CPU 离线运行，所以提交前一定要用这个环境检查。

## 4. 推荐运行方式

如果你把 `kaggle_submission_standalone.py` 也上传到了 Dataset，在 Notebook 第一格运行：

```python
!python /kaggle/input/birdclef2026-weights/kaggle_submission_standalone.py
```

脚本会自动在 `/kaggle/input` 下面寻找：

```text
fold*_best.pt
```

如果找不到，才会退而寻找其他 `.pt`、`.pth`、`.ckpt` 文件。

## 5. 直接粘贴方式

如果你不想通过 `!python` 运行，也可以打开本地文件：

```text
D:\kaggle\birdclef2026\kaggle_submission_standalone.py
```

把里面全部代码复制到 Kaggle Notebook 的一个 cell 里，然后直接运行。

如果自动找权重找错了，就修改脚本顶部：

```python
CHECKPOINT_PATHS: list[str] = []
```

改成你的权重路径，例如：

```python
CHECKPOINT_PATHS = [
    "/kaggle/input/birdclef2026-weights/fold0_best.pt",
]
```

多个 fold ensemble 可以写：

```python
CHECKPOINT_PATHS = [
    "/kaggle/input/birdclef2026-weights/fold0_best.pt",
    "/kaggle/input/birdclef2026-weights/fold1_best.pt",
]
```

## 6. 检查提交文件

脚本运行完成后，在 Notebook 新开一格：

```python
import pandas as pd

sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
sub.head()
```

列数应该是：

```text
235
```

也就是：

```text
row_id + 234 个物种概率列
```

## 7. 正式提交

检查 `/kaggle/working/submission.csv` 生成成功后：

1. 点击右上角 `Save Version`
2. 选择 `Save & Run All`
3. 等 Kaggle 后台完整跑完
4. 进入生成的 Notebook Version 页面
5. 点击 `Submit to Competition`

不要直接上传本地 CSV。这个比赛是隐藏测试集，真实 `test_soundscapes` 只会在 Kaggle 评分时挂载到 Notebook 里。

## 8. 常见问题

### 找不到权重

报错类似：

```text
No checkpoint found
```

解决方法：检查 Dataset 路径，然后手动填写 `CHECKPOINT_PATHS`。

### 找不到 timm

报错类似：

```text
No module named 'timm'
```

说明 Kaggle 环境没有可用的 `timm`。解决方法是上传一个包含 `timm` wheel 的 Kaggle Dataset，或把 notebook internet 打开安装后再作为自定义镜像流程处理。正式提交时 internet 必须关闭。

### 运行超时

先把脚本顶部改小：

```python
BATCH_SIZE = 16
TTA = 0
```

如果用了多个 fold，先只提交一个 `fold0_best.pt`。EfficientNetV2-S 单 fold 是比较稳的第一版，多个大模型 ensemble 可能超过 CPU 90 分钟。

### 提交列不匹配

脚本最后会强制：

```python
sub = sub[sample.columns]
```

只要官方 `sample_submission.csv` 能读取，一般不会列错。不要手动改 species 列顺序。
