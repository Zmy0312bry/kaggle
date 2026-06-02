"""
使用 HuggingFace 镜像下载 timm 预训练模型并保存到本地。

用法:
    # 默认模型
    python scripts/download_model.py

    # 指定模型
    python scripts/download_model.py --model tf_efficientnetv2_s.in21k_ft_in1k

    # 指定输出目录
    python scripts/download_model.py --out-dir models/pretrained

镜像源: HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

# ====== 镜像配置 ======
# 优先使用 hf-mirror.com，国内下载更快
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_and_save(model_name: str, out_dir: Path) -> Path:
    """下载预训练模型并保存 backbone state_dict 到本地"""
    import timm

    print(f"下载模型: {model_name}")
    print(f"镜像源: {os.environ.get('HF_ENDPOINT', '默认')}")

    # 创建模型（会自动从 HuggingFace Hub 下载）
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{model_name.replace('/', '__')}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"已保存到: {save_path}  ({save_path.stat().st_size / (1024**2):.1f} MB)")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="使用镜像下载 timm 预训练模型")
    parser.add_argument("--model", default="tf_efficientnetv2_s.in21k_ft_in1k",
                        help="timm 模型名称")
    parser.add_argument("--out-dir", type=Path, default=Path("models/pretrained"),
                        help="本地保存目录")
    args = parser.parse_args()

    download_and_save(args.model, args.out_dir)


if __name__ == "__main__":
    main()
