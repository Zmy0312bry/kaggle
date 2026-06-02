from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), kernel_size=x.shape[-2:]).pow(1.0 / self.p).flatten(1)
        if x.ndim == 3:
            return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return x


class FeaturePool(nn.Module):
    def __init__(self, pooling: str = "avg") -> None:
        super().__init__()
        if pooling not in {"avg", "max", "avgmax", "gem", "attn"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.gem = GeM()

    @property
    def feature_multiplier(self) -> int:
        return 2 if self.pooling in {"avgmax", "attn"} else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return x
        if self.pooling == "gem":
            return self.gem(x)
        if x.ndim == 4:
            avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
            if self.pooling == "attn":
                b, _, h, w = x.shape
                attn = torch.softmax(x.mean(dim=1).flatten(1), dim=1).view(b, 1, h, w)
                weighted = (x * attn).sum(dim=(-2, -1))
                peak = F.adaptive_max_pool2d(x, 1).flatten(1)
                return torch.cat([weighted, peak], dim=1)
            if self.pooling == "max":
                return F.adaptive_max_pool2d(x, 1).flatten(1)
            if self.pooling == "avgmax":
                return torch.cat([avg, F.adaptive_max_pool2d(x, 1).flatten(1)], dim=1)
            return avg
        avg = x.mean(dim=1)
        if self.pooling == "max":
            return x.max(dim=1).values
        if self.pooling == "avgmax":
            return torch.cat([avg, x.max(dim=1).values], dim=1)
        return avg


def _adapt_pretrained_state(state: dict, model: nn.Module) -> None:
    for key, value in list(state.items()):
        if value.ndim != 4 or value.shape[1] != 3:
            continue
        model_weight = model.state_dict().get(key)
        if model_weight is None or model_weight.ndim != 4 or model_weight.shape[1] not in {1, 2}:
            continue
        adapted = value.mean(dim=1, keepdim=True)
        if model_weight.shape[1] == 2:
            adapted = adapted.repeat(1, 2, 1, 1) / 2.0
        state[key] = adapted
        print(f"[pretrained_path] Adapted {key}: {list(value.shape)} -> {list(adapted.shape)}")


class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        dropout: float = 0.2,
        pooling: str = "avg",
        head_hidden: int = 0,
        drop_path_rate: float = 0.0,
        in_chans: int = 1,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use BirdCLEFModel: pip install timm") from exc

        if pretrained_path is not None:
            pretrained = False

        model_kwargs = {}
        if drop_path_rate > 0:
            model_kwargs["drop_path_rate"] = drop_path_rate
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="",
                **model_kwargs,
            )
        except TypeError:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="",
            )

        if pretrained_path is not None:
            pretrained_path = Path(pretrained_path)
            if not pretrained_path.exists():
                raise FileNotFoundError(f"Pretrained path not found: {pretrained_path}")
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            _adapt_pretrained_state(state, self.backbone)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"[pretrained_path] missing keys: {len(missing)}")
            if unexpected:
                print(f"[pretrained_path] unexpected keys: {len(unexpected)}")
            print(f"Loaded pretrained backbone from {pretrained_path}")

        self.pool = FeaturePool(pooling)
        head_in = self.backbone.num_features * self.pool.feature_multiplier
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(head_in, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(head_in, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.pool(features)
        return self.head(features)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {"model": state}
