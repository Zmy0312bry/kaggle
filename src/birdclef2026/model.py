from __future__ import annotations

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
        self.pooling = pooling
        self.gem = GeM()

    @property
    def feature_multiplier(self) -> int:
        return 2 if self.pooling == "avgmax" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return x
        if self.pooling == "gem":
            return self.gem(x)
        if x.ndim == 4:
            avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
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


class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        pooling: str = "avg",
        head_hidden: int = 0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use BirdCLEFModel: pip install timm") from exc
        model_kwargs = {}
        if drop_path_rate > 0:
            model_kwargs["drop_path_rate"] = drop_path_rate
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                global_pool="",
                **model_kwargs,
            )
        except TypeError:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                global_pool="",
            )
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
