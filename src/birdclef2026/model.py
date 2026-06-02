from __future__ import annotations

import torch
from torch import nn


class BirdCLEFModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use BirdCLEFModel: pip install timm") from exc
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return self.head(features)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {"model": state}
