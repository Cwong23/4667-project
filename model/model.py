import torch
import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet with its final layer replaced for `num_classes` outputs."""
    model = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze/unfreeze everything except the classifier head (`fc`)."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = not frozen


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, classes) from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt["classes"]
