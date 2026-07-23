"""Model definitions and checkpoint loading for water segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn


class BatchNormDoubleConv(nn.Module):
    """Conv3×3 → BatchNorm → ReLU, repeated twice."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GroupNormDoubleConv(nn.Module):
    """Conv3×3 → GroupNorm → ReLU, repeated twice.

    Used by ``PretrainedResNet18UNet`` (notebook ``03_deep_learning_resnet_encoder.ipynb``).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Custom 4-level U-Net (BatchNorm), from notebook ``03_deep_learning_unet.ipynb``.

    Needed to load the tracked checkpoint ``models/unet_water_v2_best.pt``.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 16) -> None:
        super().__init__()
        f = base_filters

        self.encoder1 = BatchNormDoubleConv(in_channels, f)
        self.encoder2 = BatchNormDoubleConv(f, f * 2)
        self.encoder3 = BatchNormDoubleConv(f * 2, f * 4)
        self.encoder4 = BatchNormDoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = BatchNormDoubleConv(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.decoder4 = BatchNormDoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.decoder3 = BatchNormDoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.decoder2 = BatchNormDoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.decoder1 = BatchNormDoubleConv(f * 2, f)

        self.output = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))

        x = self.bottleneck(self.pool(skip4))

        x = self.up4(x)
        x = self.decoder4(torch.cat((x, skip4), dim=1))
        x = self.up3(x)
        x = self.decoder3(torch.cat((x, skip3), dim=1))
        x = self.up2(x)
        x = self.decoder2(torch.cat((x, skip2), dim=1))
        x = self.up1(x)
        x = self.decoder1(torch.cat((x, skip1), dim=1))

        return self.output(x)


# ---- ResNet18 backbone helpers (from notebook 03_deep_learning_resnet_encoder) ----


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder without the final pooling / FC layers."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        layers.extend(BasicBlock(self.in_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        skip0 = self.relu(self.bn1(self.conv1(x)))
        skip1 = self.layer1(self.maxpool(skip0))
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        x = self.layer4(skip3)
        return skip0, skip1, skip2, skip3, x


class UpBlock(nn.Module):
    """Transposed conv → concat skip → GroupNorm DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = GroupNormDoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((self.up(x), skip), dim=1))


class PretrainedResNet18UNet(nn.Module):
    """U-Net with a custom ResNet18 encoder and GroupNorm decoder.

    From notebook ``03_deep_learning_resnet_encoder.ipynb``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder()

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            GroupNormDoubleConv(32, 32),
        )
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0, skip1, skip2, skip3, x = self.encoder(x)
        x = self.up4(x, skip3)
        x = self.up3(x, skip2)
        x = self.up2(x, skip1)
        x = self.up1(x, skip0)
        x = self.up0(x)
        return self.output(x)


Architecture = Literal["unet", "unetplusplus"]


def build_smp_model(architecture: Architecture, encoder_name: str = "resnet18", pretrained: bool = False) -> nn.Module:
    """Build a segmentation model from ``segmentation-models-pytorch``.

    Parameters
    ----------
    architecture : ``"unet"`` or ``"unetplusplus"``
    encoder_name : str
        Any SMP-compatible encoder name (e.g. ``"resnet18"``, ``"mobilenet_v2"``).
    pretrained : bool
        Whether to load pretrained encoder weights (ImageNet).

    Returns
    -------
    nn.Module
        A model returning logits (``activation=None``).
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch is required. Install with: uv sync"
        ) from exc

    model_class = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}.get(architecture)
    if model_class is None:
        msg = f"Unknown SMP architecture: {architecture}. Use 'unet' or 'unetplusplus'."
        raise ValueError(msg)

    return model_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def _detect_format(state_dict: dict[str, torch.Tensor]) -> Literal["smp", "custom_unet", "resnet_unet"]:
    """Detect the architecture that produced *state_dict*.

    * ``"custom_unet"`` — keys like ``encoder1.block.0.weight``
    * ``"resnet_unet"`` — keys like ``encoder.conv1.weight`` and ``up4.up.weight``
    * ``"smp"`` — keys like ``encoder.conv1.weight`` and ``decoder.blocks...``
    """
    for key in state_dict:
        if key.startswith("encoder1."):
            return "custom_unet"
    encoder_seen = any(k.startswith("encoder.") for k in state_dict)
    up_seen = any(k.startswith(("up", "down")) for k in state_dict)
    if encoder_seen and up_seen:
        return "resnet_unet"
    if encoder_seen:
        return "smp"
    raise ValueError("Cannot detect model architecture from state dict keys.")


def _extract_image_size(cfg: Any) -> tuple[int, int]:
    if isinstance(cfg, dict):
        raw = cfg.get("image_size", (512, 512))
    else:
        raw = (512, 512)
    if isinstance(raw, list):
        raw = tuple(raw)
    return raw


def _extract_threshold(cfg: Any) -> float:
    if isinstance(cfg, dict):
        return float(cfg.get("threshold", 0.5))
    return 0.5


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and return model, config, and metadata.

    Supports three formats:

    * **SMP with architecture tag** (``modal_notebooks/03_deep-learning-unet.ipynb``) —
      top-level ``"architecture"`` and ``"state_dict"`` keys.

    * **SMP without tag** (``notebooks/03_deep_learning_unet_resnet_encoder.ipynb``) —
      top-level ``"model_state_dict"`` but weights begin with ``"encoder."``.

    * **Custom U-Net** (``notebooks/03_deep_learning_unet.ipynb``) —
      top-level ``"model_state_dict"`` and weights begin with ``"encoder1."``.

    Parameters
    ----------
    path
        Path to the ``.pt`` checkpoint file.
    map_location
        Device to load tensors onto (e.g. ``"cpu"``, ``"cuda:0"``).

    Returns
    -------
    dict with keys:
        - ``model``: loaded ``nn.Module`` in eval mode
        - ``config``: training / architecture config dict
        - ``threshold``: prediction threshold from checkpoint (default 0.5)
        - ``image_size``: ``(width, height)`` tuple expected by the model
    """
    path = Path(path)
    if map_location is None:
        map_location = "cpu"

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # --- 1. SMP checkpoint with architecture tag ---
    if "architecture" in checkpoint and "state_dict" in checkpoint:
        arch = checkpoint["architecture"]
        encoder_name = checkpoint.get("encoder_name", "resnet18")
        cfg = checkpoint.get("config", {})
        model = build_smp_model(arch, encoder_name=encoder_name, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return {
            "model": model,
            "config": cfg,
            "threshold": _extract_threshold(cfg),
            "image_size": _extract_image_size(cfg),
        }

    # --- 2. Checkpoints with model_state_dict ---
    if "model_state_dict" in checkpoint:
        cfg = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
        fmt = _detect_format(state_dict)

        if fmt == "smp":
            model = build_smp_model("unet", encoder_name="resnet18", pretrained=False)
            model.load_state_dict(state_dict)
            model.eval()
            return {
                "model": model,
                "config": cfg,
                "threshold": _extract_threshold(cfg),
                "image_size": _extract_image_size(cfg),
            }

        if fmt == "resnet_unet":
            model = PretrainedResNet18UNet()
            model.load_state_dict(state_dict)
            model.eval()
            return {
                "model": model,
                "config": cfg,
                "threshold": _extract_threshold(cfg),
                "image_size": _extract_image_size(cfg),
            }

        if fmt == "custom_unet":
            base_filters = int(cfg.get("base_filters", 16)) if isinstance(cfg, dict) else 16
            model = UNet(base_filters=base_filters)
            model.load_state_dict(state_dict)
            model.eval()
            return {
                "model": model,
                "config": cfg,
                "threshold": _extract_threshold(cfg),
                "image_size": _extract_image_size(cfg),
            }

    raise ValueError(
        "Unrecognised checkpoint format. Expected keys 'architecture'/'state_dict' "
        "(SMP) or 'model_state_dict'/'config' (any notebook)."
    )
