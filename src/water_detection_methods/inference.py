"""Inference pipeline for water segmentation models.

Usage
-----
    from water_detection_methods.inference import Predictor

    predictor = Predictor("path/to/checkpoint.pt")
    mask, overlay = predictor.predict("image.jpg", return_overlay=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import restore_mask_from_padding
from .model import load_checkpoint

# Normalisation ImageNet (identique au notebook 03)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def _letterbox_image(
    image: Image.Image,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Letterbox a PIL image to *target_size* ``(width, height)``.

    Returns the padded float32 array in ``[0, 1]`` and padding metadata.
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top

    canvas = Image.new("RGB", target_size, color=(0, 0, 0))
    canvas.paste(resized, (pad_left, pad_top))

    metadata = {
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "target_size": target_size,
    }
    return np.asarray(canvas, dtype=np.float32) / 255.0, metadata


def _preprocess(image_array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HWC float32 array to normalized CHW tensor on *device*."""
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().to(device)
    tensor = (tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return tensor.unsqueeze(0)  # add batch dimension


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (0.0, 0.45, 1.0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Superpose a binary mask on an RGB image."""
    result = image.copy()
    color_array = np.array(color, dtype=np.float32)
    binary = mask > 0
    result[binary] = (1.0 - alpha) * result[binary] + alpha * color_array
    return np.clip(result, 0.0, 1.0)


class Predictor:
    """Water segmentation predictor.

    Parameters
    ----------
    checkpoint_path
        Path to a ``.pt`` checkpoint (custom U-Net or SMP format).
    device
        Torch device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        loaded = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model = loaded["model"]
        self.config = loaded["config"]
        self.threshold = float(loaded.get("threshold", 0.5))
        self.image_size: tuple[int, int] = loaded.get("image_size", (512, 512))

        self.model.to(self.device)

    def predict(
        self,
        image_path: str | Path,
        return_overlay: bool = True,
        raw_logits: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Segment water in an image.

        Parameters
        ----------
        image_path
            Path to the input image.
        return_overlay
            If ``True``, return ``(mask, overlay)``. Otherwise return just the
            binary mask.
        raw_logits
            If ``True``, return the unthresholded probability map instead of a
            binary mask.

        Returns
        -------
        Binary ``uint8`` mask of shape ``(H, W)`` with values ``{0, 1}``, or
        ``(mask, overlay)`` where overlay is the mask projected on the image.
        """
        image = Image.open(image_path).convert("RGB")
        image_array, metadata = _letterbox_image(image, self.image_size)

        input_tensor = _preprocess(image_array, self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits.float())

        prob_map = probabilities.squeeze().cpu().numpy()
        padded_mask = (prob_map >= self.threshold).astype(np.uint8)

        mask = restore_mask_from_padding(padded_mask, metadata, threshold=0.5)

        if raw_logits:
            restored_prob = Image.fromarray(
                ((prob_map * 255).astype(np.uint8)), mode="L"
            ).resize(metadata["original_size"], Image.Resampling.NEAREST)
            return np.asarray(restored_prob, dtype=np.float32) / 255.0

        if not return_overlay:
            return mask

        # Original image for overlay
        original = np.asarray(image, dtype=np.float32) / 255.0
        overlay = _overlay(original, mask)
        return mask, overlay

    def predict_batch(
        self,
        image_paths: list[str | Path],
        return_overlay: bool = False,
    ) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
        """Segment multiple images."""
        return [self.predict(p, return_overlay=return_overlay) for p in image_paths]


def predict_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    threshold: float | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Convenience function: load checkpoint and predict one image.

    Returns a binary ``uint8`` mask.
    """
    predictor = Predictor(checkpoint_path, device=device)
    if threshold is not None:
        predictor.threshold = threshold
    return predictor.predict(image_path, return_overlay=False)  # type: ignore[return-value]
