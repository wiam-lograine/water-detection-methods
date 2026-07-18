import numpy as np
from typing import Union, Any


def binarize(mask: Union[list, np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities or grayscale masks to binary masks."""
    # 1. Convert input to a numpy array (if it isn't already)
    arr = np.asarray(mask)

    # 2. Safety check: Make sure the threshold is actually a valid percentage
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0.")
    
    # 3. The core logic
    return (arr >= threshold).astype(np.uint8)


def intersection_over_union(y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray], threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Compute IoU for binary segmentation.

    IoU = intersection / union. It is strict: false positives and false
    negatives both reduce the score. `eps=1e-7` avoids division by zero when
    both masks are empty, without changing the result in normal cases.
    """
    y_true = binarize(y_true, threshold=threshold).astype(bool)
    y_pred = binarize(y_pred, threshold=threshold).astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float((intersection + eps) / (union + eps))


def dice_coefficient(y_true: np.ndarray, y_pred:np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """Compute Dice coefficient for binary segmentation.

    Dice = 2 * intersection / (size_true + size_pred). It is often used in
    segmentation because it directly measures the overlap between two masks.
    `eps=1e-7` is only a numerical safety value for empty masks.
    """
    y_true = binarize(y_true, threshold=threshold).astype(bool)
    y_pred = binarize(y_pred, threshold=threshold).astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    return float((2 * intersection + eps) / (total + eps))


def pixel_accuracy(y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray], threshold: float = 0.5) -> float:
    """Compute pixel-wise accuracy.

    This metric counts the percentage of correctly classified pixels. It is
    easy to understand but can be misleading when water occupies a small part
    of the image, so it should be reported with IoU and Dice.
    """
    y_true = binarize(y_true, threshold=threshold)
    y_pred = binarize(y_pred, threshold=threshold)
    return float((y_true == y_pred).mean())
