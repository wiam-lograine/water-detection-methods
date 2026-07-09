import numpy as np


def binarize(mask, threshold=0.5):
    """Convert probabilities or grayscale masks to binary masks."""
    return (np.asarray(mask) >= threshold).astype(np.uint8)


def intersection_over_union(y_true, y_pred, threshold=0.5, eps=1e-7):
    """Compute IoU for binary segmentation."""
    y_true = binarize(y_true, threshold=threshold).astype(bool)
    y_pred = binarize(y_pred, threshold=threshold).astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float((intersection + eps) / (union + eps))


def dice_coefficient(y_true, y_pred, threshold=0.5, eps=1e-7):
    """Compute Dice coefficient for binary segmentation."""
    y_true = binarize(y_true, threshold=threshold).astype(bool)
    y_pred = binarize(y_pred, threshold=threshold).astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    return float((2 * intersection + eps) / (total + eps))


def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Compute pixel-wise accuracy."""
    y_true = binarize(y_true, threshold=threshold)
    y_pred = binarize(y_pred, threshold=threshold)
    return float((y_true == y_pred).mean())

