import numpy as np


def binarize(mask, threshold=0.5):
    """Convert probabilities or grayscale masks to binary masks.

    `threshold=0.5` is the standard decision threshold for probability maps:
    pixels with probability >= 50% are considered water. For already-binary
    masks with values 0 and 1, this keeps the mask unchanged.
    """
    return (np.asarray(mask) >= threshold).astype(np.uint8)


def intersection_over_union(y_true, y_pred, threshold=0.5, eps=1e-7):
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


def dice_coefficient(y_true, y_pred, threshold=0.5, eps=1e-7):
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


def pixel_accuracy(y_true, y_pred, threshold=0.5):
    """Compute pixel-wise accuracy.

    This metric counts the percentage of correctly classified pixels. It is
    easy to understand but can be misleading when water occupies a small part
    of the image, so it should be reported with IoU and Dice.
    """
    y_true = binarize(y_true, threshold=threshold)
    y_pred = binarize(y_pred, threshold=threshold)
    return float((y_true == y_pred).mean())
