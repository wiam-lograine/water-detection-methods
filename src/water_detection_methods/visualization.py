import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title=None, ax=None):
    """Display one RGB image.

    If an axis is provided, the function draws inside it. Otherwise it creates
    a new figure. This makes the helper usable both for single images and for
    multi-panel comparisons in notebooks.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def show_mask(mask, title=None, ax=None):
    """Display one binary mask.

    The blue colormap is used because the mask represents detected water. This
    is only a visualization choice; the mask itself remains binary.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mask, cmap="Blues")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def overlay_mask(image, mask, color=(0.0, 0.45, 1.0), alpha=0.45):
    """Overlay a binary or probability mask on an RGB image.

    Default values:
    - color=(0.0, 0.45, 1.0): a saturated blue, chosen because it visually
      represents water and contrasts with many industrial backgrounds.
    - alpha=0.45: keeps the original image visible while making the detected
      water zone clear. A value near 0 would be invisible; a value near 1 would
      hide image details.
    """
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()
    # Any positive value is treated as selected. This supports both binary masks
    # and probability masks that were already thresholded elsewhere.
    mask = mask > 0

    color_array = np.array(color, dtype=np.float32)
    image[mask] = (1.0 - alpha) * image[mask] + alpha * color_array
    return np.clip(image, 0.0, 1.0)


def show_image_mask_overlay(image, mask, title=None):
    """Display image, mask, and overlay side by side.

    This is the standard visualization used in the notebooks: original image,
    predicted or ground-truth mask, and the mask projected on the image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    show_image(image, "Image", axes[0])
    show_mask(mask, "Masque eau", axes[1])
    show_image(overlay_mask(image, mask), "Overlay", axes[2])
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, axes
