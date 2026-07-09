import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title=None, ax=None):
    """Display one RGB image."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def show_mask(mask, title=None, ax=None):
    """Display one binary mask."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mask, cmap="Blues")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def overlay_mask(image, mask, color=(0.0, 0.45, 1.0), alpha=0.45):
    """Overlay a binary or probability mask on an RGB image."""
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = mask > 0

    color_array = np.array(color, dtype=np.float32)
    image[mask] = (1.0 - alpha) * image[mask] + alpha * color_array
    return np.clip(image, 0.0, 1.0)


def show_image_mask_overlay(image, mask, title=None):
    """Display image, mask, and overlay side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    show_image(image, "Image", axes[0])
    show_mask(mask, "Masque eau", axes[1])
    show_image(overlay_mask(image, mask), "Overlay", axes[2])
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig, axes

