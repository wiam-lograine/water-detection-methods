from typing import Any, Optional, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def show_image(
    image: Union[list, np.ndarray],
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
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


def show_mask(
    mask: Union[list, np.ndarray],
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
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


def overlay_mask(
    image: Union[list, np.ndarray],
    mask: Union[list, np.ndarray],
    color: tuple[float, float, float] = (0.0, 0.45, 1.0),
    alpha: float = 0.45,
) -> np.ndarray:
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


def show_image_mask_overlay(
    image: Union[list, np.ndarray],
    mask: Union[list, np.ndarray],
    title: Optional[str] = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
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


def plot_metric_bars(
    results: dict[str, dict[str, float]],
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "dice", "iou"),
    title: str = "Comparaison des résultats de segmentation",
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot grouped metric bars from a ``name -> metric mapping``.

    ``results`` can contain dataset splits, for example Train/Validation/Test,
    or different models evaluated with the same protocol. Metric values are
    expected in ``[0, 1]``. The function returns the Matplotlib figure and axis
    so callers can further customize or save the diagram.
    """
    if not results:
        raise ValueError("The results mapping is empty.")
    if not metrics:
        raise ValueError("At least one metric must be requested.")

    names = list(results)
    for name, values in results.items():
        missing = [metric for metric in metrics if metric not in values]
        if missing:
            raise KeyError(f"Missing metrics for {name}: {missing}")
        invalid = [
            metric for metric in metrics if not 0.0 <= values[metric] <= 1.0
        ]
        if invalid:
            raise ValueError(f"Metrics outside [0, 1] for {name}: {invalid}")

    positions = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, axis = plt.subplots(figsize=(max(9, 2.2 * len(names)), 5))

    for index, metric in enumerate(metrics):
        values = [results[name][metric] for name in names]
        offset = (index - (len(metrics) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values,
            width=width,
            label=metric.capitalize(),
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)

    axis.set_title(title)
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(positions, names)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=min(len(metrics), 5), loc="upper center")
    fig.tight_layout()
    return fig, axis
