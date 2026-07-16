import numpy as np
from typing import Any, TypeAlias
import numpy.typing as npt


FloatImage: TypeAlias = npt.NDArray[np.floating[Any]]

def blue_dominance_threshold(
    image: FloatImage,
    blue_min: float=0.18,
    blue_red_ratio: float=1.05,
    blue_green_ratio: float =0.85,
    brightness_min: float =0.05,
) -> npt.NDArray[np.uint8]:
    """Segment water with a simple blue-channel dominance rule.

    The input image is expected to be RGB and normalized in [0, 1], as returned
    by `load_image`.

    This baseline is based on a simple observation: in many outdoor water
    images, water pixels often have a relatively strong blue component compared
    with red and green. It is not a universal rule, but it is easy to explain
    and useful as a first comparison point before ML or Deep Learning.

    Default thresholds:
    - blue_min=0.18: keeps pixels with at least a small blue intensity. Since
      values are normalized in [0, 1], 0.18 is permissive enough to keep dark
      water while removing very dark pixels and shadows.
    - blue_red_ratio=1.05: requires blue to be slightly stronger than red. A
      small margin avoids selecting neutral gray surfaces while remaining
      tolerant to muddy or industrial water that is not perfectly blue.
    - blue_green_ratio=0.85: does not require blue to be stronger than green,
      because natural water can appear greenish. The value keeps green-blue
      water while rejecting many yellow/brown surfaces.
    - brightness_min=0.05: removes almost black pixels, where color ratios are
      unreliable because all channels are close to zero.

    These values are starting points for the baseline interface. They should be
    tuned on the real laverie images and compared with annotated masks.
    """
    image = np.asarray(image, dtype=np.float32)

    # Split channels to make the rule explicit and easy to discuss in reports.
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    brightness = image.mean(axis=2)

    # A pixel is selected as water only if it satisfies all simple conditions.
    mask = (
        (blue >= blue_min)
        & (blue >= red * blue_red_ratio)
        & (blue >= green * blue_green_ratio)
        & (brightness >= brightness_min)
    )
    return mask.astype(np.uint8)


def mask_coverage(mask):
    """Return the percentage of pixels selected by a binary mask.

    This is useful for quick interpretation in the GUI: if coverage is very
    high, the thresholds may be too permissive; if it is close to zero, they may
    be too strict or the image may not contain visible water.
    """
    mask = np.asarray(mask).squeeze() > 0
    return float(mask.mean() * 100.0)
