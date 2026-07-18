from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image


from .paths import LOCAL_IMAGES_DIR, WATER_V2_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(root: Union[str, Path], recursive: bool = True) -> list[Path]:
    """Return sorted image paths from a directory.

    `recursive=True` is the default because `water_v2` is organized with many
    subfolders. Sorting makes experiments reproducible: the same code always
    reads images in the same order.
    """
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_local_images(local_dir: Union[str, Path] = LOCAL_IMAGES_DIR) -> list[Path]:
    """Return images from the local laverie folder.

    The default folder is `IMGs`, which contains the real images from the
    laverie. These images are mainly used to test whether a method trained or
    tuned on `water_v2` generalizes to the real industrial context.
    """
    return list_image_paths(local_dir)


def list_water_v2_images(dataset_dir: Union[str, Path] = WATER_V2_DIR) -> list[Path]:
    """Return all images from water_v2/JPEGImages.

    `water_v2/JPEGImages` contains the raw input images. This helper does not
    require annotations, so it can also be used for simple visualization or
    unsupervised baselines.
    """
    return list_image_paths(Path(dataset_dir) / "JPEGImages")


def find_water_v2_pairs(dataset_dir: Union[str, Path] = WATER_V2_DIR) -> list[tuple[Path, Path]]:
    """Pair water_v2 images with masks when a matching annotation exists.

    The function keeps only images that have a corresponding annotation in
    `water_v2/Annotations`. This is required for supervised segmentation,
    because the model needs both the image and the expected water mask.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "JPEGImages"
    annotations_dir = dataset_dir / "Annotations"
    pairs = []

    for image_path in list_image_paths(images_dir):
        relative = image_path.relative_to(images_dir)
        candidate_paths = [
            annotations_dir / relative.with_suffix(".png"),
            annotations_dir / relative.with_suffix(".jpg"),
            annotations_dir / relative.with_suffix(".jpeg"),
        ]
        # Masks may be stored with different image extensions, so we test the
        # common possibilities while preserving the same relative subfolder.
        mask_path = next((path for path in candidate_paths if path.exists()), None)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


def load_image(path: Union[str, Path], size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Load an RGB image as a numpy array in [0, 1].

    Images are converted to RGB so every approach receives the same three
    channels, even if the source file is grayscale or has an alpha channel.
    Normalizing to [0, 1] makes threshold values easier to interpret and keeps
    the format compatible with most Deep Learning pipelines.
    """
    image = Image.open(path).convert("RGB")
    if size is not None:
        # Bilinear interpolation is appropriate for natural images because it
        # gives smoother resized images than nearest-neighbor interpolation.
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Union[str, Path], size: Optional[tuple[int, int]] = None, threshold: int = 0) -> np.ndarray:
    """Load an annotation mask as a binary numpy array.

    `threshold=0` means every non-black pixel is considered water. This is a
    practical default for annotation masks where the background is encoded as 0
    and the class of interest is encoded with any positive value.

    If a dataset uses soft masks or grayscale confidence maps, this threshold
    can be increased, for example to 127 for masks in [0, 255].
    """
    mask = Image.open(path).convert("L")
    if size is not None:
        # Nearest-neighbor interpolation preserves class labels. Bilinear
        # interpolation would create artificial gray values between classes.
        mask = mask.resize(size, Image.Resampling.NEAREST)
    mask_array = np.asarray(mask)
    return (mask_array > threshold).astype(np.uint8)


def load_pair(image_path: Union[str, Path], mask_path: Union[str, Path], size: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its binary mask.

    This helper keeps notebooks short and guarantees that image and mask are
    resized with the correct interpolation methods.
    """
    return load_image(image_path, size=size), load_mask(mask_path, size=size)


def load_pair_with_padding(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    size: tuple[int, int] = (512, 512),
    image_fill: tuple[int, int, int] = (0, 0, 0),
    mask_fill: int = 0,
    mask_threshold: int = 0,
    return_metadata: bool = False,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Load and letterbox an image/mask pair without changing its aspect ratio.

    ``size`` follows Pillow's ``(width, height)`` convention. The image is
    resized with bilinear interpolation, while the categorical mask always
    uses nearest-neighbor interpolation. Both are centered on canvases of the
    requested size using identical padding.

    ``return_metadata=True`` also returns the geometry needed to remove the
    padding and restore a predicted mask to the original image resolution.
    """
    target_width, target_height = size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target size: {size}")

    image: Image.Image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size:
        raise ValueError(
            "Image and mask must have the same original size: "
            f"{image.size} != {mask.size} for {image_path} and {mask_path}"
        )

    original_width, original_height = image.size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    image = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    mask = mask.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.NEAREST,
    )

    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top

    image_canvas = Image.new("RGB", size, color=image_fill)
    mask_canvas = Image.new("L", size, color=mask_fill)
    image_canvas.paste(image, (pad_left, pad_top))
    mask_canvas.paste(mask, (pad_left, pad_top))

    image_array = np.asarray(image_canvas, dtype=np.float32) / 255.0
    mask_array = (
        np.asarray(mask_canvas) > mask_threshold
    ).astype(np.uint8)

    if not return_metadata:
        return image_array, mask_array

    metadata = {
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "target_size": size,
    }
    return image_array, mask_array, metadata


def restore_mask_from_padding(mask: Union[list, np.ndarray], metadata: dict[str, Any], threshold: float = 0.5) -> np.ndarray:
    """Remove letterbox padding and restore a mask to its original size."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask_array.shape}")

    pad_left, pad_top, pad_right, pad_bottom = metadata["padding"]
    target_width, target_height = metadata["target_size"]
    crop_right = target_width - pad_right
    crop_bottom = target_height - pad_bottom
    cropped = mask_array[pad_top:crop_bottom, pad_left:crop_right]

    binary = (cropped >= threshold).astype(np.uint8) * 255
    restored = Image.fromarray(binary, mode="L").resize(
        metadata["original_size"],
        resample=Image.Resampling.NEAREST,
    )
    return (np.asarray(restored) > 0).astype(np.uint8)


def train_val_split(items: list[Any], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Any], list[Any]]:
    """Deterministic split for paths or pairs.

    `val_ratio=0.2` is a common starting point: 80% of the data is kept for
    training and 20% for validation. This gives enough data for learning while
    keeping a meaningful validation set.

    `seed=42` makes the split reproducible, so metrics can be compared between
    approaches under the same train/validation separation.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    val_count = int(len(items) * val_ratio)
    val_indices = set(indices[:val_count])
    train_items = [item for index, item in enumerate(items) if index not in val_indices]
    val_items = [item for index, item in enumerate(items) if index in val_indices]
    return train_items, val_items
