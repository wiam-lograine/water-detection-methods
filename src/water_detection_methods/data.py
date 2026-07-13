from pathlib import Path

import numpy as np
from PIL import Image

from .paths import LOCAL_IMAGES_DIR, WATER_V2_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(root, recursive=True):
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


def list_local_images(local_dir=LOCAL_IMAGES_DIR):
    """Return images from the local laverie folder.

    The default folder is `IMGs`, which contains the real images from the
    laverie. These images are mainly used to test whether a method trained or
    tuned on `water_v2` generalizes to the real industrial context.
    """
    return list_image_paths(local_dir)


def list_water_v2_images(dataset_dir=WATER_V2_DIR):
    """Return all images from water_v2/JPEGImages.

    `water_v2/JPEGImages` contains the raw input images. This helper does not
    require annotations, so it can also be used for simple visualization or
    unsupervised baselines.
    """
    return list_image_paths(Path(dataset_dir) / "JPEGImages")


def find_water_v2_pairs(dataset_dir=WATER_V2_DIR):
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


def load_image(path, size=None):
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


def load_mask(path, size=None, threshold=0):
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


def load_pair(image_path, mask_path, size=None) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its binary mask.

    This helper keeps notebooks short and guarantees that image and mask are
    resized with the correct interpolation methods.
    """
    return load_image(image_path, size=size), load_mask(mask_path, size=size)


def train_val_split(items, val_ratio=0.2, seed=42):
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
