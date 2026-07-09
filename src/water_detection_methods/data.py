from pathlib import Path

import numpy as np
from PIL import Image

from .paths import LOCAL_IMAGES_DIR, WATER_V2_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(root, recursive=True):
    """Return sorted image paths from a directory."""
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_local_images(local_dir=LOCAL_IMAGES_DIR):
    """Return images from the local laverie folder."""
    return list_image_paths(local_dir)


def list_water_v2_images(dataset_dir=WATER_V2_DIR):
    """Return all images from water_v2/JPEGImages."""
    return list_image_paths(Path(dataset_dir) / "JPEGImages")


def find_water_v2_pairs(dataset_dir=WATER_V2_DIR):
    """Pair water_v2 images with masks when a matching annotation exists."""
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
        mask_path = next((path for path in candidate_paths if path.exists()), None)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


def load_image(path, size=None):
    """Load an RGB image as a numpy array in [0, 1]."""
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path, size=None, threshold=0):
    """Load an annotation mask as a binary numpy array."""
    mask = Image.open(path).convert("L")
    if size is not None:
        mask = mask.resize(size, Image.NEAREST)
    mask_array = np.asarray(mask)
    return (mask_array > threshold).astype(np.uint8)


def load_pair(image_path, mask_path, size=None):
    """Load an image and its binary mask."""
    return load_image(image_path, size=size), load_mask(mask_path, size=size)


def train_val_split(items, val_ratio=0.2, seed=42):
    """Deterministic split for paths or pairs."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    val_count = int(len(items) * val_ratio)
    val_indices = set(indices[:val_count])
    train_items = [item for index, item in enumerate(items) if index not in val_indices]
    val_items = [item for index, item in enumerate(items) if index in val_indices]
    return train_items, val_items

