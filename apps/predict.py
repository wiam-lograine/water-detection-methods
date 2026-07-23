#!/usr/bin/env python3
"""CLI for water segmentation inference.

Examples
--------
    python apps/predict.py IMGs/img1.jpg --checkpoint models/unet_water_v2_best.pt -o output/
    python apps/predict.py IMGs/ --checkpoint runs/best.pt --batch --overlay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from water_detection_methods.inference import Predictor  # noqa: E402
from water_detection_methods.paths import MODELS_DIR  # noqa: E402


def _save_mask(mask: np.ndarray, output_path: Path) -> None:
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img.save(output_path)


def _save_overlay(overlay: np.ndarray, output_path: Path) -> None:
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8).clip(0, 255), mode="RGB")
    overlay_img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Water segmentation inference")
    parser.add_argument(
        "input",
        type=str,
        help="Path to an image file or a directory of images",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=str(MODELS_DIR / "unet_water_v2_best.pt"),
        help="Path to the .pt checkpoint (default: models/unet_water_v2_best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Directory to save predictions (default: outputs/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override prediction threshold",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all images in INPUT directory",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Save overlay image alongside the mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device (default: "cuda" if available else "cpu")',
    )
    parser.add_argument(
        "--save-prob",
        action="store_true",
        help="Save raw probability map instead of binary mask",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    if args.batch or input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in exts
        ) if input_path.is_dir() else sorted(
            p for p in input_path.glob("*") if p.suffix.lower() in exts
        )
        if not image_paths:
            print(f"Aucune image trouvée dans {input_path}")
            sys.exit(1)
    else:
        if not input_path.is_file():
            print(f"Fichier introuvable : {input_path}")
            sys.exit(1)
        image_paths = [input_path]

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Périphérique : {args.device or ('CUDA' if __import__('torch').cuda.is_available() else 'CPU')}")
    print(f"Images      : {len(image_paths)}")

    predictor = Predictor(args.checkpoint, device=args.device)
    if args.threshold is not None:
        predictor.threshold = args.threshold

    for image_path in image_paths:
        result = predictor.predict(image_path, return_overlay=args.overlay, raw_logits=args.save_prob)

        stem = image_path.stem
        if args.save_prob:
            path = output_dir / f"{stem}_prob.png"
            prob_img = Image.fromarray(
                ((result * 255).astype(np.uint8).clip(0, 255)), mode="L"
            )
            prob_img.save(path)
            print(f"  Probabilités : {path}")
        elif args.overlay:
            mask, overlay = result
            _save_mask(mask, output_dir / f"{stem}_mask.png")
            _save_overlay(overlay, output_dir / f"{stem}_overlay.png")
            print(f"  Masque  : {output_dir / f'{stem}_mask.png'}")
            print(f"  Overlay : {output_dir / f'{stem}_overlay.png'}")
        else:
            _save_mask(result, output_dir / f"{stem}_mask.png")
            print(f"  Masque  : {output_dir / f'{stem}_mask.png'}")

    print("Terminé.")


if __name__ == "__main__":
    main()
