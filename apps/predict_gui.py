#!/usr/bin/env python3
"""Tkinter GUI for water segmentation inference.
 
Lets the user select an image and a model checkpoint, runs inference,
and displays the original, mask, and overlay side by side.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from water_detection_methods.inference import Predictor  # noqa: E402
from water_detection_methods.paths import MODELS_DIR, OUTPUTS_DIR  # noqa: E402

PANEL_SIZE = (400, 300)


def image_array_to_pil(image):
    image = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def mask_to_pil(mask):
    mask = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").convert("RGB")


def fit_on_panel(image, size=PANEL_SIZE):
    panel: Image.Image = Image.new("RGB", size, (245, 247, 250))
    preview: Image.Image = image.copy()
    preview.thumbnail(size, resample = Image.Resampling.LANCZOS)
    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2
    panel.paste(preview, (x, y))
    return panel


class PredictGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection d'eau - Inference")
        self.root.minsize(1320, 650)

        self.image_path = None
        self.image = None
        self.mask = None
        self.overlay = None
        self.predictor = None
        self.panel_images = {}

        self.checkpoint_path = tk.StringVar(value=str(MODELS_DIR / "unet_water_v2_best.pt"))
        self.threshold = tk.DoubleVar(value=0.5)

        self._build_layout()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Toolbar
        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(5, weight=1)

        ttk.Button(toolbar, text="Choisir une image", command=self.open_image).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Choisir un modele", command=self.select_checkpoint).grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Label(toolbar, text="Seuil:").grid(row=0, column=2, padx=(0, 4))
        threshold_slider = ttk.Scale(
            toolbar, from_=0.0, to=1.0, variable=self.threshold,
            command=lambda _v: None,
        )
        threshold_slider.grid(row=0, column=3, padx=(0, 4))
        self.threshold_label = ttk.Label(toolbar, width=5)
        self.threshold_label.grid(row=0, column=4, padx=(0, 12))

        def refresh_threshold(*_args):
            self.threshold_label.configure(text=f"{self.threshold.get():.2f}")
        self.threshold.trace_add("write", refresh_threshold)
        refresh_threshold()

        ttk.Button(toolbar, text="Lancer l'inference", command=self.run_inference).grid(
            row=0, column=5, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Sauvegarder", command=self.save_outputs).grid(
            row=0, column=6
        )
        self.status = ttk.Label(toolbar, text="Aucune image chargee")
        self.status.grid(row=0, column=7, sticky="w", padx=(12, 0))

        # Panels
        self.original_panel = self._create_panel(main, "Image originale", 0)
        self.mask_panel = self._create_panel(main, "Masque predit", 1)
        self.overlay_panel = self._create_panel(main, "Overlay", 2)

        # Status bar
        self.metrics = ttk.Label(main, text="")
        self.metrics.grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _create_panel(self, parent, title, column):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=1, column=column, sticky="nsew", padx=6)
        parent.columnconfigure(column, weight=1)
        label = ttk.Label(frame)
        label.grid(row=0, column=0)
        self._set_panel_image(label, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        return label

    def open_image(self):
        filetypes = [("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Tous les fichiers", "*.*")]
        path = filedialog.askopenfilename(
            title="Choisir une image",
            initialdir=ROOT / "IMGs" if (ROOT / "IMGs").exists() else ROOT,
            filetypes=filetypes,
        )
        if not path:
            return
        self.image_path = Path(path)
        self.image = np.asarray(Image.open(self.image_path).convert("RGB"), dtype=np.float32) / 255.0
        self.status.configure(text=str(self.image_path))
        self.mask = None
        self.overlay = None
        self._set_panel_image(self.original_panel, image_array_to_pil(self.image))
        self._set_panel_image(self.mask_panel, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        self._set_panel_image(self.overlay_panel, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        self.metrics.configure(text="")

    def select_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Choisir un checkpoint",
            initialdir=MODELS_DIR if MODELS_DIR.exists() else ROOT,
            filetypes=[("PyTorch", "*.pt"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        self.checkpoint_path.set(path)

    def run_inference(self):
        if self.image_path is None:
            messagebox.showinfo("Aucune image", "Choisissez d'abord une image.")
            return

        ckpt = self.checkpoint_path.get()
        if not Path(ckpt).is_file():
            messagebox.showerror("Fichier introuvable", f"Checkpoint introuvable :\n{ckpt}")
            return

        try:
            self.predictor = Predictor(ckpt)
        except Exception as exc:
            messagebox.showerror("Erreur de chargement", f"Impossible de charger le modele :\n{exc}")
            return

        self.predictor.threshold = self.threshold.get()
        self.status.configure(text=f"{self.image_path.name} - inference en cours...")
        self.root.update_idletasks()

        try:
            mask, overlay = self.predictor.predict(self.image_path, return_overlay=True)
        except Exception as exc:
            messagebox.showerror("Erreur d'inference", f"L'inference a echoue :\n{exc}")
            self.status.configure(text=str(self.image_path))
            return

        self.mask = mask
        self.overlay = overlay

        self._set_panel_image(self.original_panel, image_array_to_pil(self.image))
        self._set_panel_image(self.mask_panel, mask_to_pil(self.mask))
        self._set_panel_image(self.overlay_panel, image_array_to_pil(self.overlay))

        coverage = float(self.mask.mean()) * 100.0
        self.metrics.configure(
            text=f"Eau detectee : {coverage:.1f} % | Seuil : {self.predictor.threshold:.2f}"
        )
        self.status.configure(text=str(self.image_path))

    def _set_panel_image(self, label, pil_image):
        preview = fit_on_panel(pil_image)
        tk_image = ImageTk.PhotoImage(preview)
        label.configure(image=tk_image)
        self.panel_images[label] = tk_image

    def save_outputs(self):
        if self.image_path is None or self.mask is None or self.overlay is None:
            messagebox.showinfo("Aucun resultat", "Lancez d'abord l'inference.")
            return

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        mask_path = OUTPUTS_DIR / f"{stem}_mask.png"
        overlay_path = OUTPUTS_DIR / f"{stem}_overlay.png"

        mask_to_pil(self.mask).save(mask_path)
        image_array_to_pil(self.overlay).save(overlay_path)

        messagebox.showinfo(
            "Resultats sauvegardes",
            f"Masque : {mask_path}\nOverlay : {overlay_path}",
        )


def main():
    root = tk.Tk()
    app = PredictGUI(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()
