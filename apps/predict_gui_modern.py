#!/usr/bin/env python3
"""ttkbootstrap GUI for water segmentation inference.

Modern themed version of predict_gui.py using ttkbootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog, PhotoImage  # keep native file dialogs

from water_detection_methods.inference import Predictor  # noqa: E402
from water_detection_methods.paths import MODELS_DIR, OUTPUTS_DIR  # noqa: E402

PANEL_SIZE = (420, 320)


def image_array_to_pil(image: np.ndarray) -> Image.Image:
    arr = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    arr = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").convert("RGB")


def fit_on_panel(image, size=PANEL_SIZE):
    panel: Image.Image = Image.new("RGB", size, (245, 247, 250))
    preview: Image.Image = image.copy()
    preview.thumbnail(size, resample = Image.Resampling.LANCZOS)
    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2
    panel.paste(preview, (x, y))
    return panel


class PredictGUI:
    def __init__(self) -> None:
        self.root = ttk.Window(themename="superhero")
        self.root.title("Detection d'eau - Inference")
        self.root.minsize(1380, 700)

        self.image_path: Path | None = None
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.overlay: np.ndarray | None = None
        self.predictor: Predictor | None = None
        self.panel_images: dict = {}

        self.checkpoint_path = ttk.StringVar(value=str(MODELS_DIR / "unet_water_v2_best.pt"))
        self.threshold = ttk.DoubleVar(value=0.5)

        self._build_layout()

    def _build_layout(self) -> None:
        # ── Header ──────────────────────────────────────────────────────
        header = ttk.Frame(self.root, padding=(16, 14, 16, 10))
        header.pack(fill=X)
        ttk.Label(
            header, text="Detection d'eau — Segmentation Sementique par IA",
            font=("Segoe UI", 18, "bold"),
        ).pack(side=LEFT)

        theme_btn = ttk.Menubutton(header, text="Theme", bootstyle=SECONDARY)
        theme_menu = ttk.Menu(theme_btn)
        for name in ["superhero", "darkly", "cyborg", "flatly", "journal", "litera", "minty", "cosmo"]:
            theme_menu.add_command(
                label=name,
                command=lambda n=name: self.root.style.theme_use(n),
            )
        theme_btn["menu"] = theme_menu
        theme_btn.pack(side=RIGHT, padx=(8, 0))

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        toolbar.pack(fill=X)

        ttk.Button(
            toolbar, text="  Choisir une image", bootstyle=PRIMARY,
            command=self.open_image,
        ).pack(side=LEFT, padx=(0, 8))

        ttk.Button(
            toolbar, text="  Choisir un modele", bootstyle=SECONDARY,
            command=self.select_checkpoint,
        ).pack(side=LEFT, padx=(0, 12))

        ttk.Label(toolbar, text="Seuil :").pack(side=LEFT, padx=(0, 4))
        threshold_slider = ttk.Scale(
            toolbar, from_=0.0, to=1.0, variable=self.threshold,
            command=lambda _v: self._refresh_threshold_label(),
            length=140,
        )
        threshold_slider.pack(side=LEFT, padx=(0, 6))
        self.threshold_label = ttk.Label(
            toolbar, text="0.50", font=("Segoe UI", 10, "bold"), width=4,
        )
        self.threshold_label.pack(side=LEFT, padx=(0, 16))

        ttk.Button(
            toolbar, text=" Inference", bootstyle=SUCCESS,
            command=self.run_inference,
        ).pack(side=LEFT, padx=(0, 8))

        ttk.Button(
            toolbar, text=" Sauvegarder", bootstyle=INFO,
            command=self.save_outputs,
        ).pack(side=LEFT)

        self.status = ttk.Label(
            toolbar, text="Aucune image chargee",
            font=("Segoe UI", 9), foreground="#888888",
        )
        self.status.pack(side=LEFT, padx=(20, 0))

        # ── Separator ───────────────────────────────────────────────────
        ttk.Separator(self.root, orient=HORIZONTAL).pack(fill=X, padx=16)

        # ── Panels ──────────────────────────────────────────────────────
        panels_frame = ttk.Frame(self.root, padding=(16, 16, 16, 4))
        panels_frame.pack(fill=BOTH, expand=True)
        panels_frame.columnconfigure(0, weight=1)
        panels_frame.columnconfigure(1, weight=1)
        panels_frame.columnconfigure(2, weight=1)

        self.original_panel = self._create_panel(panels_frame, "Image originale", 0)
        self.mask_panel = self._create_panel(panels_frame, "Masque predit", 1)
        self.overlay_panel = self._create_panel(panels_frame, "Overlay", 2)

        # ── Metrics bar ─────────────────────────────────────────────────
        metrics_bar = ttk.Frame(self.root, padding=(16, 8, 16, 14))
        metrics_bar.pack(fill=X)
        self.metrics = ttk.Label(metrics_bar, text="", font=("Segoe UI", 10))
        self.metrics.pack(side=LEFT)

        # progress bar for inference feedback
        self.progress = ttk.Progressbar(
            metrics_bar, mode=INDETERMINATE, length=160, bootstyle=SUCCESS,
        )
        self.progress.pack(side=RIGHT)

    def _refresh_threshold_label(self) -> None:
        self.threshold_label.configure(text=f"{self.threshold.get():.2f}")

    def _create_panel(self, parent: ttk.Frame, title: str, column: int) -> ttk.Label:
        frame = ttk.Labelframe(parent, text=title, padding=10, bootstyle=PRIMARY)
        frame.grid(row=0, column=column, sticky="nsew", padx=8)
        label = ttk.Label(frame)
        label.pack()
        self._set_panel_image(label, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        return label

    def open_image(self) -> None:
        filetypes = [
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("Tous les fichiers", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Choisir une image",
            initialdir=ROOT / "IMGs" if (ROOT / "IMGs").exists() else ROOT,
            filetypes=filetypes,
        )
        if not path:
            return
        self.image_path = Path(path)
        self.image = np.asarray(
            Image.open(self.image_path).convert("RGB"), dtype=np.float32
        ) / 255.0
        self.status.configure(text=self.image_path.name)
        self.mask = None
        self.overlay = None
        self._set_panel_image(self.original_panel, image_array_to_pil(self.image))
        self._set_panel_image(self.mask_panel, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        self._set_panel_image(self.overlay_panel, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        self.metrics.configure(text="")

    def select_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir un checkpoint",
            initialdir=MODELS_DIR if MODELS_DIR.exists() else ROOT,
            filetypes=[("PyTorch", "*.pt"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        self.checkpoint_path.set(path)

    def run_inference(self) -> None:
        if self.image_path is None:
            Messagebox.show_info("Choisissez d'abord une image.", title="Aucune image")
            return

        ckpt = self.checkpoint_path.get()
        if not Path(ckpt).is_file():
            Messagebox.show_error(f"Checkpoint introuvable :\n{ckpt}", title="Erreur")
            return

        self.progress.start()
        self.status.configure(text=f"{self.image_path.name} - inference en cours...")
        self.root.update_idletasks()

        try:
            self.predictor = Predictor(ckpt)
        except Exception as exc:
            self.progress.stop()
            Messagebox.show_error(f"Impossible de charger le modele :\n{exc}", title="Erreur")
            self.status.configure(text=self.image_path.name)
            return

        self.predictor.threshold = self.threshold.get()

        try:
            mask, overlay = self.predictor.predict(self.image_path, return_overlay=True)
        except Exception as exc:
            self.progress.stop()
            Messagebox.show_error(f"L'inference a echoue :\n{exc}", title="Erreur")
            self.status.configure(text=self.image_path.name)
            return

        self.progress.stop()
        self.mask = mask
        self.overlay = overlay

        self._set_panel_image(self.original_panel, image_array_to_pil(self.image))
        self._set_panel_image(self.mask_panel, mask_to_pil(self.mask))
        self._set_panel_image(self.overlay_panel, image_array_to_pil(self.overlay))

        coverage = float(self.mask.mean()) * 100.0
        self.metrics.configure(
            text=f"Eau detectee : {coverage:.1f} %  |  Seuil : {self.predictor.threshold:.2f}"
        )
        self.status.configure(text=self.image_path.name)

    def _set_panel_image(self, label: ttk.Label, pil_image: Image.Image) -> None:
        preview: Image.Image = fit_on_panel(pil_image)
        tk_image = ImageTk.PhotoImage(preview)
        label.configure(image=tk_image)
        self.panel_images[label] = tk_image

    def save_outputs(self) -> None:
        if self.image_path is None or self.mask is None or self.overlay is None:
            Messagebox.show_info("Lancez d'abord l'inference.", title="Aucun resultat")
            return

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        mask_path = OUTPUTS_DIR / f"{stem}_mask.png"
        overlay_path = OUTPUTS_DIR / f"{stem}_overlay.png"

        mask_to_pil(self.mask).save(mask_path)
        image_array_to_pil(self.overlay).save(overlay_path)

        Messagebox.show_info(
            f"Masque : {mask_path}\nOverlay : {overlay_path}",
            title="Resultats sauvegardes",
        )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = PredictGUI()
    app.run()
