from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from water_detection_methods.baselines import blue_dominance_threshold, mask_coverage
from water_detection_methods.data import IMAGE_EXTENSIONS, load_image
from water_detection_methods.paths import LOCAL_IMAGES_DIR, OUTPUTS_DIR
from water_detection_methods.visualization import overlay_mask

PANEL_SIZE = (360, 270)


def image_array_to_pil(image):
    image = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def mask_to_pil(mask):
    mask = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L").convert("RGB")


def fit_on_panel(image, size=PANEL_SIZE):
    panel: Image.Image = Image.new("RGB", size, (245, 247, 250))
    preview: Image.Image = image.copy()
    preview.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2
    panel.paste(preview, (x, y))
    return panel


class ThresholdGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Detection d'eau - Baseline seuillage")
        self.root.minsize(1180, 650)

        self.image_path = None
        self.image = None
        self.mask = None
        self.overlay = None
        self.panel_images = {}

        self.blue_min = tk.DoubleVar(value=0.18)
        self.blue_red_ratio = tk.DoubleVar(value=1.05)
        self.blue_green_ratio = tk.DoubleVar(value=0.85)
        self.brightness_min = tk.DoubleVar(value=0.05)
        self.alpha = tk.DoubleVar(value=0.45)

        self._build_layout()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(2, weight=1)

        ttk.Button(toolbar, text="Choisir une image", command=self.open_image).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Sauvegarder les resultats", command=self.save_outputs).grid(
            row=0, column=1, padx=(0, 12)
        )
        self.status = ttk.Label(toolbar, text="Aucune image chargee")
        self.status.grid(row=0, column=2, sticky="w")

        self.original_panel = self._create_panel(main, "Image originale", 0)
        self.mask_panel = self._create_panel(main, "Masque", 1)
        self.overlay_panel = self._create_panel(main, "Overlay", 2)

        controls = ttk.LabelFrame(main, text="Parametres du seuillage", padding=12)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        for column in range(4):
            controls.columnconfigure(column, weight=1)

        self._add_slider(
            controls,
            "Bleu minimum",
            self.blue_min,
            0.0,
            1.0,
            0,
            "Intensite minimale du canal bleu.",
        )
        self._add_slider(
            controls,
            "Bleu / rouge",
            self.blue_red_ratio,
            0.5,
            2.0,
            1,
            "Plus la valeur est haute, plus le bleu doit dominer le rouge.",
        )
        self._add_slider(
            controls,
            "Bleu / vert",
            self.blue_green_ratio,
            0.5,
            2.0,
            2,
            "Plus la valeur est haute, plus le bleu doit dominer le vert.",
        )
        self._add_slider(
            controls,
            "Luminosite min.",
            self.brightness_min,
            0.0,
            1.0,
            3,
            "Filtre les zones trop sombres.",
        )
        self._add_slider(
            controls,
            "Transparence overlay",
            self.alpha,
            0.05,
            0.95,
            4,
            "Controle la force de l'overlay bleu.",
        )

        self.metrics = ttk.Label(main, text="Pixels detectes comme eau : -")
        self.metrics.grid(row=3, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _create_panel(self, parent, title, column):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=1, column=column, sticky="nsew", padx=6)
        parent.columnconfigure(column, weight=1)
        label = ttk.Label(frame)
        label.grid(row=0, column=0)
        self._set_panel_image(label, Image.new("RGB", PANEL_SIZE, (245, 247, 250)))
        return label

    def _add_slider(
        self,
        parent,
        label,
        variable,
        from_value,
        to_value,
        row,
        help_text,
    ):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        slider = ttk.Scale(
            parent,
            from_=from_value,
            to=to_value,
            variable=variable,
            command=lambda _value: self.update_prediction(),
        )
        slider.grid(row=row, column=1, sticky="ew", padx=8)
        value_label = ttk.Label(parent, width=8)
        value_label.grid(row=row, column=2, sticky="w")
        ttk.Label(parent, text=help_text).grid(row=row, column=3, sticky="w", padx=(8, 0))

        def refresh_value(*_args):
            value_label.configure(text=f"{variable.get():.2f}")

        variable.trace_add("write", refresh_value)
        refresh_value()

    def open_image(self):
        filetypes = [
            ("Images", " ".join(f"*{extension}" for extension in sorted(IMAGE_EXTENSIONS))),
            ("Tous les fichiers", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Choisir une image",
            initialdir=LOCAL_IMAGES_DIR if LOCAL_IMAGES_DIR.exists() else ROOT,
            filetypes=filetypes,
        )
        if not path:
            return

        self.image_path = Path(path)
        self.image = load_image(self.image_path)
        self.status.configure(text=str(self.image_path))
        self.update_prediction()

    def update_prediction(self):
        if self.image is None:
            return

        self.mask = blue_dominance_threshold(
            self.image,
            blue_min=self.blue_min.get(),
            blue_red_ratio=self.blue_red_ratio.get(),
            blue_green_ratio=self.blue_green_ratio.get(),
            brightness_min=self.brightness_min.get(),
        )
        self.overlay = overlay_mask(self.image, self.mask, alpha=self.alpha.get())

        self._set_panel_image(self.original_panel, image_array_to_pil(self.image))
        self._set_panel_image(self.mask_panel, mask_to_pil(self.mask))
        self._set_panel_image(self.overlay_panel, image_array_to_pil(self.overlay))
        self.metrics.configure(
            text=f"Pixels detectes comme eau : {mask_coverage(self.mask):.2f} %"
        )

    def _set_panel_image(self, label, pil_image):
        preview = fit_on_panel(pil_image)
        tk_image = ImageTk.PhotoImage(preview)
        label.configure(image=tk_image)
        self.panel_images[label] = tk_image

    def save_outputs(self):
        if self.image_path is None or self.mask is None or self.overlay is None:
            messagebox.showinfo("Aucune image", "Choisissez d'abord une image.")
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
    app = ThresholdGui(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    main()
