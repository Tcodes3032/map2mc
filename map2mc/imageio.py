"""Loading and validating map images."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFile

# Allow large but not absurd maps (protects against decompression bombs).
Image.MAX_IMAGE_PIXELS = 400_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

MIN_SIZE = 64
SUPPORTED = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


class MapImageError(Exception):
    """Raised for unreadable, truncated or unsuitable images (message is user-facing)."""


def load_map_image(path: str, background_rgb=(47, 48, 55)) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB array or raise MapImageError.

    Transparent pixels are composited onto `background_rgb`, so they are later
    classified as "outside the map".
    """
    if not os.path.isfile(path):
        raise MapImageError(f"Datei nicht gefunden: {path}")
    if os.path.getsize(path) == 0:
        raise MapImageError("Die Datei ist leer.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise MapImageError(f"Dateityp {ext or '(keiner)'} wird nicht unterstützt. Erlaubt: {', '.join(SUPPORTED)}")
    try:
        with Image.open(path) as probe:
            probe.verify()          # detects truncated / corrupt files
        with Image.open(path) as im:
            im.load()
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, tuple(background_rgb) + (255,))
                bg.alpha_composite(im)
                im = bg
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except Image.DecompressionBombError as e:
        raise MapImageError(f"Bild ist zu groß ({e}).") from e
    except (OSError, SyntaxError, ValueError) as e:
        raise MapImageError(f"Bild ist beschädigt oder unlesbar: {e}") from e
    h, w = arr.shape[:2]
    if h < MIN_SIZE or w < MIN_SIZE:
        raise MapImageError(f"Bild ist zu klein ({w}×{h}); mindestens {MIN_SIZE}×{MIN_SIZE} Pixel.")
    if arr.std() < 2.0:
        raise MapImageError("Bild enthält praktisch nur eine Farbe – keine Karte erkennbar.")
    return arr
