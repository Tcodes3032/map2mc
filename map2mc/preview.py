"""Preview renderers (pure NumPy): shaded top view and an oblique 3D view."""
from __future__ import annotations

import cv2
import numpy as np

from .config import CID, CLASSES

# colours close to the Minecraft blocks used for each class
BLOCK_COLORS = {
    "void": (38, 60, 110), "water_deep": (38, 70, 150), "water_shallow": (60, 140, 190),
    "sand": (219, 207, 163), "grass": (110, 160, 70), "forest": (70, 120, 50),
    "desert": (190, 105, 50), "rock": (125, 125, 125), "dark_rock": (45, 40, 45),
    "path": (150, 120, 75), "road": (80, 82, 88), "urban": (160, 160, 160),
    "snow": (245, 250, 255), "blossom": (230, 160, 200), "magic": (120, 100, 150),
    "unknown": (255, 0, 255),
}


def _lut():
    return np.array([BLOCK_COLORS.get(c[0], c[2]) for c in CLASSES], np.float32)


def shaded(classes: np.ndarray, height: np.ndarray, land: np.ndarray, exaggeration: float = 1.0) -> np.ndarray:
    """Top-down preview: block colours × hillshade, water tinted by depth."""
    col = _lut()[classes]
    gy, gx = np.gradient(height * exaggeration)
    shade = np.clip(0.75 + (-gx - gy) * 0.22, 0.35, 1.25)
    col = col * np.where(land, shade, 1.0)[..., None]
    depth = np.clip(-height / 24.0, 0, 1)[..., None]
    water = np.array([70, 160, 200], np.float32) * (1 - depth) + np.array([25, 55, 120], np.float32) * depth
    col = np.where(land[..., None], col, water)
    return np.clip(col, 0, 255).astype(np.uint8)


def height_view(height: np.ndarray) -> np.ndarray:
    h = height.astype(np.float32)
    lo, hi = float(h.min()), float(max(h.max(), 1))
    norm = np.where(h >= 0, 0.5 + 0.5 * h / hi, 0.5 - 0.5 * h / min(lo, -1))
    norm = np.clip(norm, 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def oblique(classes: np.ndarray, height: np.ndarray, land: np.ndarray, rotation: int = 0,
            size: int = 640, exaggeration: float = 1.6, tilt: float = 0.55) -> np.ndarray:
    """Oblique 3D view: columns are drawn front-to-back with a y-buffer.

    rotation: 0..3 (quarter turns).  The source is downsampled to `size`
    columns so this stays interactive for big maps.
    """
    rgb = shaded(classes, height, land)
    h = np.where(land, height, 0.0).astype(np.float32)   # water shown as flat surface
    rgb = np.rot90(rgb, rotation)
    h = np.rot90(h, rotation)
    H, W = h.shape
    f = min(1.0, size / max(H, W))
    if f < 1.0:
        rgb = cv2.resize(rgb, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        h = cv2.resize(h, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    H, W = h.shape
    hs = h * f * exaggeration                            # heights in output pixels
    rows = np.arange(H, dtype=np.float32) * tilt
    top = rows[:, None] - hs                            # screen y of each column top
    off = float(max(0.0, -top.min())) + 4
    top += off
    out_h = int(rows[-1] + off + 8)
    img = np.zeros((out_h, W, 3), np.uint8)
    img[:] = (24, 26, 32)
    # front (bottom rows) to back: a column is visible where it rises above the y-buffer
    ybuf = np.full(W, out_h, np.float32)
    xs = np.arange(W)
    for r in range(H - 1, -1, -1):
        t = top[r]
        vis = t < ybuf
        if not vis.any():
            continue
        y0 = np.floor(t[vis]).astype(int)
        y1 = np.ceil(ybuf[vis]).astype(int)
        c = rgb[r, vis]
        side = (c * 0.72).astype(np.uint8)
        xv = xs[vis]
        lengths = np.clip(y1 - y0, 1, None)
        maxlen = int(lengths.max())
        for k in range(min(maxlen, 64)):           # paint vertical spans (bounded)
            m = k < lengths
            yy = np.clip(y0[m] + k, 0, out_h - 1)
            img[yy, xv[m]] = c[m] if k < 2 else side[m]   # flat rows span 1-2 px: no side shading
        ybuf[vis] = t[vis]
    return img
