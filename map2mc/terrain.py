"""Height map reconstruction from a 2D map image.

A top-down map contains no explicit elevation, so heights are reconstructed
from cues that *are* in the image – and nothing random:

  1. distance from the coast   – land rises gently inland, beaches stay low
  2. terrain class            – rock / dark-rock areas become mountains whose
                                 peaks sit in the middle of the rock area
  3. map shading (optional)   – many maps are hill-shaded; the brightness
                                 gradient is integrated into relief (FFT
                                 shape-from-shading, light from the north-west)
  4. built-up areas and roads are flattened so they stay walkable
  5. water depth grows with distance from land; inland lakes get their own
     water level from the surrounding shore

All heights are in blocks relative to the sea surface (water_level).
Because elevation is an estimate, the GUI offers raise/lower/flatten brushes
and the import of a real height map that replaces the estimate.
"""
from __future__ import annotations

import cv2
import numpy as np

from .analysis import Structure, _to_lab
from .config import CID, Settings


def _dist(mask: np.ndarray) -> np.ndarray:
    """Distance (pixels) of every True pixel to the nearest False pixel."""
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    if mask.all():
        return np.full(mask.shape, float(max(mask.shape)), np.float32)
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)


def _masked_blur(v: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages over `mask` (keeps coastlines sharp)."""
    if sigma <= 0:
        return v
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(v * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    out = np.where(den > 1e-4, num / np.maximum(den, 1e-4), v)
    return np.where(mask, out, v).astype(np.float32)


def shading_relief(img: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Integrate hill-shading into relative relief (unit-less, ~[-1,1])."""
    L = _to_lab(img)[..., 0]
    L = cv2.medianBlur(L.astype(np.float32), 5)         # suppress tiny icons / tree dots
    hp = L - cv2.GaussianBlur(L, (0, 0), 12)
    hp = np.clip(hp, -12, 12) * land
    h, w = hp.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    kt = (kx + ky) / np.sqrt(2.0)                      # derivative direction NW -> SE
    eps = 0.01
    S = np.fft.fft2(hp)
    H = S * np.conj(2j * np.pi * kt) / ((2 * np.pi * kt) ** 2 + eps ** 2)
    rel = np.real(np.fft.ifft2(H)).astype(np.float32)
    rel -= cv2.GaussianBlur(rel, (0, 0), 25)           # remove drift / streaks
    s = np.percentile(np.abs(rel[land]), 98) if land.any() else 1.0
    return np.clip(rel / max(s, 1e-6), -1.5, 1.5)


def compute_heights(img: np.ndarray, st: Structure, cfg: Settings, progress=None):
    """Return (height, lake_surface) float32 arrays in blocks relative to water level.

    height        – top solid block (land > 0, sea floor < 0)
    lake_surface  – water surface of inland water (0 for the sea)
    """
    s = float(cfg.blocks_per_pixel)
    cls = st.classes
    land = st.land
    water = ~land
    H = float(cfg.max_land_height)

    d = _dist(land) * s                                # blocks to nearest water
    dmax = max(float(d.max()), 1.0)
    h = (cfg.beach_height * np.clip(d / 8.0, 0, 1)
         + 0.35 * H * (1 - np.exp(-d / 140.0))
         + 0.25 * H * (d / dmax) ** 0.8).astype(np.float32)
    if progress:
        progress(0.2)

    # beaches stay flat
    sand = cls == CID["sand"]
    h = np.where(sand, np.minimum(h, cfg.beach_height + 0.06 * d), h)

    # mountains: peaks in the middle of rock areas
    for key, factor, falloff in (("rock", 1.0, 45.0), ("dark_rock", 1.3, 55.0), ("snow", 1.2, 60.0)):
        m = cls == CID[key]
        if m.any():
            m2 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            dr = _dist(m2) * s
            bump = cfg.mountain_height * factor * (1 - np.exp(-dr / falloff))
            bump = cv2.GaussianBlur(bump.astype(np.float32), (0, 0), 3)
            h = h + bump * land
    if progress:
        progress(0.4)

    if cfg.relief_strength > 0:
        rel = shading_relief(img, land)
        coast_fade = np.clip(d / 24.0, 0, 1)
        h = h + rel * cfg.relief_strength * 0.25 * H * coast_fade
    if progress:
        progress(0.6)

    h = _masked_blur(h.astype(np.float32), land, cfg.smoothing)

    # flatten towns and roads (they must be walkable)
    flat = np.isin(cls, [CID["urban"], CID["road"], CID["path"]])
    if flat.any():
        strong = _masked_blur(h, land, 6.0)
        w = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 2.0)
        h = h * (1 - w) + strong * w
    h = np.where(land, np.maximum(h, 1.0), h)
    if progress:
        progress(0.75)

    # --- water
    dw = _dist(water) * s
    shallow = cls == CID["water_shallow"]
    depth = np.where(shallow, np.minimum(2.0 + dw * 0.04, 5.0),
                     np.minimum(float(cfg.max_ocean_depth), 4.0 + dw * 0.10))
    depth = _masked_blur(depth.astype(np.float32), water, 2.0)
    lake_surface = np.zeros(h.shape, np.float32)
    ids = np.unique(st.lake_id)
    ids = ids[ids > 0]
    if ids.size:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((st.lake_id > 0).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            x0, y0 = max(0, x - 3), max(0, y - 3)
            x1, y1 = min(h.shape[1], x + ww + 3), min(h.shape[0], y + hh + 3)
            sub = lab[y0:y1, x0:x1] == i
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~sub
            shore = h[y0:y1, x0:x1][ring & land[y0:y1, x0:x1]]
            surf = max(0.0, float(np.percentile(shore, 10)) - 1.0) if shore.size else 0.0
            ls = lake_surface[y0:y1, x0:x1]
            ls[sub] = round(surf)
            dl = depth[y0:y1, x0:x1]
            dl[sub] = np.minimum(dl[sub], 2.0 + min(6.0, area ** 0.5 * 0.3))
    floor = lake_surface - depth
    h = np.where(water, floor, h).astype(np.float32)
    if progress:
        progress(1.0)
    return h, lake_surface


def apply_edits(h: np.ndarray, delta: np.ndarray | None, override: np.ndarray | None,
                land: np.ndarray) -> np.ndarray:
    """Combine estimated heights with manual brush edits / an imported height map."""
    out = h.copy()
    if override is not None:
        out = np.where(land, np.maximum(override, 1.0), out)
    if delta is not None:
        out = out + delta
    return out.astype(np.float32)
