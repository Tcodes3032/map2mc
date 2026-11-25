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


# classes whose colour is not terrain shading (flat man-made surfaces, water)
_NON_TERRAIN = ("urban", "road", "path", "water_deep", "water_shallow", "void", "unknown")


def _fill_masked(v: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Replace invalid pixels by a smooth average of nearby valid ones."""
    m = valid.astype(np.float32)
    num = cv2.GaussianBlur(v * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), float(v[valid].mean()) if valid.any() else 0.0)
    return np.where(valid, v, fill).astype(np.float32)


def shading_relief(img: np.ndarray, land: np.ndarray, cls: np.ndarray | None = None) -> np.ndarray:
    """Integrate hill-shading into relative relief (unit-less, ~[-1,1]).

    Map shading changes the lightness of a surface with its slope towards the
    light (north-west).  Before integrating, everything that is *not* shading
    is removed from the lightness:
      * small map icons and tree dots (morphological open/close),
      * the base brightness of each terrain class (albedo, e.g. forest is
        darker than grass) and the anti-aliased seams between classes,
      * towns, roads, paths and water, which are filled from their surroundings.
    The integration is a regularised least-squares solve in the Fourier domain:
    only the slope along the light direction is visible in the picture, so the
    perpendicular direction is damped instead of amplified (no diagonal stripes).
    """
    L = _to_lab(img)[..., 0].astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    L = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k)        # dark dots (tree crowns, icons)
    L = cv2.morphologyEx(L, cv2.MORPH_OPEN, k)         # bright dots (roofs, markers)
    valid = land.copy()
    if cls is not None:
        for key in _NON_TERRAIN:
            valid &= cls != CID[key]
        # albedo: compare each class with its own typical lightness
        for c in np.unique(cls[valid]):
            m = valid & (cls == c)
            if m.sum() > 50:
                L[m] -= float(np.median(L[m]))
        L[valid] += 50.0
        edge = cv2.morphologyEx(cls, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        valid &= ~edge
    if valid.sum() < 100:
        return np.zeros(land.shape, np.float32)
    L = _fill_masked(L, valid, 6.0)
    hp = L - cv2.GaussianBlur(L, (0, 0), 12)
    hp = np.clip(hp, -12, 12) * land
    h, w = hp.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    kt = 2 * np.pi * (kx + ky) / np.sqrt(2.0)          # derivative direction NW -> SE
    k2 = (2 * np.pi) ** 2 * (kx * kx + ky * ky)
    lam, eps = 0.15, 0.01
    S = np.fft.fft2(hp)
    Hf = S * np.conj(1j * kt) / (kt ** 2 + lam * k2 + eps ** 2)
    rel = np.real(np.fft.ifft2(Hf)).astype(np.float32)
    rel -= cv2.GaussianBlur(rel, (0, 0), 25)           # remove drift
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

    d = _dist(land) * s                                # blocks to nearest water (sea, lake or river)
    # the large-scale rise follows the distance to the *sea*: a river or lake
    # inside the island only cuts a local valley, it does not turn the whole
    # inland into lowland
    inland_water = water & ~st.ocean
    d_sea = _dist(~st.ocean) * s if inland_water.any() else d
    dmax = max(float(d_sea.max()), 1.0)
    h = (cfg.beach_height * np.clip(d / 8.0, 0, 1)
         + 0.35 * H * (1 - np.exp(-d_sea / 140.0))
         + 0.25 * H * (d_sea / dmax) ** 0.8).astype(np.float32)
    if inland_water.any():
        # valley: the land sinks towards rivers and lakes over ~25-60 blocks
        width = float(np.clip(0.05 * dmax, 25.0, 60.0))
        valley_depth = 0.35 * (h - cfg.beach_height)
        h = h - valley_depth * np.exp(-d / width)
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
        rel = shading_relief(img, land, cls)
        coast_fade = np.clip(d / 24.0, 0, 1)
        h = h + rel * cfg.relief_strength * 0.15 * H * coast_fade
    if progress:
        progress(0.6)

    h = _masked_blur(h.astype(np.float32), land, cfg.smoothing)

    # flatten towns and roads (they must be walkable)
    flat = np.isin(cls, [CID["urban"], CID["road"], CID["path"]])
    if flat.any():
        strong = _masked_blur(h, land, 6.0)
        w = cv2.GaussianBlur(flat.astype(np.float32), (0, 0), 2.0)
        h = h * (1 - w) + strong * w
    # towns become level plateaus (one height per connected town, soft edge)
    towns = np.isin(cls, [CID["urban"], CID["road"]]) & land
    if towns.any():
        n, lab = cv2.connectedComponents(towns.astype(np.uint8), connectivity=8)
        plateau = h.copy()
        for i in range(1, n):
            m = lab == i
            if m.sum() >= 20:
                plateau[m] = float(np.median(h[m]))
        tw = (towns & (plateau != h)).astype(np.float32)
        tw = np.maximum(tw, cv2.GaussianBlur(tw, (0, 0), 3.0))
        plateau = _masked_blur(np.where(tw > 0, plateau, h).astype(np.float32), land, 1.0)
        h = h * (1 - tw) + plateau * tw
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
