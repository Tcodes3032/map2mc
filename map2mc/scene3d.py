"""Builds the 3D scene with levels of detail (like the Voxy mod):

  * detail  – real blocks with textures around the focus point (size chosen in the GUI)
  * middle  – 2×2-block columns up to MID_RADIUS around the focus
  * far     – coarse columns (4…16 blocks) for the whole map, so everything is visible

Each finer level cuts its area out of the coarser one (no overlap / z-fighting).
Scene coordinates: world coordinates minus the focus point (keeps float precision
high where the camera looks); y stays the world y.
"""
from __future__ import annotations

import os

import numpy as np

from . import blockmesh, worldread

MID_RADIUS = 384
MID_CELL = 2
FAR_CELLS = 160_000          # about this many far cells for the whole map


class LodCache:
    """World overview (top surface of every column), loaded once per world."""

    def __init__(self):
        self.key = None
        self.data = None

    def get(self, world: str, progress=None, log=None):
        reg = os.path.join(world, "region")
        key = (os.path.abspath(world), max((os.path.getmtime(os.path.join(reg, f)) for f in os.listdir(reg)),
                                           default=0))
        if self.key != key:
            self.data = worldread.load_lod(world, progress, log)
            self.key = key
        return self.data


def _far_cell(w: int, d: int) -> int:
    c = 4
    while (w // c) * (d // c) > FAR_CELLS and c < 32:
        c *= 2
    return c


def build_scene(world: str, xc: int, zc: int, size: int, cache: LodCache, jar: str | None,
                progress=None, log=None, lod: bool = True) -> dict:
    progress = progress or (lambda f: None)
    top, blk, lnames, wx0, wz0 = cache.get(world, lambda f: progress(f * 0.4), log)
    Zw, Xw = top.shape
    # --- detail box, aligned to the middle grid
    size = max(16, size - size % MID_CELL)
    x0 = wx0 + ((xc - size // 2 - wx0) // MID_CELL) * MID_CELL
    z0 = wz0 + ((zc - size // 2 - wz0) // MID_CELL) * MID_CELL
    fx, fz = x0 + size // 2, z0 + size // 2                  # focus = scene origin
    vol, names, y0 = worldread.read_box(world, x0, z0, size, size, progress=lambda f: progress(0.4 + f * 0.3))
    fl = blockmesh.surface_floor(vol, names)
    atlas = blockmesh.Atlas(jar)
    white = blockmesh.white_tile(atlas)
    colors = blockmesh.tile_colors(atlas, lnames) if lod else None
    opaque, water, img = blockmesh.build_mesh(vol, names, atlas, fl)
    for m in (opaque, water):
        m[:, 0] += x0 - fx
        m[:, 1] += y0
        m[:, 2] += z0 - fz
    rows = img.shape[0] // blockmesh.TILE
    u0, v0, du, dv = atlas.uv(np.array([white]), rows)
    wuv = (float(u0[0] + du / 2), float(v0[0] + dv / 2))
    parts = [opaque]
    far_cell = None
    if lod:
        # --- middle ring
        far_cell = _far_cell(Xw, Zw)
        m0x = ((x0 - MID_RADIUS - wx0) // far_cell) * far_cell
        m0z = ((z0 - MID_RADIUS - wz0) // far_cell) * far_cell
        m1x = -(-(x0 + size + MID_RADIUS - wx0) // far_cell) * far_cell
        m1z = -(-(z0 + size + MID_RADIUS - wz0) // far_cell) * far_cell
        m0x, m0z = max(0, m0x), max(0, m0z)
        m1x, m1z = min(Xw - Xw % far_cell, m1x), min(Zw - Zw % far_cell, m1z)
        if m1x > m0x and m1z > m0z:
            t = top[m0z:m1z, m0x:m1x]
            b = blk[m0z:m1z, m0x:m1x]
            cz, cx = t.shape[0] // MID_CELL, t.shape[1] // MID_CELL
            gx = wx0 + m0x + np.arange(cx) * MID_CELL
            gz = wz0 + m0z + np.arange(cz) * MID_CELL
            inside = ((gz[:, None] >= z0) & (gz[:, None] < z0 + size) & (gx[None, :] >= x0) & (gx[None, :] < x0 + size))
            parts.append(blockmesh.lod_mesh(t, b, colors, MID_CELL, (wx0 + m0x - fx, wz0 + m0z - fz), ~inside, wuv))
        # --- far: whole map
        cz, cx = Zw // far_cell, Xw // far_cell
        gx = np.arange(cx) * far_cell
        gz = np.arange(cz) * far_cell
        inside = ((gz[:, None] >= m0z) & (gz[:, None] < m1z) & (gx[None, :] >= m0x) & (gx[None, :] < m1x))
        parts.append(blockmesh.lod_mesh(top, blk, colors, far_cell, (wx0 - fx, wz0 - fz), ~inside, wuv))
    progress(1.0)
    allv = np.concatenate(parts)
    # camera target: median surface of the detail area
    tt = top[max(0, z0 - wz0):z0 - wz0 + size, max(0, x0 - wx0):x0 - wx0 + size]
    tt = tt[tt > -32768]
    cy = float(np.median(tt)) if tt.size else float(y0 + vol.shape[0] / 2)
    wlist = np.array(["water" in str(n) for n in lnames] + [False], bool)
    water_mask = wlist[np.clip(blk.astype(np.int64), -1, len(lnames))]
    return dict(ground=(top, water_mask, wx0 - fx, wz0 - fz), opaque=allv, water=water, atlas=img, center=(0.0, cy, 0.0), focus=(fx, fz), size=size,
                world_size=float(max(Xw, Zw)), y_min=y0 + fl, y_max=y0 + vol.shape[0] - 1, far_cell=far_cell,
                faces=(len(allv) + len(water)) // 6, detail_faces=(len(opaque) + len(water)) // 6)
