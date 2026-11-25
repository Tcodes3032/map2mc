"""Read a box of blocks back from a generated world's region files.

Used by the 3D block view and the .litematic export, so both show exactly what
Minecraft will load (not a re-computation from the map).
"""
from __future__ import annotations

import os

import numpy as np

from . import anvil
from .nbt import plain


def state_string(entry: dict) -> str:
    """Palette entry {'Name':..., 'Properties':{...}} -> 'minecraft:x[a=b,c=d]'."""
    name = entry["Name"]
    props = entry.get("Properties")
    if not props:
        return name
    props = plain(props)
    return name + "[" + ",".join(f"{k}={props[k]}" for k in sorted(props)) + "]"


def world_bounds(world: str) -> tuple[int, int, int, int] | None:
    """Block bounds (x0, z0, x1, z1 exclusive) covered by region files."""
    reg = os.path.join(world, "region")
    if not os.path.isdir(reg):
        return None
    rs = []
    for f in os.listdir(reg):
        p = f.split(".")
        if len(p) == 4 and p[0] == "r" and p[3] == "mca":
            rs.append((int(p[1]), int(p[2])))
    if not rs:
        return None
    xs, zs = [r[0] for r in rs], [r[1] for r in rs]
    return min(xs) * 512, min(zs) * 512, (max(xs) + 1) * 512, (max(zs) + 1) * 512


def read_box(world: str, x0: int, z0: int, sx: int, sz: int, y_min: int = anvil.MIN_Y, y_max: int = anvil.MAX_Y,
             trim: bool = True, progress=None):
    """Return (vol, names, y0).

    vol   uint16 array [y, z, x] of indices into `names` (0 = air)
    names list of block state strings
    y0    world y of vol[0]
    With trim=True the y range is cut down to the layers that contain blocks.
    """
    H = y_max - y_min + 1
    vol = np.zeros((H, sz, sx), np.uint16)
    names = ["minecraft:air"]
    nidx = {"minecraft:air": 0}
    chunks = [(cx, cz) for cz in range(z0 >> 4, ((z0 + sz - 1) >> 4) + 1)
              for cx in range(x0 >> 4, ((x0 + sx - 1) >> 4) + 1)]
    for k, (cx, cz) in enumerate(chunks):
        if progress and k % 16 == 0:
            progress(k / max(1, len(chunks)))
        path = os.path.join(world, "region", f"r.{cx >> 5}.{cz >> 5}.mca")
        if not os.path.exists(path):
            continue
        ch = anvil.read_region_chunk(path, cx & 31, cz & 31)
        if ch is None:
            continue
        # overlap of this chunk with the box (in chunk-local coordinates)
        ax0, az0 = max(x0, cx * 16), max(z0, cz * 16)
        ax1, az1 = min(x0 + sx, cx * 16 + 16), min(z0 + sz, cz * 16 + 16)
        lx0, lz0 = ax0 - cx * 16, az0 - cz * 16
        for s in ch["sections"]:
            bs = s.get("block_states")
            if bs is None:
                continue
            pal = [state_string(p) for p in bs["palette"]]
            if pal == ["minecraft:air"]:
                continue
            loc = np.array([nidx.setdefault(n, len(nidx)) for n in pal], np.uint16)
            if bs.get("data") is not None:
                bits = max(4, int(np.ceil(np.log2(len(pal)))))
                idx = anvil.unpack_indices(np.asarray(bs["data"]), bits, 4096).reshape(16, 16, 16)
                blk = loc[idx]
            else:
                blk = np.full((16, 16, 16), loc[0], np.uint16)
            sy = int(s["Y"].value if hasattr(s["Y"], "value") else s["Y"]) * 16
            ya, yb = max(sy, y_min), min(sy + 16, y_max + 1)
            if ya >= yb:
                continue
            vol[ya - y_min:yb - y_min, az0 - z0:az1 - z0, ax0 - x0:ax1 - x0] = \
                blk[ya - sy:yb - sy, lz0:lz0 + (az1 - az0), lx0:lx0 + (ax1 - ax0)]
    # names in index order
    names = [None] * len(nidx)
    for n, i in nidx.items():
        names[i] = n
    y0 = y_min
    if trim:
        used = np.nonzero(vol.reshape(H, -1).any(1))[0]
        if used.size:
            vol = vol[used[0]:used[-1] + 1]
            y0 = y_min + int(used[0])
        else:
            vol = vol[:1]
    if progress:
        progress(1.0)
    return np.ascontiguousarray(vol), names, y0


# ------------------------------------------------------------------ LOD (top surface) data
# Per region: map2mc_lod/r.X.Z.npz with
#   top  int16 [512,512]  y of the highest visible block (NONE = no block)
#   blk  uint16 [512,512] index into names
#   names                 block state strings
# Written by the generator; for older worlds computed once from the region files.
LOD_DIR = "map2mc_lod"
NONE = -32768


def _skip_table(names):
    from .blockmesh import BlockInfo
    info = BlockInfo(list(names))
    return info.kind == 0, info.kind == 2          # air, plants (not used as surface)


def top_of_columns(vol: np.ndarray, skip: np.ndarray):
    """vol [y,z,x] ids, skip[id] True = transparent for the LOD.  -> (top index or -1, id)."""
    vis = ~skip[vol]
    has = vis.any(0)
    idx = vol.shape[0] - 1 - np.argmax(vis[::-1], axis=0)
    zz, xx = np.indices(has.shape)
    ids = vol[idx, zz, xx]
    idx = np.where(has, idx, -1)
    return idx, ids


def save_region_lod(world: str, rx: int, rz: int, top: np.ndarray, blk: np.ndarray, names: list[str]):
    d = os.path.join(world, LOD_DIR)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"r.{rx}.{rz}.tmp.npz")
    np.savez_compressed(tmp, top=top.astype(np.int16), blk=blk.astype(np.uint16), names=np.array(names))
    os.replace(tmp, os.path.join(d, f"r.{rx}.{rz}.npz"))


def scan_region_lod(world: str, rx: int, rz: int):
    """Compute the LOD of one region from its .mca file (fallback for older worlds)."""
    top = np.full((512, 512), NONE, np.int16)
    blk = np.zeros((512, 512), np.uint16)
    names: list[str] = []
    nidx: dict[str, int] = {}
    for s in range(0, 512, 16):
        vol, nm, y0 = read_box(world, rx * 512, rz * 512 + s, 512, 16)
        air, plant = _skip_table(nm)
        idx, ids = top_of_columns(vol, air | plant)
        g = np.array([nidx.setdefault(n, len(nidx)) for n in nm], np.uint16)
        top[s:s + 16] = np.where(idx >= 0, idx + y0, NONE)
        blk[s:s + 16] = g[ids]
    names = [None] * len(nidx)
    for n, i in nidx.items():
        names[i] = n
    return top, blk, names


def load_lod(world: str, progress=None, log=None):
    """Whole-world LOD -> (top int16 [Z,X], blk uint16, names, x0, z0)."""
    b = world_bounds(world)
    if b is None:
        return None
    x0, z0, x1, z1 = b
    top = np.full((z1 - z0, x1 - x0), NONE, np.int16)
    blk = np.zeros(top.shape, np.uint16)
    names: list[str] = []
    nidx: dict[str, int] = {}
    regs = [(rx, rz) for rz in range(z0 // 512, z1 // 512) for rx in range(x0 // 512, x1 // 512)
            if os.path.exists(os.path.join(world, "region", f"r.{rx}.{rz}.mca"))]
    for k, (rx, rz) in enumerate(regs):
        p = os.path.join(world, LOD_DIR, f"r.{rx}.{rz}.npz")
        mca = os.path.join(world, "region", f"r.{rx}.{rz}.mca")
        if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(mca) - 5:
            z = np.load(p)
            t, bl, nm = z["top"], z["blk"], [str(n) for n in z["names"]]
        else:
            if log and k == 0:
                log("Erzeuge Übersichtsdaten (einmalig für diese Welt) …")
            t, bl, nm = scan_region_lod(world, rx, rz)
            save_region_lod(world, rx, rz, t, bl, nm)
        g = np.array([nidx.setdefault(n, len(nidx)) for n in nm] or [0], np.uint16)
        oz, ox = rz * 512 - z0, rx * 512 - x0
        top[oz:oz + 512, ox:ox + 512] = t
        blk[oz:oz + 512, ox:ox + 512] = g[bl]
        if progress:
            progress((k + 1) / len(regs))
    names = [None] * len(nidx)
    for n, i in nidx.items():
        names[i] = n
    return top, blk, names or ["minecraft:air"], x0, z0
