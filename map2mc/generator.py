"""World generation: turns analysed map data into Anvil region files.

The world is built region by region (512×512 blocks, one .mca file each) in
parallel worker processes.  Each worker only reads the small source-resolution
arrays (memory-mapped .npy files) and upsamples the part it needs, so memory
stays bounded even for very large maps.

Resume: every finished region is recorded in map2mc_progress.json inside the
world folder together with a signature of all inputs.  Restarting with the
same inputs skips finished regions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from . import anvil, features, treepack
from .anvil import MIN_Y, BlockRegistry
from .config import CID, N_CLASSES, WATER_CLASSES, Settings, parse_mix

MARGIN = 16           # blocks of context around each region (trees/buildings crossing borders)
REGION = 512


# ------------------------------------------------------------------ geometry
@dataclass
class Geometry:
    """Mapping between source pixels (u, v) and world blocks (X, Z)."""
    scale: float
    cu0: int
    cv0: int
    cu1: int
    cv1: int

    @property
    def width(self) -> int:
        return int(round((self.cu1 - self.cu0) * self.scale))

    @property
    def depth(self) -> int:
        return int(round((self.cv1 - self.cv0) * self.scale))

    @property
    def xmin(self) -> int:
        return -(self.width // 2)

    @property
    def zmin(self) -> int:
        return -(self.depth // 2)

    def world_to_px(self, X, Z):
        u = (np.asarray(X, np.float64) - self.xmin + 0.5) / self.scale + self.cu0 - 0.5
        v = (np.asarray(Z, np.float64) - self.zmin + 0.5) / self.scale + self.cv0 - 0.5
        return u, v

    def px_to_world(self, u, v):
        X = (np.asarray(u, np.float64) - self.cu0) * self.scale + self.xmin
        Z = (np.asarray(v, np.float64) - self.cv0) * self.scale + self.zmin
        return X, Z

    def to_dict(self):
        return dict(scale=self.scale, cu0=self.cu0, cv0=self.cv0, cu1=self.cu1, cv1=self.cv1)


def compute_geometry(land: np.ndarray, cfg: Settings) -> Geometry:
    h, w = land.shape
    if cfg.crop_to_island and land.any():
        ys, xs = np.nonzero(land)
        m = int(math.ceil(cfg.ocean_margin / cfg.blocks_per_pixel))
        return Geometry(float(cfg.blocks_per_pixel), int(max(0, xs.min() - m)), int(max(0, ys.min() - m)),
                        int(min(w, xs.max() + 1 + m)), int(min(h, ys.max() + 1 + m)))
    return Geometry(float(cfg.blocks_per_pixel), 0, 0, int(w), int(h))


# ------------------------------------------------------------------ palette helpers
class Mix:
    """Weighted block mix chosen per column by coordinate hash."""

    def __init__(self, spec: str, reg: BlockRegistry):
        items = parse_mix(spec) or [("minecraft:stone", 1)]
        self.ids = np.array([reg.id(s) for s, _ in items], np.int32)
        self.cum = np.cumsum([w for _, w in items])

    def pick(self, X, Z, seed):
        if len(self.ids) == 1:
            return np.full(np.shape(X), self.ids[0], np.int32)
        r = features.hash2(X, Z, seed) % np.uint32(self.cum[-1])
        return self.ids[np.searchsorted(self.cum, r, side="right")]


# ------------------------------------------------------------------ overlay
class Overlay:
    """Collects feature blocks per chunk; applied in insertion order."""

    def __init__(self, bx0, bz0, bx1, bz1):
        self.box = (bx0, bz0, bx1, bz1)
        self.items: dict[tuple[int, int], list] = {}

    def add(self, X, Y, Z, ids, only_air=False):
        X = np.asarray(X, np.int64)
        Y = np.asarray(Y, np.int64)
        Z = np.asarray(Z, np.int64)
        ids = np.broadcast_to(np.asarray(ids, np.int32), X.shape)
        bx0, bz0, bx1, bz1 = self.box
        keep = (X >= bx0) & (X < bx1) & (Z >= bz0) & (Z < bz1) & (Y >= MIN_Y) & (Y <= anvil.MAX_Y)
        if not keep.any():
            return
        X, Y, Z, ids = X[keep], Y[keep], Z[keep], ids[keep]
        key = ((X >> 4) << 32) ^ ((Z >> 4) & 0xFFFFFFFF)
        order = np.argsort(key, kind="stable")
        X, Y, Z, ids, key = X[order], Y[order], Z[order], ids[order], key[order]
        uk, start = np.unique(key, return_index=True)
        ends = list(start[1:]) + [len(key)]
        for s, e in zip(start, ends):
            ck = (int(X[s] >> 4), int(Z[s] >> 4))
            self.items.setdefault(ck, []).append((X[s:e] & 15, Y[s:e], Z[s:e] & 15, ids[s:e], only_air))


# ------------------------------------------------------------------ region worker
def _load(path):
    return np.load(path, mmap_mode="r")


def generate_region(job: dict) -> dict:
    """Build one region file. Runs in a worker process."""
    t0 = time.time()
    d = job["data_dir"]
    cfg = Settings.from_dict(job["settings"])
    geo = Geometry(**job["geometry"])
    palette = job["palette"]
    rx, rz = job["rx"], job["rz"]
    WL = cfg.water_level
    seed = cfg.seed & 0x7FFFFFFF

    classes_src = _load(os.path.join(d, "classes.npy"))
    height_src = _load(os.path.join(d, "height.npy"))
    lake_src = _load(os.path.join(d, "lake.npy"))
    bridge_src = _load(os.path.join(d, "bridge.npy"))
    trees_src = np.load(os.path.join(d, "trees_world.npy"))
    if trees_src.ndim != 2 or trees_src.shape[1] < 7:      # data from an older analysis: no colours
        trees_src = np.zeros((0, 7), np.float32)
    image_src = _load(os.path.join(d, "image.npy"))
    with open(os.path.join(d, "buildings_world.json"), encoding="utf-8") as f:
        buildings = json.load(f)

    # --- area = region (clipped to the map extent, chunk aligned) + margin
    rx0, rz0 = rx * REGION, rz * REGION
    cx_min = max(rx0 >> 4, geo.xmin >> 4)
    cx_max = min((rx0 + REGION - 1) >> 4, (geo.xmin + geo.width - 1) >> 4)
    cz_min = max(rz0 >> 4, geo.zmin >> 4)
    cz_max = min((rz0 + REGION - 1) >> 4, (geo.zmin + geo.depth - 1) >> 4)
    bx0, bx1 = cx_min * 16, (cx_max + 1) * 16
    bz0, bz1 = cz_min * 16, (cz_max + 1) * 16
    ax0, az0 = bx0 - MARGIN, bz0 - MARGIN
    AW, AD = bx1 - bx0 + 2 * MARGIN, bz1 - bz0 + 2 * MARGIN
    Xs = np.arange(ax0, ax0 + AW)
    Zs = np.arange(az0, az0 + AD)
    XX, ZZ = np.meshgrid(Xs, Zs)

    # --- sample source arrays (bilinear) within a source window
    u, v = geo.world_to_px(Xs, Zs)
    H_src, W_src = classes_src.shape
    su0 = int(max(0, math.floor(u.min()) - 2)); su1 = int(min(W_src, math.ceil(u.max()) + 3))
    sv0 = int(max(0, math.floor(v.min()) - 2)); sv1 = int(min(H_src, math.ceil(v.max()) + 3))
    mapx = np.broadcast_to((u - su0).astype(np.float32)[None, :], (AD, AW)).copy()
    mapy = np.broadcast_to((v - sv0).astype(np.float32)[:, None], (AD, AW)).copy()

    def sample(arr, interp=cv2.INTER_LINEAR):
        win = np.ascontiguousarray(arr[sv0:sv1, su0:su1]).astype(np.float32)
        return cv2.remap(win, mapx, mapy, interp, borderMode=cv2.BORDER_REPLICATE)

    cls_win = np.ascontiguousarray(classes_src[sv0:sv1, su0:su1])
    best = np.full((AD, AW), -1.0, np.float32)
    cls = np.zeros((AD, AW), np.uint8)
    for c in np.unique(cls_win):
        m = cv2.remap((cls_win == c).astype(np.float32), mapx, mapy, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        better = m > best
        best[better] = m[better]
        cls[better] = c
    h = sample(height_src)
    lake = np.round(sample(lake_src, cv2.INTER_NEAREST)).astype(np.int32)
    bridge = sample(bridge_src) > 0.5

    water_lut = np.zeros(N_CLASSES, bool)
    water_lut[list(WATER_CLASSES)] = True
    is_water = water_lut[cls]
    flat_cls = np.isin(cls, [CID["urban"], CID["road"], CID["path"]])

    # --- small natural variation (deterministic, never changes the layout)
    if cfg.detail_noise > 0:
        n = value_noise_layers(XX, ZZ, seed)
        damp = np.clip(h / 3.0, 0, 1) * np.where(flat_cls, 0.15, 1.0)
        h = h + n * cfg.detail_noise * damp * (~is_water)

    surf = (WL + np.round(h)).astype(np.int32)
    water_top = np.where(is_water, WL + lake, -10_000).astype(np.int32)
    surf = np.where(is_water, np.minimum(surf, water_top - 1), np.maximum(surf, WL))
    surf = np.clip(surf, MIN_Y + 2, anvil.MAX_Y - 40)

    reg = BlockRegistry()
    AIR = reg.AIR
    ov = Overlay(bx0, bz0, bx1, bz1)

    # --- buildings: flatten ground and queue blocks
    building_cells = np.zeros((AD, AW), bool)
    house_jobs = []
    for b in buildings:
        bb = b["bbox"]                                  # world [x0, z0, x1, z1]
        if bb[2] < ax0 or bb[0] > ax0 + AW or bb[3] < az0 or bb[1] > az0 + AD or b["kind"] == "none":
            continue
        poly = np.array(b["polygon_world"], np.float64)
        fx0, fz0 = int(bb[0]) - 2, int(bb[1]) - 2
        fw, fd = int(bb[2]) - fx0 + 3, int(bb[3]) - fz0 + 3
        fp = features.footprint_mask(poly, fx0, fz0, fw, fd)
        if fp.sum() < 4:
            continue
        # local area slices
        lx0, lz0 = fx0 - ax0, fz0 - az0
        sx0, sz0 = max(0, lx0), max(0, lz0)
        sx1, sz1 = min(AW, lx0 + fw), min(AD, lz0 + fd)
        if sx1 <= sx0 or sz1 <= sz0:
            continue
        sub = fp[sz0 - lz0:sz1 - lz0, sx0 - lx0:sx1 - lx0]
        if b["kind"] == "house":
            ring = cv2.dilate(fp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            rsub = ring[sz0 - lz0:sz1 - lz0, sx0 - lx0:sx1 - lx0]
            region_surf = surf[sz0:sz1, sx0:sx1]
            region_surf[rsub] = b["base_y"]
            wsub = water_top[sz0:sz1, sx0:sx1]
            wsub[rsub] = -10_000
            building_cells[sz0:sz1, sx0:sx1] |= rsub
        house_jobs.append((b, fp, fx0, fz0))

    # --- column blocks
    mixes = {}
    sub_depth = np.zeros(N_CLASSES, np.int32)
    for i, key in enumerate(CID):
        p = palette.get(key) or palette["grass"]
        mixes[i] = (Mix(p["surface"], reg), Mix(p["subsurface"], reg))
        sub_depth[i] = int(p.get("sub_depth", 3))
    top_id = np.zeros((AD, AW), np.int32)
    sub_id = np.zeros((AD, AW), np.int32)
    for c in np.unique(cls):
        m = cls == c
        top_id[m] = mixes[int(c)][0].pick(XX[m], ZZ[m], seed + 1)
        sub_id[m] = mixes[int(c)][1].pick(XX[m], ZZ[m], seed + 2)
    town_floor = Mix(palette["urban"]["surface"], reg)
    if building_cells.any():
        top_id[building_cells] = town_floor.pick(XX[building_cells], ZZ[building_cells], seed + 3)
    sdepth = sub_depth[cls]

    STONE, DEEP, BEDROCK, WATER = (reg.id("minecraft:stone"), reg.id("minecraft:deepslate[axis=y]"),
                                   reg.id("minecraft:bedrock"), reg.id("minecraft:water[level=0]"))

    # --- vegetation (grass, flowers) on grass-topped land
    veg_p = np.array([float((palette.get(k) or {}).get("vegetation", 0)) for k in CID], np.float32)
    grass_ids = {reg.id("minecraft:grass_block")}
    r = features.rand01(XX, ZZ, seed + 7)
    can = (~is_water) & ~building_cells & ~bridge & np.isin(top_id, list(grass_ids))
    plant = can & (r < veg_p[cls])
    if plant.any():
        flowers = ["minecraft:dandelion", "minecraft:poppy", "minecraft:azure_bluet", "minecraft:oxeye_daisy",
                   "minecraft:cornflower"]
        r2 = features.rand01(XX, ZZ, seed + 8)
        ids = np.full(XX.shape, reg.id("minecraft:short_grass"), np.int32)
        fl = r2 < 0.12
        ids[fl] = np.array([reg.id(f) for f in flowers])[(r2[fl] * 1000).astype(int) % len(flowers)]
        petals = (cls == CID["blossom"]) & (r2 > 0.6)
        ids[petals] = reg.id("minecraft:pink_petals[facing=north,flower_amount=3]")
        ov.add(XX[plant], surf[plant] + 1, ZZ[plant], ids[plant], only_air=True)

    # --- trees
    tree_kind = [str((palette.get(k) or {}).get("tree", "none")) for k in CID]
    tree_dens = np.array([float((palette.get(k) or {}).get("tree_density", 0)) for k in CID]) * cfg.tree_density
    crowns = np.zeros((AD, AW), np.uint8)
    blocked = is_water | building_cells | bridge | np.isin(cls, [CID["road"], CID["urban"], CID["path"]])

    lib = treepack.load_library(treepack.resolve(cfg.tree_pack) or "") if cfg.use_tree_pack else None
    lib_ids: dict = {}

    def map_rgb(X, Z):
        """Colour of the map at a world position (3x3 px median)."""
        pu, pv = geo.world_to_px(X, Z)
        pu, pv = int(np.clip(round(float(pu)), 1, W_src - 2)), int(np.clip(round(float(pv)), 1, H_src - 2))
        return tuple(int(x) for x in np.median(np.asarray(image_src[pv - 1:pv + 2, pu - 1:pu + 2]).reshape(-1, 3), 0))

    def place_pack_tree(X, Z, lx, lz, radius, kind, material):
        """Tree from the tree pack, dressed in its own crown colour; False if nothing suits."""
        shape = features.SHAPE_OF_KIND.get(kind, "broadleaf")
        h = int(features.hash2(X, Z, seed + 41))
        i = lib.choose(shape, float(radius), h)
        if i is None:
            return False
        rr = int(np.ceil(lib.radius[i]))
        if building_cells[max(0, lz - rr):lz + rr + 1, max(0, lx - rr):lx + rr + 1].any():
            return True          # crown would grow into a house: leave the spot free
        rot = (h >> 8) & 3
        dx, dy, dz, pid, states, leaf = lib.blocks_of(i, rot)
        key = (i, rot, material)
        ids = lib_ids.get(key)
        if ids is None:
            crown = features.crown_state(material) if material else None
            wood = features.log_for_material(material) if material else None
            ids = np.array([reg.id(treepack.redress(s, crown, wood)) for s in states], np.int32)
            lib_ids[key] = ids
        y0 = int(surf[lz, lx]) + 1
        ov.add(X + dx[~leaf], y0 + dy[~leaf], Z + dz[~leaf], ids[pid[~leaf]], only_air=False)
        ov.add(X + dx[leaf], y0 + dy[leaf], Z + dz[leaf], ids[pid[leaf]], only_air=True)
        cv2.circle(crowns, (int(lx), int(lz)), int(lib.radius[i]) + 1, 1, -1)
        return True

    def place_tree(X, Z, radius, kind, variant, rgb=None):
        """kind = shape/wood type; rgb = crown colour on the map (None: the kind's own leaves)."""
        lx, lz = X - ax0, Z - az0
        if not (0 <= lx < AW and 0 <= lz < AD) or blocked[lz, lx]:
            return False
        material = features.crown_material(rgb) if rgb is not None else None
        if lib is not None and place_pack_tree(X, Z, lx, lz, radius, kind, material):
            return True
        radius = min(int(radius), 7)
        shp = features.tree_shape(kind, int(radius), int(variant))
        if shp is None:
            return False
        dx, dy, dz, leaf = shp
        y0 = int(surf[lz, lx]) + 1
        log_s, leaf_s = features.WOOD[kind]
        if material:
            leaf_s = features.crown_state(material)
            w = features.log_for_material(material)
            if w:
                log_s = treepack.redress(log_s, None, w)
        lid, fid = reg.id(log_s), reg.id(leaf_s)
        ov.add(X + dx[~leaf], y0 + dy[~leaf], Z + dz[~leaf], lid, only_air=False)
        ov.add(X + dx[leaf], y0 + dy[leaf], Z + dz[leaf], fid, only_air=True)
        cv2.circle(crowns, (int(lx), int(lz)), int(radius) + 1, 1, -1)
        return True

    if cfg.detect_single_trees and len(trees_src):
        T = trees_src
        sel = (T[:, 0] >= ax0) & (T[:, 0] < ax0 + AW) & (T[:, 1] >= az0) & (T[:, 1] < az0 + AD)
        for X, Z, rad, c, R, G, B in T[sel]:
            kind = tree_kind[int(c)]
            if kind == "none":
                kind = "auto" if int(c) in (CID["grass"], CID["forest"]) else "none"
            if kind == "none":
                continue
            rgb = (int(R), int(G), int(B))          # every detected tree keeps its own colour
            if kind == "auto":
                kind = features.kind_from_rgb(rgb)
            X, Z = int(X), int(Z)
            if features.rand01(X, Z, seed + 11) > min(1.0, cfg.tree_density):
                continue
            place_tree(X, Z, int(np.clip(round(rad), 2, 12)), kind, int(features.hash2(X, Z, seed) % 4), rgb)

    sp = max(3, int(cfg.tree_spacing))
    gx0, gx1 = ax0 // sp, (ax0 + AW) // sp + 1
    gz0, gz1 = az0 // sp, (az0 + AD) // sp + 1
    GX, GZ = np.meshgrid(np.arange(gx0, gx1), np.arange(gz0, gz1))
    jx = (features.hash2(GX, GZ, seed + 21) % sp).astype(np.int64)
    jz = (features.hash2(GX, GZ, seed + 22) % sp).astype(np.int64)
    TX, TZ = GX * sp + jx, GZ * sp + jz
    lx, lz = TX - ax0, TZ - az0
    ok = (lx >= 0) & (lx < AW) & (lz >= 0) & (lz < AD)
    use_filler = cfg.filler_trees or not cfg.detect_single_trees
    for X, Z, ix, iz in zip(TX[ok], TZ[ok], lx[ok], lz[ok]):
        if not use_filler:
            break
        c = int(cls[iz, ix])
        if tree_kind[c] == "none" or tree_dens[c] <= 0 or crowns[iz, ix]:
            continue
        if features.rand01(X, Z, seed + 23) >= tree_dens[c]:
            continue
        rad = 2 + int(features.hash2(X, Z, seed + 24) % 3)
        kind = tree_kind[c]
        rgb = None
        if kind == "auto":
            rgb = map_rgb(X, Z)
            kind = features.kind_from_rgb(rgb)
        place_tree(int(X), int(Z), rad, kind, int(features.hash2(X, Z, seed + 25) % 4), rgb)

    # --- bridges
    if bridge.any():
        deck_y = WL + 2
        bm = bridge & (surf < deck_y)
        pad = np.pad(bm, 1)
        nb_water = np.zeros_like(bm)
        land_pad = np.pad(~is_water | building_cells, 1)
        for dz_, dx_ in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nbm = pad[1 + dz_:1 + dz_ + AD, 1 + dx_:1 + dx_ + AW]
            nland = land_pad[1 + dz_:1 + dz_ + AD, 1 + dx_:1 + dx_ + AW]
            nb_water |= ~nbm & ~nland
        edge = bm & nb_water
        ov.add(XX[bm], np.full(bm.sum(), deck_y), ZZ[bm], reg.id(cfg.bridge_deck_block))
        ov.add(XX[edge], np.full(edge.sum(), deck_y + 1), ZZ[edge], reg.id(cfg.bridge_fence_block))
        pil = bm & (XX % 6 == 0) & (ZZ % 6 == 0)
        pid = reg.id(cfg.bridge_pillar_block)
        for X, Z, s0 in zip(XX[pil], ZZ[pil], surf[pil]):
            ys = np.arange(s0 + 1, deck_y)
            ov.add(np.full(len(ys), X), ys, np.full(len(ys), Z), pid)

    # --- houses / platforms (last, they override vegetation)
    for b, fp, fx0, fz0 in house_jobs:
        if b["kind"] == "house":
            parts = features.house_blocks(
                fp, int(b["base_y"]), int(b["storeys"]), cfg.storey_height,
                b.get("roof_block") or "minecraft:smooth_stone", b.get("wall_block") or cfg.wall_block,
                cfg.floor_block, cfg.window_block, cfg.building_lights)
        else:
            parts = features.platform_blocks(fp, WL + 1, cfg.bridge_deck_block, cfg.bridge_fence_block)
        for xs, ys, zs, state in parts:
            ov.add(xs + fx0, ys, zs + fz0, reg.id(state))

    # --- biomes
    biome_names = []
    biome_idx_of_class = np.zeros(N_CLASSES, np.int32)
    for i, key in enumerate(CID):
        bname = (palette.get(key) or {}).get("biome", "minecraft:plains")
        if bname not in biome_names:
            biome_names.append(bname)
        biome_idx_of_class[i] = biome_names.index(bname)

    # --- assemble chunks
    chunks = {}
    from .worldread import NONE as LOD_NONE, _skip_table, save_region_lod, top_of_columns
    air_t, plant_t = _skip_table(reg.states)
    lod_skip = air_t | plant_t
    lod_top = np.full((512, 512), LOD_NONE, np.int16)
    lod_blk = np.zeros((512, 512), np.uint16)
    for cz in range(cz_min, cz_max + 1):
        for cx in range(cx_min, cx_max + 1):
            lx0, lz0 = cx * 16 - ax0, cz * 16 - az0
            S = surf[lz0:lz0 + 16, lx0:lx0 + 16]
            Wt = water_top[lz0:lz0 + 16, lx0:lx0 + 16]
            items = ov.items.get((cx, cz), [])
            ymax = max(int(S.max()), int(Wt.max()))
            for it in items:
                ymax = max(ymax, int(it[1].max()))
            Hc = ((ymax - MIN_Y) // 16 + 2) * 16
            Hc = min(Hc, anvil.MAX_Y - MIN_Y + 1)
            Y = np.arange(MIN_Y, MIN_Y + Hc)[:, None, None]
            S3 = S[None]
            arr = np.full((Hc, 16, 16), AIR, np.int32)
            solid = Y <= S3
            arr[solid & (Y >= 0)] = STONE
            arr[solid & (Y < 0)] = DEEP
            subm = (Y < S3) & (Y >= S3 - sdepth[lz0:lz0 + 16, lx0:lx0 + 16][None])
            arr = np.where(subm, sub_id[lz0:lz0 + 16, lx0:lx0 + 16][None], arr)
            arr = np.where(Y == S3, top_id[lz0:lz0 + 16, lx0:lx0 + 16][None], arr)
            arr = np.where((Y > S3) & (Y <= Wt[None]), WATER, arr)
            arr[0] = BEDROCK
            for lx, ys, lz, ids, only_air in items:
                yi = ys - MIN_Y
                ok = (yi >= 0) & (yi < Hc)
                lx, yi, lz, ids = lx[ok], yi[ok], lz[ok], ids[ok]
                if only_air:
                    free = arr[yi, lz, lx] == AIR
                    lx, yi, lz, ids = lx[free], yi[free], lz[free], ids[free]
                arr[yi, lz, lx] = ids
            # grass under solid blocks would turn to dirt anyway; keep as is
            bc = cls[lz0 + 2:lz0 + 16:4, lx0 + 2:lx0 + 16:4]
            bio = biome_idx_of_class[bc]
            chunks[(cx & 31, cz & 31)] = anvil.encode_chunk(cx, cz, arr, MIN_Y, reg, bio, biome_names)
            ti, tid = top_of_columns(arr, lod_skip)
            oz, ox = (cz & 31) * 16, (cx & 31) * 16
            lod_top[oz:oz + 16, ox:ox + 16] = np.where(ti >= 0, ti + MIN_Y, LOD_NONE)
            lod_blk[oz:oz + 16, ox:ox + 16] = tid

    os.makedirs(os.path.join(job["world_dir"], "region"), exist_ok=True)
    path = os.path.join(job["world_dir"], "region", f"r.{rx}.{rz}.mca")
    anvil.write_region(path, chunks)
    save_region_lod(job["world_dir"], rx, rz, lod_top, lod_blk, list(reg.states))   # overview for the 3D view
    return {"rx": rx, "rz": rz, "chunks": len(chunks), "seconds": round(time.time() - t0, 2)}


def value_noise_layers(XX, ZZ, seed):
    return (features.value_noise(XX, ZZ, 24.0, seed + 31) * 0.65
            + features.value_noise(XX, ZZ, 9.0, seed + 32) * 0.35)


# ------------------------------------------------------------------ orchestration
def region_jobs(geo: Geometry):
    cx0, cx1 = geo.xmin >> 4, (geo.xmin + geo.width - 1) >> 4
    cz0, cz1 = geo.zmin >> 4, (geo.zmin + geo.depth - 1) >> 4
    out = []
    for rz in range(cz0 >> 5, (cz1 >> 5) + 1):
        for rx in range(cx0 >> 5, (cx1 >> 5) + 1):
            out.append((rx, rz))
    return out


def signature(data_dir: str, cfg: Settings, palette: dict, geo: Geometry) -> str:
    hsh = hashlib.sha1()
    for name in ("classes.npy", "height.npy", "lake.npy", "bridge.npy", "trees_world.npy", "buildings_world.json"):
        with open(os.path.join(data_dir, name), "rb") as f:
            hsh.update(hashlib.sha1(f.read()).digest())
    hsh.update(json.dumps([cfg.to_dict(), palette, geo.to_dict()], sort_keys=True).encode())
    return hsh.hexdigest()


def generate_world(data_dir: str, world_dir: str, cfg: Settings, palette: dict, geo: Geometry,
                   spawn: tuple[int, int, int], progress=None, log=None, cancel=None,
                   only_regions: list | None = None, report: dict | None = None) -> dict:
    """Generate all regions (in parallel), then write level.dat and the report.

    progress(fraction), log(str) are optional callbacks; cancel is an object with
    is_set() (e.g. threading.Event).
    """
    from .level import write_level_dat

    log = log or (lambda s: None)
    os.makedirs(os.path.join(world_dir, "region"), exist_ok=True)
    sig = signature(data_dir, cfg, palette, geo)
    prog_path = os.path.join(world_dir, "map2mc_progress.json")
    done = set()
    if os.path.exists(prog_path):
        try:
            with open(prog_path, encoding="utf-8") as f:
                pj = json.load(f)
            if pj.get("signature") == sig:
                done = {tuple(x) for x in pj.get("done", [])}
                if done:
                    log(f"Fortsetzen: {len(done)} Regionen bereits fertig.")
        except (OSError, ValueError):
            pass

    jobs = region_jobs(geo)
    if only_regions:
        jobs = [j for j in jobs if list(j) in [list(r) for r in only_regions]]
    todo = [j for j in jobs if j not in done
            or not os.path.exists(os.path.join(world_dir, "region", f"r.{j[0]}.{j[1]}.mca"))]
    total = len(jobs)
    log(f"Welt {geo.width}×{geo.depth} Blöcke, {total} Regionen ({len(todo)} zu erzeugen).")

    def save_progress():
        tmp = prog_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"signature": sig, "done": sorted(list(done))}, f)
        os.replace(tmp, prog_path)

    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
    base = dict(data_dir=data_dir, world_dir=world_dir, settings=cfg.to_dict(), palette=palette,
                geometry=geo.to_dict())
    finished = total - len(todo)
    if progress:
        progress(finished / max(total, 1))
    if todo:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            futs = {ex.submit(generate_region, dict(base, rx=rx, rz=rz)): (rx, rz) for rx, rz in todo}
            try:
                for fut in as_completed(futs):
                    if cancel is not None and cancel.is_set():
                        for f in futs:
                            f.cancel()
                        log("Abgebrochen – bereits fertige Regionen bleiben erhalten (Fortsetzen möglich).")
                        save_progress()
                        return {"cancelled": True}
                    res = fut.result()
                    done.add((res["rx"], res["rz"]))
                    finished += 1
                    save_progress()
                    log(f"Region r.{res['rx']}.{res['rz']}: {res['chunks']} Chunks in {res['seconds']} s")
                    if progress:
                        progress(finished / total)
            except BaseException:
                save_progress()
                raise

    write_level_dat(world_dir, cfg.world_name, spawn, cfg.seed, cfg.water_level, game_type=cfg.game_mode)
    if report is not None:
        with open(os.path.join(world_dir, "map2mc_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    log("Welt fertig: " + world_dir)
    return {"cancelled": False, "regions": total}
