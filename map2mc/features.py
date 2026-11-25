"""3D features: deterministic hashing/noise, trees, buildings, bridges.

Every random-looking choice is derived from a hash of the *world coordinate*,
so results are identical no matter which worker process builds which region,
and features crossing region borders line up perfectly.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import math
import zlib
import numpy as np


# ------------------------------------------------------------------ hashing / noise
def hash2(x, z, seed: int = 0) -> np.ndarray:
    """Integer hash of coordinates -> uint32 array."""
    shape = np.shape(x)
    x = np.atleast_1d(np.asarray(x, dtype=np.int64)).astype(np.uint64)
    z = np.broadcast_to(np.atleast_1d(np.asarray(z, dtype=np.int64)), x.shape).astype(np.uint64)
    m = np.uint64(0xFFFFFFFF)
    h = (x * np.uint64(374761393) + z * np.uint64(668265263) + np.uint64(seed * 1442695041 & 0xFFFFFFFF)) & m
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & m
    h = h ^ (h >> np.uint64(16))
    return h.astype(np.uint32).reshape(shape)


def rand01(x, z, seed: int = 0) -> np.ndarray:
    return hash2(x, z, seed).astype(np.float64) / 4294967296.0


def value_noise(X: np.ndarray, Z: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] on world coordinates."""
    xf, zf = X / scale, Z / scale
    x0, z0 = np.floor(xf), np.floor(zf)
    fx, fz = xf - x0, zf - z0
    fx = fx * fx * (3 - 2 * fx)
    fz = fz * fz * (3 - 2 * fz)
    x0 = x0.astype(np.int64)
    z0 = z0.astype(np.int64)
    a = rand01(x0, z0, seed)
    b = rand01(x0 + 1, z0, seed)
    c = rand01(x0, z0 + 1, seed)
    d = rand01(x0 + 1, z0 + 1, seed)
    v = (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz
    return (v * 2 - 1).astype(np.float32)


# ------------------------------------------------------------------ trees
WOOD = {
    "oak": ("minecraft:oak_log[axis=y]", "minecraft:oak_leaves[persistent=true]"),
    "birch": ("minecraft:birch_log[axis=y]", "minecraft:birch_leaves[persistent=true]"),
    "spruce": ("minecraft:spruce_log[axis=y]", "minecraft:spruce_leaves[persistent=true]"),
    "dark_oak": ("minecraft:dark_oak_log[axis=y]", "minecraft:dark_oak_leaves[persistent=true]"),
    "cherry": ("minecraft:cherry_log[axis=y]", "minecraft:cherry_leaves[persistent=true]"),
    "acacia": ("minecraft:acacia_log[axis=y]", "minecraft:acacia_leaves[persistent=true]"),
    "jungle": ("minecraft:jungle_log[axis=y]", "minecraft:jungle_leaves[persistent=true]"),
    "bush": ("minecraft:oak_log[axis=y]", "minecraft:oak_leaves[persistent=true]"),
    "pale_oak": ("minecraft:pale_oak_log[axis=y]", "minecraft:pale_oak_leaves[persistent=true]"),
    "azalea": ("minecraft:oak_log[axis=y]", "minecraft:azalea_leaves[persistent=true]"),
    "mangrove": ("minecraft:mangrove_log[axis=y]", "minecraft:mangrove_leaves[persistent=true]"),
    "palm": ("minecraft:jungle_log[axis=y]", "minecraft:jungle_leaves[persistent=true]"),
}

# Approximate on-screen colour of each leaf block (texture x default foliage tint).
LEAF_RGB = {
    "oak": (76, 112, 38), "birch": (96, 126, 62), "spruce": (58, 96, 72), "dark_oak": (52, 88, 30),
    "jungle": (54, 128, 28), "acacia": (104, 108, 34), "cherry": (232, 172, 205), "pale_oak": (138, 146, 130),
    "azalea": (96, 122, 48), "mangrove": (84, 112, 30),
}
# tree shape used for each leaf type
SHAPE_OF = {"pale_oak": "dark_oak", "azalea": "oak", "mangrove": "oak", "palm": "jungle"}
_leaf_names = list(LEAF_RGB)
_leaf_lab = None


def kind_from_rgb(rgb) -> str:
    """Leaf type for a tree crown colour from the map.

    Map crowns are drawn lighter and more saturated than leaves look in game, so
    instead of a plain nearest-colour match we decide by hue (CIE-Lab) first and
    use lightness only to separate dark / light variants:
      pink -> cherry, orange/red -> acacia, grey -> pale_oak / spruce,
      dark -> dark_oak, blue-green/teal -> spruce, yellow-green -> birch/acacia,
      plain green -> oak (light: birch).
    """
    lab = cv2.cvtColor(np.array([[rgb]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    L, a, b = lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360
    if chroma < 7:
        return "pale_oak" if L > 55 else "spruce"
    if a > 10 and (hue < 60 or hue > 300):
        return "cherry" if hue < 35 or hue > 300 else "acacia"
    if L < 36:
        return "dark_oak"
    if hue >= 150:
        return "spruce"
    if hue < 103:
        return "birch" if L > 58 else "acacia"
    return "birch" if L > 62 else "oak"


@lru_cache(maxsize=512)
def tree_shape(kind: str, radius: int, variant: int):
    """Return (dx, dy, dz, is_leaf) int arrays describing one tree relative to its root."""
    rng = np.random.default_rng(zlib.crc32(f"{kind}|{radius}|{variant}".encode()))  # stable across processes
    r = max(1, radius)
    logs, leaves = [], []

    def ellipsoid(cy, rx, ry, keep=0.85):
        rr = int(np.ceil(max(rx, ry)))
        g = np.mgrid[-rr:rr + 1, -rr:rr + 1, -rr:rr + 1]
        dy, dz, dx = g[0], g[1], g[2]
        e = (dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rx) ** 2
        edge = e > 0.65
        m = (e <= 1.0) & (~edge | (rng.random(e.shape) < keep))
        return np.stack([dx[m], dy[m] + cy, dz[m]], 1)

    def trunk(h, wide=False):
        pts = [(0, y, 0) for y in range(h)]
        if wide:
            pts += [(1, y, 0) for y in range(h)] + [(0, y, 1) for y in range(h)] + [(1, y, 1) for y in range(h)]
        return np.array(pts, int).reshape(-1, 3)

    kind = SHAPE_OF.get(kind, kind)
    if kind == "bush":
        t = 1
        logs.append(trunk(1))
        leaves.append(ellipsoid(1, min(r, 2) + 0.3, 1.3, 0.9))
    elif kind in ("oak", "birch"):
        if kind == "birch":
            r = min(r, 3)
        t = int(max(4, round(r * 1.2) + 3 + variant % 2))
        wide = kind == "oak" and r >= 5
        logs.append(trunk(t, wide))
        leaves.append(ellipsoid(t - 1, r + 0.4, max(2.0, r * 0.75), 0.8))
    elif kind == "dark_oak":
        t = int(max(5, round(r * 0.9) + 3))
        logs.append(trunk(t, True))
        leaves.append(ellipsoid(t, r + 0.6, max(2.0, r * 0.55), 0.85))
    elif kind == "cherry":
        t = int(max(4, round(r * 0.8) + 3))
        logs.append(trunk(t, r >= 5))
        leaves.append(ellipsoid(t, r + 0.8, max(2.0, r * 0.5), 0.85))
    elif kind == "acacia":
        t = int(max(5, round(r * 0.8) + 4))
        logs.append(trunk(t))
        leaves.append(ellipsoid(t, r + 0.5, 1.2, 0.9))
    elif kind == "jungle":
        t = int(max(8, r * 2 + 6))
        logs.append(trunk(t, r >= 4))
        leaves.append(ellipsoid(t - 1, r + 0.5, max(2.0, r * 0.6), 0.85))
    elif kind == "spruce":
        t = int(max(6, r * 2.4 + 3 + variant % 3))
        logs.append(trunk(t - 1, r >= 5))
        layers = []
        for y in range(2, t + 1):
            frac = (y - 2) / max(1, t - 2)
            lr = max(0.6, (1 - frac) * (r + 0.5))
            if (y % 2 == 0) and y < t - 1:
                lr *= 0.7
            g = np.mgrid[-r - 1:r + 2, -r - 1:r + 2]
            m = (g[0] ** 2 + g[1] ** 2) <= lr * lr
            layers.append(np.stack([g[1][m], np.full(m.sum(), y), g[0][m]], 1))
        layers.append(np.array([[0, t, 0], [0, t + 1, 0]]))
        leaves.append(np.concatenate(layers))
    else:
        return None
    L = np.concatenate(logs) if logs else np.zeros((0, 3), int)
    F = np.concatenate(leaves) if leaves else np.zeros((0, 3), int)
    # remove leaves that coincide with logs
    if len(F):
        key_l = set(map(tuple, L.tolist()))
        F = np.array([p for p in F.tolist() if tuple(p) not in key_l], int).reshape(-1, 3)
    pts = np.concatenate([L, F])
    is_leaf = np.concatenate([np.zeros(len(L), bool), np.ones(len(F), bool)])
    return pts[:, 0], pts[:, 1], pts[:, 2], is_leaf


# ------------------------------------------------------------------ buildings
def footprint_mask(poly_world: np.ndarray, x0: int, z0: int, w: int, h: int) -> np.ndarray:
    """Rasterize a polygon (world block coords) into a (h, w) bool mask with origin (x0, z0)."""
    m = np.zeros((h, w), np.uint8)
    pts = np.round((poly_world - [x0, z0]) * 16).astype(np.int32)
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def house_blocks(fp: np.ndarray, base: int, storeys: int, sh: int, roof: str, wall: str,
                 floor: str, window: str, lights: bool):
    """Build a walkable house on footprint `fp` ((h,w) bool, local coords).

    Returns list of (xs, ys, zs, state) in local coords (x = column, z = row).
    Layout: solid floor, walls with windows, door on the south side, inner
    floors connected by a ladder, flat roof, optional lanterns.
    """
    out = []
    if fp.sum() < 4:
        return out
    k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    inner = cv2.erode(fp.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    perim = fp & ~inner
    top = base + storeys * sh                       # last interior y is top-1? -> walls up to top
    zs_f, xs_f = np.nonzero(fp)
    zs_p, xs_p = np.nonzero(perim)
    zs_i, xs_i = np.nonzero(inner)

    def add(xs, zs, y, state):
        if len(xs):
            out.append((xs, np.full(len(xs), y), zs, state))

    # clear the whole volume first (air inside), then floor
    for y in range(base + 1, top + 1):
        add(xs_f, zs_f, y, "minecraft:air")
    add(xs_f, zs_f, base, floor)
    # walls with windows
    pad = np.pad(perim, 1)
    horiz = pad[1:-1, :-2] | pad[1:-1, 2:]
    vert = pad[:-2, 1:-1] | pad[2:, 1:-1]
    corner = (horiz & vert)[zs_p, xs_p]
    for s in range(storeys):
        fy = base + s * sh
        for y in range(fy + 1, fy + sh + 1):
            add(xs_p, zs_p, y, wall)
        if sh >= 3:
            win = (~corner) & (((xs_p + zs_p) % 3) == 1)
            add(xs_p[win], zs_p[win], fy + 2, window)
            if sh >= 4:
                add(xs_p[win], zs_p[win], fy + 3, window)
        if s > 0:
            add(xs_i, zs_i, fy, floor)
    add(xs_f, zs_f, top + 1, roof)
    # door: perimeter cell closest to the middle of the south side
    if len(xs_p):
        cx = xs_f.mean()
        cand = ~corner
        if cand.any():
            score = -zs_p * 4.0 + np.abs(xs_p - cx)
            score[~cand] = 1e9
            i = int(np.argmin(score))
            dx, dz = int(xs_p[i]), int(zs_p[i])
            out.append((np.array([dx, dx]), np.array([base + 1, base + 2]), np.array([dz, dz]), "minecraft:air"))
    # ladder between storeys
    if storeys > 1 and len(xs_i):
        for x, z in zip(xs_i, zs_i):
            for (nx, nz, facing) in ((x - 1, z, "east"), (x + 1, z, "west"), (x, z - 1, "south"), (x, z + 1, "north")):
                if 0 <= nz < fp.shape[0] and 0 <= nx < fp.shape[1] and perim[nz, nx]:
                    ys = np.arange(base + 1, top + 1)
                    out.append((np.full(len(ys), x), ys, np.full(len(ys), z), f"minecraft:ladder[facing={facing}]"))
                    break
            else:
                continue
            break
    # lanterns
    if lights and len(xs_i):
        cxm, czm = xs_i.mean(), zs_i.mean()
        i = int(np.argmin((xs_i - cxm) ** 2 + (zs_i - czm) ** 2))
        for s in range(storeys):
            out.append((np.array([xs_i[i]]), np.array([base + s * sh + 1]), np.array([zs_i[i]]),
                        "minecraft:lantern[hanging=false]"))
    return out


def platform_blocks(fp: np.ndarray, deck_y: int, deck: str, fence: str):
    out = []
    zs, xs = np.nonzero(fp)
    if not len(xs):
        return out
    out.append((xs, np.full(len(xs), deck_y), zs, deck))
    inner = cv2.erode(fp.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    pz, px = np.nonzero(fp & ~inner)
    out.append((px, np.full(len(px), deck_y + 1), pz, fence))
    return out


# ------------------------------------------------------------------ crown colour -> block
# In-game look of possible crown blocks (texture average incl. biome tint, measured from
# the 1.21.11 client textures).  Leaves first; the other blocks are used when no leaf
# type fits the map colour (black, grey, white, red, purple, blue ... crowns).
CROWN_MATERIALS = {
    "oak_leaves": (67, 96, 26), "birch_leaves": (65, 84, 43), "spruce_leaves": (47, 75, 47),
    "jungle_leaves": (73, 103, 26), "acacia_leaves": (69, 99, 27), "dark_oak_leaves": (52, 80, 22),
    "mangrove_leaves": (60, 86, 23), "cherry_leaves": (229, 172, 194), "azalea_leaves": (90, 114, 44),
    "flowering_azalea_leaves": (99, 111, 60), "pale_oak_leaves": (116, 121, 114),
    "moss_block": (89, 109, 45), "nether_wart_block": (114, 2, 2), "warped_wart_block": (22, 119, 121),
    "black_wool": (20, 21, 25), "gray_wool": (62, 68, 71), "light_gray_wool": (142, 142, 134),
    "white_wool": (233, 236, 236), "brown_wool": (114, 71, 40), "orange_wool": (240, 118, 19),
    "yellow_wool": (248, 197, 39), "red_wool": (160, 39, 34), "purple_wool": (121, 42, 172),
    "magenta_wool": (189, 68, 179), "pink_wool": (237, 141, 172), "light_blue_wool": (58, 175, 217),
    "cyan_wool": (21, 137, 145), "blue_wool": (53, 57, 157),
}
_MAT_NAMES = list(CROWN_MATERIALS)
_MAT_LEAF = np.array([n.endswith("_leaves") for n in _MAT_NAMES])
_mat_lab = None


def _lab(rgb):
    return cv2.cvtColor(np.array([[rgb]], np.uint8), cv2.COLOR_RGB2LAB)[0, 0].astype(float)


def _lch(rgb):
    L, a, b = _lab(rgb)
    L, a, b = L * 100 / 255, a - 128, b - 128
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


_MAT_LCH = {n: _lch(c) for n, c in CROWN_MATERIALS.items()}


@lru_cache(maxsize=4096)
def _crown_material_q(q):
    rgb = tuple(min(255, c * 8 + 4) for c in q)
    L, C, H = _lch(rgb)
    if 100 <= H <= 180 and C >= 9:
        # green crown -> real leaves; every shade gets its own leaf type
        if H >= 155:
            return "spruce_leaves"                       # blue-green / teal
        if L < 36:
            return "mangrove_leaves" if H < 115 else "dark_oak_leaves"
        if H < 108:
            return "azalea_leaves" if L > 58 else "acacia_leaves"
        if C > 38 and L > 50:
            return "azalea_leaves" if L > 62 else "jungle_leaves"
        if L > 62:
            return "birch_leaves"
        return "oak_leaves"
    if C < 12:
        # grey scale: black / grey / pale / white
        cands = ["black_wool", "gray_wool", "pale_oak_leaves", "light_gray_wool", "white_wool"]
        return min(cands, key=lambda n: abs(_MAT_LCH[n][0] - L))
    # coloured, not green: same hue family, then closest lightness (leaves win ties)
    def score(n):
        l2, c2, h2 = _MAT_LCH[n]
        dh = min(abs(h2 - H), 360 - abs(h2 - H))
        if c2 < 12 or dh > 40:
            return 1e9
        return dh * 0.6 + abs(l2 - L) * 0.8 - (6 if n.endswith("_leaves") else 0)
    best = min(_MAT_NAMES, key=score)
    if score(best) >= 1e9:
        return "oak_leaves"
    return best


def crown_material(rgb) -> str:
    """Block name (without namespace) for a crown drawn in colour rgb on the map."""
    q = tuple(int(c) // 8 for c in rgb)
    return _crown_material_q(q)


def crown_state(material: str) -> str:
    if material.endswith("_leaves"):
        return f"minecraft:{material}[distance=7,persistent=true,waterlogged=false]"
    return "minecraft:" + material


def log_for_material(material: str) -> str | None:
    """Trunk wood that suits the crown (None = keep the model's own wood)."""
    return {"cherry_leaves": "cherry", "pale_oak_leaves": "pale_oak", "black_wool": "dark_oak",
            "gray_wool": "dark_oak", "nether_wart_block": "crimson", "warped_wart_block": "warped",
            "mangrove_leaves": "mangrove", "spruce_leaves": "spruce", "birch_leaves": "birch",
            "dark_oak_leaves": "dark_oak", "jungle_leaves": "jungle", "acacia_leaves": "acacia"}.get(material)


SHAPE_OF_KIND = {"spruce": "conifer", "jungle": "tropical", "acacia": "acacia", "palm": "palm", "bush": "bush"}
