"""Turn a block volume into a textured triangle mesh (for the 3D block view).

* Textures come from the vanilla client jar (assets/minecraft/textures/block),
  which is looked up in the usual places; without a jar every block gets a flat
  colour tile, so the view still works.
* Only faces next to air / transparent blocks are emitted (hidden faces culled).
* Plants are crossed quads, fences / lanterns / ladders small boxes, water is a
  separate translucent mesh.
* Grass, leaves and water get Minecraft's default biome tints.

Vertex layout (float32): x, y, z, u, v, r, g, b
"""
from __future__ import annotations

import glob
import io
import os
import zipfile
import zlib

import numpy as np
from PIL import Image

TILE = 16
ATLAS_COLS = 32

AIR = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:structure_void", "minecraft:light"}
PLANTS = {"short_grass", "tall_grass", "fern", "large_fern", "dandelion", "poppy", "blue_orchid", "allium",
          "azure_bluet", "oxeye_daisy", "cornflower", "lily_of_the_valley", "dead_bush", "sweet_berry_bush",
          "bush", "firefly_bush", "torchflower", "wither_rose", "sunflower", "lilac", "rose_bush", "peony",
          "pink_petals", "wildflowers", "leaf_litter", "short_dry_grass", "tall_dry_grass", "sugar_cane",
          "brown_mushroom", "red_mushroom", "seagrass", "kelp", "kelp_plant", "open_eyeblossom", "closed_eyeblossom"}
TRANSPARENT_FULL = ("_leaves", "glass", "ice")        # full cubes that do not hide neighbours

GRASS_TINT = (0x91, 0xBD, 0x59)
FOLIAGE_TINT = (0x77, 0xAB, 0x2F)
LEAF_TINT = {"birch_leaves": (0x80, 0xA7, 0x55), "spruce_leaves": (0x61, 0x99, 0x61)}
UNTINTED_LEAVES = {"cherry_leaves", "azalea_leaves", "flowering_azalea_leaves", "pale_oak_leaves"}
WATER_TINT = (0x3F, 0x76, 0xE4)

# face directions: (dy, dz, dx), name, shade
DIRS = [((1, 0, 0), "top", 1.0), ((-1, 0, 0), "bottom", 0.5), ((0, -1, 0), "north", 0.8),
        ((0, 1, 0), "south", 0.8), ((0, 0, -1), "west", 0.62), ((0, 0, 1), "east", 0.62)]
# quad corners (x, y, z) for a unit cube face, counter-clockwise seen from outside
QUAD = {
    "top": [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)],
    "bottom": [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    "north": [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)],
    "south": [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    "west": [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    "east": [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
}
QUAD_UV = [(0, 1), (1, 1), (1, 0), (0, 0)]
TRI = [0, 1, 2, 0, 2, 3]


# ------------------------------------------------------------------ textures
def find_client_jar(extra: str | None = None) -> str | None:
    """Locate a 1.21.11 client jar (only read for textures)."""
    cands = [extra] if extra else []
    appdata = os.environ.get("APPDATA", "")
    home = os.path.expanduser("~")
    cands += [os.path.join(appdata, ".minecraft", "versions", "1.21.11", "1.21.11.jar"),
              os.path.join(home, ".gradle", "caches", "fabric-loom", "1.21.11", "minecraft-client.jar")]
    cands += sorted(glob.glob(os.path.join(appdata, ".minecraft", "versions", "1.21*", "*.jar")), reverse=True)
    for c in cands:
        if c and os.path.isfile(c):
            try:
                with zipfile.ZipFile(c) as z:
                    z.getinfo("assets/minecraft/textures/block/stone.png")
                return c
            except (KeyError, zipfile.BadZipFile, OSError):
                continue
    return None


def base_name(state: str) -> str:
    return state.split("[", 1)[0].split(":")[-1]


def props(state: str) -> dict:
    if "[" not in state:
        return {}
    return dict(p.split("=", 1) for p in state[state.index("[") + 1:-1].split(",") if "=" in p)


def texture_names(n: str) -> dict:
    """Candidate texture files for top / side / bottom of block n (first existing wins)."""
    t = {"top": [n + "_top", n], "side": [n + "_side", n], "bottom": [n + "_bottom", n]}
    special = {
        "grass_block": ("grass_block_top", "grass_block_side", "dirt"),
        "podzol": ("podzol_top", "podzol_side", "dirt"),
        "mycelium": ("mycelium_top", "mycelium_side", "dirt"),
        "dirt_path": ("dirt_path_top", "dirt_path_side", "dirt"),
        "snow_block": ("snow", "snow", "snow"),
        "water": ("water_still", "water_still", "water_still"),
        "lava": ("lava_still", "lava_still", "lava_still"),
        "sandstone": ("sandstone_top", "sandstone", "sandstone_bottom"),
        "red_sandstone": ("red_sandstone_top", "red_sandstone", "red_sandstone_bottom"),
        "smooth_sandstone": ("sandstone_top", "sandstone_top", "sandstone_top"),
        "basalt": ("basalt_top", "basalt_side", "basalt_top"),
        "polished_basalt": ("polished_basalt_top", "polished_basalt_side", "polished_basalt_top"),
        "blackstone": ("blackstone_top", "blackstone", "blackstone_top"),
        "deepslate": ("deepslate_top", "deepslate", "deepslate_top"),
        "smooth_stone_slab": ("smooth_stone", "smooth_stone_slab_side", "smooth_stone"),
        "hay_block": ("hay_block_top", "hay_block_side", "hay_block_top"),
        "bookshelf": ("oak_planks", "bookshelf", "oak_planks"),
        "crafting_table": ("crafting_table_top", "crafting_table_front", "oak_planks"),
    }
    if n in special:
        a, b, c = special[n]
        return {"top": [a], "side": [b], "bottom": [c]}
    if n.endswith(("_log", "_stem")) or n.endswith("_wood"):
        core = n[:-5] + "_log" if n.endswith("_wood") else n
        return {"top": [core + "_top"], "side": [core], "bottom": [core + "_top"]}
    for suf, base in (("_fence", "_planks"), ("_fence_gate", "_planks"), ("_slab", "_planks"),
                      ("_stairs", "_planks"), ("_door", "_planks"), ("_trapdoor", "_planks")):
        if n.endswith(suf):
            stem = n[: -len(suf)]
            c = [stem + base, stem, stem + "s", stem + "_block"]
            return {"top": c, "side": c, "bottom": c}
    if n in ("lantern", "soul_lantern"):
        return {"top": [n], "side": [n], "bottom": [n]}
    return t


def tint_for(n: str, face: str):
    if n == "grass_block" and face == "top":
        return GRASS_TINT
    if n in ("short_grass", "tall_grass", "fern", "large_fern", "sugar_cane", "bush"):
        return GRASS_TINT
    if n.endswith("_leaves") and n not in UNTINTED_LEAVES:
        return LEAF_TINT.get(n, FOLIAGE_TINT)
    if n in ("vine",):
        return FOLIAGE_TINT
    if n == "water":
        return WATER_TINT
    return (255, 255, 255)


def _fallback_color(n: str):
    from .preview import BLOCK_COLORS
    key = "minecraft:" + n
    if key in BLOCK_COLORS:
        return tuple(int(c) for c in BLOCK_COLORS[key])
    h = zlib.crc32(n.encode())
    return (90 + h % 120, 90 + (h >> 8) % 120, 90 + (h >> 16) % 120)


class Atlas:
    """Texture atlas built on demand from the client jar."""

    def __init__(self, jar: str | None):
        self.jar = jar
        self._zip = zipfile.ZipFile(jar) if jar else None
        self.tiles: list[np.ndarray] = []
        self.index: dict[str, int] = {}
        self.slot: dict[tuple[str, str], int] = {}

    def _load_png(self, name: str) -> np.ndarray | None:
        if self._zip is None:
            return None
        try:
            data = self._zip.read(f"assets/minecraft/textures/block/{name}.png")
        except KeyError:
            return None
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        w = im.width
        im = im.crop((0, 0, w, w))          # animated textures: first frame
        if w != TILE:
            im = im.resize((TILE, TILE), Image.NEAREST)
        return np.asarray(im, np.uint8)

    def _add(self, key: str, img: np.ndarray) -> int:
        if key not in self.index:
            self.index[key] = len(self.tiles)
            self.tiles.append(img)
        return self.index[key]

    def tile(self, n: str, face: str) -> int:
        f = "top" if face == "top" else ("bottom" if face == "bottom" else "side")
        k = (n, f)
        if k in self.slot:
            return self.slot[k]
        idx = None
        for cand in texture_names(n)[f]:
            if cand in self.index:
                idx = self.index[cand]
                break
            img = self._load_png(cand)
            if img is not None:
                idx = self._add(cand, img)
                break
        if idx is None:
            col = _fallback_color(n)
            img = np.empty((TILE, TILE, 4), np.uint8)
            img[..., :3] = col
            img[..., 3] = 255
            # light noise so flat tiles still read as blocks
            rng = np.random.default_rng(zlib.crc32(n.encode()))
            img[..., :3] = np.clip(img[..., :3].astype(int) + rng.integers(-12, 13, (TILE, TILE, 1)), 0, 255)
            idx = self._add("#" + n, img)
        self.slot[k] = idx
        return idx

    def image(self) -> np.ndarray:
        rows = max(1, (len(self.tiles) + ATLAS_COLS - 1) // ATLAS_COLS)
        out = np.zeros((rows * TILE, ATLAS_COLS * TILE, 4), np.uint8)
        for i, t in enumerate(self.tiles):
            r, c = divmod(i, ATLAS_COLS)
            out[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
        return out

    def uv(self, idx: np.ndarray, rows: int):
        """Top-left uv of tile idx in an atlas with `rows` rows."""
        r, c = np.divmod(idx, ATLAS_COLS)
        return c / ATLAS_COLS, r / rows, 1.0 / ATLAS_COLS, 1.0 / rows


# ------------------------------------------------------------------ mesh
class BlockInfo:
    """Per-palette-entry flags for vectorised meshing."""

    def __init__(self, names: list[str]):
        n = len(names)
        self.names = names
        self.base = [base_name(s) for s in names]
        self.kind = np.zeros(n, np.uint8)          # 0 air, 1 cube, 2 cross, 3 water, 4 small box
        self.occludes = np.zeros(n, bool)          # hides the neighbour's face
        self.cull_same = np.zeros(n, bool)         # transparent cube: hide faces against itself
        for i, (s, b) in enumerate(zip(names, self.base)):
            if s.split("[")[0] in AIR:
                continue
            if b in ("water", "bubble_column"):
                self.kind[i] = 3
            elif b in PLANTS or b.endswith(("_sapling", "_tulip", "_flower", "_petals", "_roots")):
                self.kind[i] = 2
            elif b.endswith(("_fence", "_fence_gate", "_wall")) or b in ("lantern", "soul_lantern", "ladder",
                                                                        "torch", "wall_torch", "chain", "iron_bars"):
                self.kind[i] = 4
            else:
                self.kind[i] = 1
                transp = b.endswith(TRANSPARENT_FULL) or "glass" in b
                self.occludes[i] = not transp
                self.cull_same[i] = transp
        self.water_like = self.kind == 3


def _small_box(state: str):
    """(x0,y0,z0,x1,y1,z1) for small-shaped blocks."""
    b = base_name(state)
    if b.endswith(("_fence", "_wall", "iron_bars", "chain")):
        return (0.375, 0, 0.375, 0.625, 1, 0.625)
    if b.endswith("_fence_gate"):
        return (0, 0.3, 0.4375, 1, 1, 0.5625)
    if "lantern" in b:
        return (0.3125, 0, 0.3125, 0.6875, 0.5625, 0.6875)
    if "torch" in b:
        return (0.4375, 0, 0.4375, 0.5625, 0.625, 0.5625)
    if b == "ladder":
        f = props(state).get("facing", "north")
        t = 1 / 16
        return {"north": (0, 0, 1 - t, 1, 1, 1), "south": (0, 0, 0, 1, 1, t),
                "west": (1 - t, 0, 0, 1, 1, 1), "east": (0, 0, 0, t, 1, 1)}.get(f, (0, 0, 0, 1, 1, 1))
    return (0.25, 0, 0.25, 0.75, 0.75, 0.75)


def build_mesh(vol: np.ndarray, names: list[str], atlas: Atlas, y_floor: int = 0):
    """vol [y,z,x] palette indices.  Returns (opaque_vertices, water_vertices, atlas_rgba).

    y_floor: layers below this index are not drawn (keeps the view on the surface)."""
    info = BlockInfo(names)
    v = vol[y_floor:]
    Y, Z, X = v.shape
    kind = info.kind[v]
    pad = np.pad(v, 1, constant_values=0)
    # tiles per (palette id, face) and per-face tint
    face_tile = {}
    face_tint = {}
    for _, fname, _ in DIRS:
        face_tile[fname] = np.array([atlas.tile(b, fname) if info.kind[i] in (1, 3, 4) else 0
                                     for i, b in enumerate(info.base)], np.int32)
        face_tint[fname] = np.array([tint_for(b, fname) for b in info.base], np.float32) / 255.0
    cross_tile = np.array([atlas.tile(b, "side") if info.kind[i] == 2 else 0 for i, b in enumerate(info.base)],
                          np.int32)

    opaque_parts, water_parts = [], []

    def emit(pos, quad, tiles, shade, tint, uvq=QUAD_UV, target=opaque_parts):
        # pos (n,3) xyz block origins; quad 4 corners (n,4,3) or (4,3)
        n = len(pos)
        if n == 0:
            return
        q = np.asarray(quad, np.float32)
        corners = pos[:, None, :].astype(np.float32) + (q if q.ndim == 3 else q[None])
        target.append((corners, tiles, np.asarray(uvq, np.float32), shade, tint))

    # ---- full cubes and water faces
    for (dy, dz, dx), fname, shade in DIRS:
        nb = pad[1 + dy:1 + dy + Y, 1 + dz:1 + dz + Z, 1 + dx:1 + dx + X]
        cube = kind == 1
        vis = cube & ~info.occludes[nb] & ~(info.cull_same[v] & (nb == v))
        yy, zz, xx = np.nonzero(vis)
        ids = v[yy, zz, xx]
        emit(np.stack([xx, yy, zz], 1), QUAD[fname], face_tile[fname][ids], shade, face_tint[fname][ids])
        # water: faces against non-water, non-occluding blocks
        wat = kind == 3
        wvis = wat & ~info.water_like[nb] & ~info.occludes[nb]
        yy, zz, xx = np.nonzero(wvis)
        ids = v[yy, zz, xx]
        q = np.array(QUAD[fname], np.float32)
        q[:, 1] = np.minimum(q[:, 1], 0.875)            # water surface sits 2 px low
        emit(np.stack([xx, yy, zz], 1), q, face_tile[fname][ids], shade, face_tint[fname][ids], target=water_parts)

    # ---- plants: two crossed quads
    yy, zz, xx = np.nonzero(kind == 2)
    if len(yy):
        ids = v[yy, zz, xx]
        pos = np.stack([xx, yy, zz], 1)
        low = np.array([("petals" in info.base[i] or "litter" in info.base[i]) for i in range(len(names))])[ids]
        for quad in ([(0.15, 0, 0.15), (0.85, 0, 0.85), (0.85, 1, 0.85), (0.15, 1, 0.15)],
                     [(0.15, 0, 0.85), (0.85, 0, 0.15), (0.85, 1, 0.15), (0.15, 1, 0.85)]):
            emit(pos[~low], quad, cross_tile[ids[~low]], 0.9, face_tint["north"][ids[~low]])
        emit(pos[low], [(0, 0.05, 1), (1, 0.05, 1), (1, 0.05, 0), (0, 0.05, 0)], cross_tile[ids[low]], 1.0,
             face_tint["top"][ids[low]])

    # ---- small boxes (fences, lanterns, ladders, torches)
    yy, zz, xx = np.nonzero(kind == 4)
    for y, z, x in zip(yy, zz, xx):
        i = int(v[y, z, x])
        x0, y0, z0, x1, y1, z1 = _small_box(names[i])
        for _, fname, shade in DIRS:
            q = np.array(QUAD[fname], np.float32)
            q[:, 0] = np.where(q[:, 0] > 0, x1, x0)
            q[:, 1] = np.where(q[:, 1] > 0, y1, y0)
            q[:, 2] = np.where(q[:, 2] > 0, z1, z0)
            emit(np.array([[x, y, z]]), q, face_tile[fname][[i]], shade, face_tint[fname][[i]])

    img = atlas.image()
    rows = img.shape[0] // TILE

    def assemble(parts):
        if not parts:
            return np.zeros((0, 8), np.float32)
        out = []
        for corners, tiles, uvq, shade, tint in parts:
            n = len(corners)
            u0, v0, du, dv = atlas.uv(tiles, rows)
            # inset uvs by a hair to avoid bleeding from neighbouring tiles
            eps_u, eps_v = du * 0.002, dv * 0.002
            uu = u0[:, None] + eps_u + uvq[None, :, 0] * (du - 2 * eps_u)
            vv = v0[:, None] + eps_v + uvq[None, :, 1] * (dv - 2 * eps_v)
            col = np.broadcast_to(np.asarray(tint, np.float32).reshape(-1, 3) * shade, (n, 3))
            vert = np.empty((n, 4, 8), np.float32)
            vert[..., 0:3] = corners
            vert[..., 3] = uu
            vert[..., 4] = vv
            vert[..., 5:8] = col[:, None, :]
            out.append(vert[:, TRI].reshape(-1, 8))
        m = np.concatenate(out)
        m[:, 1] += y_floor
        return m

    return assemble(opaque_parts), assemble(water_parts), img


def surface_floor(vol: np.ndarray, names: list[str], below: int = 6) -> int:
    """Lowest layer worth drawing: a few blocks under the lowest surface column."""
    info = BlockInfo(names)
    solid = info.kind[vol] != 0
    Y = vol.shape[0]
    any_col = solid.any(0)
    if not any_col.any():
        return 0
    top = Y - 1 - np.argmax(solid[::-1], axis=0)
    return int(max(0, top[any_col].min() - below))


# ------------------------------------------------------------------ level of detail (Voxy-like)
def tile_colors(atlas: Atlas, names: list[str]) -> np.ndarray:
    """Average on-screen colour (0..1) of the top face of every block, incl. biome tint."""
    out = np.zeros((len(names), 3), np.float32)
    for i, n in enumerate(names):
        b = base_name(n)
        t = atlas.tiles[atlas.tile(b, "top")]
        a = t[..., 3:4].astype(np.float32) / 255
        rgb = (t[..., :3] * a).sum((0, 1)) / max(1e-3, a.sum())
        out[i] = rgb / 255 * np.array(tint_for(b, "top"), np.float32) / 255
    return out


def white_tile(atlas: Atlas) -> int:
    if "#white" not in atlas.index:
        atlas._add("#white", np.full((TILE, TILE, 4), 255, np.uint8))
    return atlas.index["#white"]


def lod_mesh(top: np.ndarray, blk: np.ndarray, colors: np.ndarray, cell: int, origin=(0, 0),
             keep: np.ndarray | None = None, white_uv=(0.0, 0.0)):
    """Height-field of cell x cell columns (top = highest block of the cell).

    top [Z,X] world y (NONE = empty), blk ids into colors.  origin = (x, z) of top[0,0]
    in the output frame.  keep [cz,cx] bool: cells to draw (finer levels cover the rest).
    Returns vertices (n, 8) like build_mesh (uv = white tile, colour = block colour)."""
    Z, X = top.shape
    cz, cx = Z // cell, X // cell
    t = top[:cz * cell, :cx * cell].reshape(cz, cell, cx, cell).astype(np.int32)
    t = np.where(t == -32768, -100000, t)
    tmax = t.max((1, 3))
    # block of the highest column in the cell
    flat = t.transpose(0, 2, 1, 3).reshape(cz, cx, cell * cell)
    arg = flat.argmax(2)
    b = blk[:cz * cell, :cx * cell].reshape(cz, cell, cx, cell).transpose(0, 2, 1, 3).reshape(cz, cx, cell * cell)
    bid = np.take_along_axis(b, arg[..., None], 2)[..., 0]
    valid = tmax > -100000
    if keep is not None:
        valid &= keep
    h = np.where(tmax > -100000, tmax + 1, -64).astype(np.float32)     # surface = top of the block
    col = colors[bid]
    parts = []

    def quads(corners, shade, cc):
        n = len(corners)
        if n == 0:
            return
        v = np.empty((n, 4, 8), np.float32)
        v[..., 0:3] = corners
        v[..., 3] = white_uv[0]
        v[..., 4] = white_uv[1]
        v[..., 5:8] = (cc * shade)[:, None, :]
        parts.append(v[:, TRI].reshape(-1, 8))

    zz, xx = np.nonzero(valid)
    x0 = origin[0] + xx * cell
    z0 = origin[1] + zz * cell
    y = h[zz, xx]
    c = col[zz, xx]
    q = np.array(QUAD["top"], np.float32)
    corners = np.stack([x0[:, None] + q[None, :, 0] * cell, np.repeat(y[:, None], 4, 1),
                        z0[:, None] + q[None, :, 2] * cell], -1)
    quads(corners, 1.0, c)
    # side walls where the neighbour cell is lower
    hp = np.pad(h, 1, constant_values=-64)
    for fname, dz_, dx_, shade in (("north", -1, 0, 0.8), ("south", 1, 0, 0.8), ("west", 0, -1, 0.62),
                                   ("east", 0, 1, 0.62)):
        nb = hp[1 + dz_:1 + dz_ + cz, 1 + dx_:1 + dx_ + cx][zz, xx]
        m = nb < y
        if not m.any():
            continue
        q = np.array(QUAD[fname], np.float32)
        lo, hi = nb[m], y[m]
        ys = np.where(q[None, :, 1] > 0, hi[:, None], lo[:, None])
        corners = np.stack([x0[m][:, None] + q[None, :, 0] * cell, ys, z0[m][:, None] + q[None, :, 2] * cell], -1)
        quads(corners, shade * 0.92, c[m])
    return np.concatenate(parts) if parts else np.zeros((0, 8), np.float32)
