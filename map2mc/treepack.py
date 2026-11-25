"""Tree packs: import WorldEdit/Sponge .schem trees and place them in the world.

Import (once, GUI button "Baum-Paket importieren"):
  * source = .7z / .zip archive or a folder with .schem files (Sponge format v2/v3)
  * every schematic is analysed: dominant leaf type (oak, birch, spruce, ...),
    palm or not, trunk base, crown radius, height
  * leaves are rewritten to persistent=true (packs often contain decaying
    leaves - without logs nearby they would vanish in game), old names fixed
    (grass -> short_grass), only natural-looking trees are kept (no glass,
    fantasy, crystals ...) and only sizes that fit the region margin
  * the result is one compact library file (.m2trees, numpy .npz)

Placement (generator): for a tree of leaf type k and crown radius r a pack tree
of the same type with a similar radius is chosen deterministically from the
position (seamless across region workers) and rotated in 90° steps.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import tempfile
import zipfile
import zlib
from functools import lru_cache

import numpy as np

from . import nbt

LEAF_KINDS = ["oak", "birch", "spruce", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "azalea",
              "pale_oak"]
MAX_RADIUS = 15          # must stay below the generator's region margin (16)
MAX_HEIGHT = 48
# folders / blocks that are not natural trees
BAD_WORDS = ("glass", "fantasy", "alien", "crystal", "iron", "cobble", "cave", "bonsai", "cartoon", "mushroom",
             "boulder", "rock", "fallen", "stump", "seaweed", "water", "city", "dead", "grasses", "flower")
BAD_BLOCKS = ("concrete", "wool", "terracotta", "iron_", "gold_", "diamond", "amethyst", "glowstone", "sea_lantern",
              "shroomlight", "end_rod", "prismarine", "purpur", "quartz", "obsidian", "crying", "_ore", "lantern",
              "torch", "froglight", "copper")
RENAMES = {"minecraft:grass": "minecraft:short_grass"}

# Curated selection for the "5K Trees" pack (reviewed visually): folder -> shape group,
# plus name fragments of individual models that do not fit a natural map.
# Other packs fall back to a shape group from leaf type / folder words.
SHAPE_GROUPS = ["broadleaf", "conifer", "palm", "tropical", "acacia", "bush", "swamp"]
CURATED = {
    "/Oaks": "broadleaf", "/OaksOther": "broadleaf", "/SimpleOak": "broadleaf", "/Birch": "broadleaf",
    "/GardenTree": "broadleaf",
    "/Spruce": "conifer", "/SpruceSlim": "conifer", "/FirCone": "conifer", "/SimpleSpruce": "conifer",
    "/FirsRealistic": "conifer", "/SpruceCone": "conifer",
    "/Palms": "palm", "/PalmsOther": "palm", "/VarietyPalms": "palm",
    "/AncientTree": "tropical", "/Jungle": "tropical", "/BirchTall": "tropical", "/JungleTallSlim": "tropical",
    "/Acacia": "acacia",
    "/Bush": "bush", "/MiniShrub": "bush", "/TreeShrub": "bush", "/Shrub": "bush", "/GardenBush": "bush",
    "/SwampTree": "swamp",
}
CURATED_SKIP_DIRS = ("/BlockyDefault", "/SimplePalm", "/ForestDecor", "/DecorPlants")
CURATED_BAD_NAMES = ("windswept", "roughoak", "stick", "spike", "acaciapine", "slimsprucedark", "sideways",
                     "bushpalm", "roughoakpalm", "baresmall", "acacialittle", "junglesapling", "littletropical",
                     "roundjunglebus", "nether", "melon", "bushwithgrass", "alpineoak")


def shape_of(path: str, kind: str) -> str | None:
    """Shape group of a pack model, or None to skip it."""
    p = "/" + path.replace("\\", "/").strip("/")
    d = p.rsplit("/", 1)[0]
    name = p.rsplit("/", 1)[-1].lower()
    if any(d.endswith(x) for x in CURATED_SKIP_DIRS):
        return None
    for suffix, grp in CURATED.items():
        if d.endswith(suffix):
            if any(w in name for w in CURATED_BAD_NAMES):
                return None
            if grp == "broadleaf" and "willow" in name:
                return "swamp"
            return grp
    low = p.lower()
    if "palm" in low:
        return "palm"
    if "bush" in low or "shrub" in low:
        return "bush"
    return {"spruce": "conifer", "jungle": "tropical", "acacia": "acacia"}.get(kind, "broadleaf")


# ------------------------------------------------------------------ reading .schem
def _varints(data: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(data, np.uint8)
    if not (a & 0x80).any():
        return a[:n].astype(np.int64)
    out = np.empty(n, np.int64)
    i = k = 0
    while k < n:
        v = shift = 0
        while True:
            b = int(a[i])
            i += 1
            v |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        out[k] = v
        k += 1
    return out


def read_schem(raw: bytes):
    """Sponge schematic (v2 or v3) -> (vol [y,z,x] int, palette list of state strings)."""
    root = nbt.decode(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    s = root.get("Schematic", root)
    w, h, l = (int(nbt.plain(s[k])) for k in ("Width", "Height", "Length"))
    if "Blocks" in s:                        # v3
        pal, data = s["Blocks"]["Palette"], s["Blocks"]["Data"]
    else:                                    # v2
        pal, data = s["Palette"], s["BlockData"]
    names = [None] * (max(int(nbt.plain(v)) for v in pal.values()) + 1)
    for k, v in pal.items():
        names[int(nbt.plain(v))] = k
    names = [n or "minecraft:air" for n in names]
    idx = _varints(np.asarray(data).astype(np.int8).tobytes(), w * h * l)
    return idx.reshape(h, l, w), names


def _split(state: str):
    if "[" in state:
        name, rest = state.split("[", 1)
        props = dict(p.split("=", 1) for p in rest.rstrip("]").split(",") if "=" in p)
    else:
        name, props = state, {}
    return name, props


def _join(name, props):
    return name + ("[" + ",".join(f"{k}={props[k]}" for k in sorted(props)) + "]" if props else "")


def fix_state(state: str) -> str:
    """Persistent leaves, renamed blocks."""
    name, props = _split(state)
    name = RENAMES.get(name, name)
    if name.endswith("_leaves"):
        props["persistent"] = "true"
        props.setdefault("distance", "7")
        props.setdefault("waterlogged", "false")
    return _join(name, props)


def leaf_kind(name: str) -> str | None:
    b = name.split(":")[-1]
    if not b.endswith("_leaves"):
        return None
    b = b[:-7]
    if b == "flowering_azalea":
        b = "azalea"
    return b if b in LEAF_KINDS else None


# ------------------------------------------------------------------ analysing one tree
def analyse(vol: np.ndarray, names: list[str], path: str):
    """Return a library entry dict or (None, reason)."""
    low = path.lower().replace("\\", "/")
    if any(w in low for w in BAD_WORDS):
        return None, "category"
    fixed = [fix_state(n) for n in names]
    base = [n.split("[")[0] for n in fixed]
    air = np.array([b in ("minecraft:air", "minecraft:cave_air", "minecraft:structure_void") for b in base])
    if any(any(w in b for w in BAD_BLOCKS) or (b.endswith(":glass") or b.endswith("_stained_glass"))
           for b, a in zip(base, air) if not a and (vol == base.index(b)).any()):
        return None, "blocks"
    solid = ~air[vol]
    if solid.sum() < 6:
        return None, "empty"
    kinds = [leaf_kind(b) for b in base]
    is_leaf = np.array([k is not None for k in kinds])
    is_log = np.array([b.endswith(("_log", "_wood", "_stem", "_hyphae")) for b in base])
    leaf_vox = is_leaf[vol]
    if leaf_vox.sum() < 8:
        return None, "no leaves"
    cnt = {}
    for i, k in enumerate(kinds):
        if k:
            cnt[k] = cnt.get(k, 0) + int((vol == i).sum())
    kind = max(cnt, key=cnt.get)
    # trunk base: lowest layer with logs (else lowest solid layer)
    log_vox = is_log[vol]
    ref = log_vox if log_vox.any() else solid
    ys = np.nonzero(ref.any((1, 2)))[0]
    by = int(ys[0])
    zz, xx = np.nonzero(ref[by])
    bx, bz = int(round(xx.mean())), int(round(zz.mean()))
    ly, lz, lx = np.nonzero(leaf_vox)
    dist = np.hypot(lx - bx, lz - bz)
    radius = float(np.percentile(dist, 90)) if len(dist) else 1.0
    sy, sz_, sx = np.nonzero(solid)
    reach = float(np.hypot(sx - bx, sz_ - bz).max())
    height = int(sy.max() - by + 1)
    if reach > MAX_RADIUS or height > MAX_HEIGHT:
        return None, "too big"
    shape = shape_of(path, kind)
    if shape is None:
        return None, "curated out"
    palm = shape == "palm"
    # compact list of blocks relative to the trunk base
    pal_fixed = sorted({fixed[i] for i in np.unique(vol[solid])})
    pidx = {s: i for i, s in enumerate(pal_fixed)}
    remap = np.array([pidx.get(f, -1) for f in fixed])
    ids = remap[vol[sy, sz_, sx]]
    blocks = np.stack([sx - bx, sy - by, sz_ - bz, ids], 1).astype(np.int16)
    leafmask = np.array([leaf_kind(s.split("[")[0]) is not None for s in pal_fixed])
    return {
        "name": os.path.splitext(os.path.basename(path))[0], "category": os.path.dirname(path).replace("\\", "/"),
        "kind": kind, "palm": palm, "shape": shape, "radius": round(max(1.0, radius), 2), "height": height,
        "palette": pal_fixed, "leaf": leafmask.tolist(), "blocks": blocks,
    }, "ok"


# ------------------------------------------------------------------ import a pack
def _iter_sources(src: str, tmp: str):
    """Yield (relative path, bytes) of every .schem file in a folder / .zip / .7z."""
    if os.path.isdir(src):
        for dp, _, fs in os.walk(src):
            for f in fs:
                if f.lower().endswith(".schem"):
                    p = os.path.join(dp, f)
                    with open(p, "rb") as fh:
                        yield os.path.relpath(p, src), fh.read()
    elif src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.lower().endswith(".schem"):
                    yield n, z.read(n)
    elif src.lower().endswith(".7z"):
        import py7zr
        with py7zr.SevenZipFile(src) as z:
            names = [n for n in z.getnames() if n.lower().endswith(".schem")]
            z.extract(path=tmp, targets=names)       # extracted into an empty temp folder, data only
        yield from _iter_sources(tmp, tmp)
    else:
        raise ValueError("Baum-Paket: bitte einen Ordner, eine .zip- oder .7z-Datei wählen")


def count_sources(src: str) -> int | None:
    try:
        if src.lower().endswith(".7z"):
            import py7zr
            with py7zr.SevenZipFile(src) as z:
                return sum(n.lower().endswith(".schem") for n in z.getnames())
        if src.lower().endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                return sum(n.lower().endswith(".schem") for n in z.namelist())
    except Exception:  # noqa: BLE001
        return None
    return None


def import_pack(src: str, out_path: str, progress=None, log=None, cancel=None) -> dict:
    """Build a .m2trees library from a tree pack.  Returns a summary."""
    log = log or (lambda s: None)
    out_path = os.path.abspath(out_path)
    total = count_sources(src)
    tmp = tempfile.mkdtemp(prefix="map2mc_trees_")
    entries, reasons = [], {}
    try:
        if src.lower().endswith(".7z"):
            log("Entpacke Archiv …")
        for k, (rel, raw) in enumerate(_iter_sources(src, tmp)):
            if cancel is not None and cancel.is_set():
                break
            try:
                vol, names = read_schem(raw)
                e, why = analyse(vol, names, rel)
            except Exception as ex:  # noqa: BLE001  (broken file in the pack)
                e, why = None, "unreadable"
                log(f"  übersprungen (nicht lesbar): {rel}: {ex}")
            reasons[why] = reasons.get(why, 0) + 1
            if e:
                entries.append(e)
            if progress and total and k % 25 == 0:
                progress(min(0.99, (k + 1) / total))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not entries:
        raise ValueError("Im Paket wurden keine verwendbaren Bäume gefunden.")
    counts = [len(e["blocks"]) for e in entries]
    blocks = np.concatenate([e.pop("blocks") for e in entries]).astype(np.int16)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_out = out_path + ".tmp.npz"
    np.savez_compressed(tmp_out, blocks=blocks, offsets=offsets,
                        meta=np.array(json.dumps({"source": os.path.basename(src), "trees": entries})))
    os.replace(tmp_out, out_path)
    by_kind = {}
    for e in entries:
        k = e["shape"]
        by_kind[k] = by_kind.get(k, 0) + 1
    summary = {"path": out_path, "trees": len(entries), "by_kind": dict(sorted(by_kind.items())),
               "skipped": {k: v for k, v in reasons.items() if k != "ok"}}
    if progress:
        progress(1.0)
    return summary


DEFAULT_PACK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "treepacks",
                            "default.m2trees")


def resolve(path: str) -> str | None:
    """Settings value -> library file ('' = bundled default pack)."""
    if path and os.path.isfile(path):
        return path
    return DEFAULT_PACK if os.path.isfile(DEFAULT_PACK) else None


# ------------------------------------------------------------------ library for placement
class Library:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.blocks = z["blocks"]
        self.offsets = z["offsets"]
        meta = json.loads(str(z["meta"]))
        self.trees = meta["trees"]
        self.by_shape: dict[str, list[int]] = {}
        for i, t in enumerate(self.trees):
            shp = t.get("shape") or ("palm" if t.get("palm") else
                                     {"spruce": "conifer", "jungle": "tropical", "acacia": "acacia"}.get(t["kind"], "broadleaf"))
            t["shape"] = shp
            self.by_shape.setdefault(shp, []).append(i)
        self.radius = np.array([t["radius"] for t in self.trees], np.float32)

    def choose(self, shape: str, radius: float, h: int) -> int | None:
        """Deterministic choice (h = position hash) of a tree of a shape group with a similar crown."""
        cand = list(self.by_shape.get(shape, []))
        if shape == "broadleaf" and radius >= 7:          # big crowns: also the large tropical trees
            cand += self.by_shape.get("tropical", [])
        if not cand:
            cand = self.by_shape.get("broadleaf", [])
        if not cand:
            return None
        r = self.radius[cand]
        good = [c for c, rr in zip(cand, r) if 0.6 * radius <= rr <= 1.6 * radius + 0.5]
        if not good:   # nearest sizes
            order = np.argsort(np.abs(r - radius))[:3]
            good = [cand[i] for i in order]
        return good[h % len(good)]

    def blocks_of(self, i: int, rot: int):
        """(dx, dy, dz, states, is_leaf) of tree i rotated by rot*90° clockwise (seen from above)."""
        t = self.trees[i]
        b = self.blocks[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)
        dx, dy, dz, pid = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        for _ in range(rot % 4):
            dx, dz = -dz, dx
        states = [rotate_state(s, rot % 4) for s in t["palette"]]
        leaf = np.array(t["leaf"], bool)[pid]
        return dx, dy, dz, pid, states, leaf


@lru_cache(maxsize=4)
def load_library(path: str) -> Library | None:
    if not path or not os.path.isfile(path):
        return None
    return Library(path)


_FACING = ["north", "east", "south", "west"]


@lru_cache(maxsize=None)
def rotate_state(state: str, rot: int) -> str:
    """Rotate a block state clockwise by rot*90° (x' = -z, z' = x)."""
    if rot % 4 == 0 or "[" not in state:
        return state
    name, props = _split(state)
    out = dict(props)
    for _ in range(rot % 4):
        p = dict(out)
        if p.get("facing") in _FACING:
            p["facing"] = _FACING[(_FACING.index(p["facing"]) + 1) % 4]
        if p.get("axis") in ("x", "z"):
            p["axis"] = "z" if p["axis"] == "x" else "x"
        if all(d in out for d in _FACING):
            for j, d in enumerate(_FACING):
                p[_FACING[(j + 1) % 4]] = out[d]
        if "rotation" in p and p["rotation"].isdigit():
            p["rotation"] = str((int(p["rotation"]) + 4) % 16)
        out = p
    return _join(name, out)


@lru_cache(maxsize=None)
def swap_wood(state: str, kind: str) -> str:
    """Re-dress an oak tree as another wood type (packs without cherry/azalea/... trees)."""
    name, props = _split(state)
    if not name.startswith("minecraft:oak_"):
        return state
    part = name[len("minecraft:oak_"):]
    if part == "leaves":
        leaves = {"azalea": "minecraft:azalea_leaves"}.get(kind, f"minecraft:{kind}_leaves")
        return _join(leaves, props)
    if kind in ("azalea",):
        return state
    if part in ("log", "wood"):
        return _join(f"minecraft:{kind}_{part}", props)
    return state


@lru_cache(maxsize=None)
def redress(state: str, crown: str | None, wood: str | None) -> str:
    """Give a pack tree its own colour: leaves -> crown state, trunk -> wood type."""
    name, props = _split(state)
    if crown and name.endswith("_leaves"):
        return crown
    if wood and name.startswith("minecraft:") and name.endswith(("_log", "_wood")):
        part = "log" if name.endswith("_log") else "wood"
        stripped = "stripped_" if name.startswith("minecraft:stripped_") else ""
        if wood in ("crimson", "warped"):
            part = "stem" if part == "log" else "hyphae"
        return _join(f"minecraft:{stripped}{wood}_{part}", {k: v for k, v in props.items() if k == "axis"})
    return state
