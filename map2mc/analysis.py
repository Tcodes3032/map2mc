"""Map image analysis: pixel classification, cleanup and feature detection.

Pipeline (all at source-image resolution):
    classify_colors()   -> class id per pixel + confidence ("review" mask)
    cleanup_classes()   -> remove specks, fill unknown pixels, keep thin roads
    derive_structure()  -> roads vs. paths, bridges, inland water bodies
    detect_trees()      -> individual tree positions + crown radius from dark/pink blobs
    detect_buildings()  -> building footprints (polygons) inside built-up areas

Nothing here invents content: every structure comes from something visible
in the image.  Pixels that match no known colour are flagged in the review
mask so the user can fix them with the class brush.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CID, CLASS_KEYS, N_CLASSES, WATER_CLASSES

WATER_SET = np.zeros(N_CLASSES, bool)
WATER_SET[list(WATER_CLASSES)] = True


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


# ------------------------------------------------------------------ classification
def classify_colors(img: np.ndarray, prototypes: dict, max_dist: float, progress=None):
    """Nearest-prototype classification in CIE-Lab.

    Returns (classes uint8, review bool): `review` marks pixels whose colour is
    far from every prototype or ambiguous between two classes.
    """
    h, w = img.shape[:2]
    smooth = cv2.medianBlur(img, 3)                       # remove compression noise
    protos = []
    for key, cols in prototypes.items():
        if key not in CID:
            continue
        for c in cols:
            protos.append((CID[key], np.array(c, np.float32)))
    if not protos:
        raise ValueError("Keine Farbprototypen definiert.")
    proto_lab = _to_lab(np.array([[p[1] for p in protos]], np.uint8))[0]
    proto_cls = np.array([p[0] for p in protos])

    classes = np.empty((h, w), np.uint8)
    review = np.zeros((h, w), bool)
    strip = max(1, 4_000_000 // max(w, 1))              # bounded memory for huge maps
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        lab = _to_lab(smooth[y0:y1])
        # best distance per class
        best = np.full((N_CLASSES,) + lab.shape[:2], np.inf, np.float32)
        for (cls, pl) in zip(proto_cls, proto_lab):
            d = lab - pl
            d = np.sqrt((d * d).sum(-1))
            np.minimum(best[cls], d, out=best[cls])
        order = np.argsort(best, axis=0)
        c1 = order[0]
        d1 = np.take_along_axis(best, order[:1], 0)[0]
        d2 = np.take_along_axis(best, order[1:2], 0)[0]
        cls = c1.astype(np.uint8)
        far = d1 > max_dist
        cls[far] = CID["unknown"]
        classes[y0:y1] = cls
        review[y0:y1] = far | ((d2 - d1) < 2.0)
        if progress:
            progress(y1 / h)
    return classes, review


def _majority(classes: np.ndarray, ksize: int, exclude=()):
    """Per-pixel majority class in a ksize×ksize window (ignoring `exclude`)."""
    best_cnt = np.full(classes.shape, -1.0, np.float32)
    best_cls = classes.copy()
    for c in range(N_CLASSES):
        if c in exclude:
            continue
        m = (classes == c).astype(np.float32)
        if not m.any():
            continue
        cnt = cv2.boxFilter(m, -1, (ksize, ksize), normalize=False, borderType=cv2.BORDER_REPLICATE)
        better = cnt > best_cnt
        best_cnt[better] = cnt[better]
        best_cls[better] = c
    return best_cls, best_cnt


def fill_unknown(classes: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Replace 'unknown' pixels by the majority of their known neighbours."""
    cls = classes.copy()
    unk = CID["unknown"]
    k = 3
    for _ in range(max_iter):
        mask = cls == unk
        if not mask.any():
            break
        maj, cnt = _majority(cls, k, exclude=(unk,))
        fill = mask & (cnt > 0)
        cls[fill] = maj[fill]
        if not fill.any():
            k += 2
    cls[cls == unk] = CID["grass"] if (cls != unk).any() else CID["water_deep"]
    return cls


def cleanup_classes(classes: np.ndarray, min_region_px: int) -> np.ndarray:
    """Remove isolated specks while keeping thin linear features (roads)."""
    cls = fill_unknown(classes)
    maj, _ = _majority(cls, 5)
    own = np.zeros(cls.shape, np.float32)
    for c in range(N_CLASSES):
        m = cls == c
        if m.any():
            own[m] = cv2.boxFilter(m.astype(np.float32), -1, (5, 5), normalize=False,
                                   borderType=cv2.BORDER_REPLICATE)[m]
    isolated = own <= 3
    cls[isolated] = maj[isolated]
    # small connected components -> surrounding majority
    if min_region_px > 1:
        for c in range(N_CLASSES):
            m = (cls == c).astype(np.uint8)
            if not m.any():
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_region_px)[0] + 1
            if small.size == 0:
                continue
            sm = np.isin(lab, small)
            tmp = cls.copy()
            tmp[sm] = CID["unknown"]
            tmp = fill_unknown(tmp)
            cls[sm] = tmp[sm]
    return cls


# ------------------------------------------------------------------ structure
@dataclass
class Structure:
    classes: np.ndarray          # final classes (bridges replaced by water)
    bridge: np.ndarray           # bool, road/path crossing water
    ocean: np.ndarray            # bool, water connected to the map edge
    lake_id: np.ndarray          # int32, >0 for inland water bodies
    land: np.ndarray             # bool


def remove_background_artifacts(cls: np.ndarray) -> np.ndarray:
    """Land specks that touch no real water (only the background) are not part
    of the map – e.g. watermarks or logos in the image corner.  -> background."""
    cls = cls.copy()
    void = cls == CID["void"]
    realwater = (cls == CID["water_deep"]) | (cls == CID["water_shallow"])
    land = (~void & ~realwater).astype(np.uint8)
    n, lab = cv2.connectedComponents(land, connectivity=8)
    if n <= 1:
        return cls
    near_water = cv2.dilate(realwater.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    keep = np.unique(lab[near_water & (lab > 0)])
    drop = (lab > 0) & ~np.isin(lab, keep)
    cls[drop] = CID["void"]
    return cls


def absorb_enclosed_in_towns(cls: np.ndarray, max_px: int = 400, ring_share: float = 0.7) -> np.ndarray:
    """Small patches of another class that are (almost) completely surrounded by
    built-up area are roof colours (e.g. turquoise glass roofs read as 'water',
    red roofs read as 'blossom') -> part of the town."""
    cls = cls.copy()
    urban = cls == CID["urban"]
    if not urban.any():
        return cls
    # (a) town hull: closed + hole-filled built-up area.  Roof-like colours inside
    #     it (red roofs read as 'magic', dark roofs as 'rock', glass as 'water')
    #     belong to the town.  Parks (grass/forest/sand) are left alone.
    hull = cv2.morphologyEx(urban.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ff = np.pad(hull, 1)
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    hull = (ff[1:-1, 1:-1] != 2)
    big = cv2.morphologyEx(urban.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hull.astype(np.uint8), connectivity=8)
    towns = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 150]) & \
        cv2.dilate(big.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    towns = cv2.morphologyEx(towns.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool) & hull
    rooflike = np.isin(cls, [CID[k] for k in ("magic", "rock", "dark_rock", "blossom", "desert", "snow",
                                               "water_shallow", "water_deep")])
    # water that is part of the open sea is never a roof
    sea_n, sea_lab = cv2.connectedComponents(WATER_SET[cls].astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([sea_lab[0], sea_lab[-1], sea_lab[:, 0], sea_lab[:, -1]]))
    ocean = np.isin(sea_lab, border[border > 0])
    cls[towns & rooflike & ~ocean] = CID["urban"]
    urban = cls == CID["urban"]
    # (b) small enclosed patches anywhere
    keep_cls = (CID["road"], CID["path"], CID["urban"])
    for c in range(N_CLASSES):
        if c in keep_cls:
            continue
        m = (cls == c).astype(np.uint8)
        if not m.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > max_px:
                continue
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(cls.shape[1], x + w + 2), min(cls.shape[0], y + h + 2)
            sub = lab[y0:y1, x0:x1] == i
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~sub
            if ring.any() and urban[y0:y1, x0:x1][ring].mean() >= ring_share:
                cls[y0:y1, x0:x1][sub] = CID["urban"]
    return cls


def derive_structure(cls_in: np.ndarray) -> Structure:
    cls = absorb_enclosed_in_towns(remove_background_artifacts(cls_in))
    # roads: dirt-coloured lines inside / next to built-up areas become asphalt
    urban = (cls == CID["urban"]).astype(np.uint8)
    if urban.any():
        near_urban = cv2.dilate(urban, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        cls[(cls == CID["path"]) & near_urban] = CID["road"]

    water = WATER_SET[cls]
    # bridges: road/path pixels surrounded mostly by water
    wf = cv2.blur(water.astype(np.float32), (9, 9))
    roadlike = (cls == CID["path"]) | (cls == CID["road"])
    bridge = roadlike & (wf >= 0.40)
    if bridge.any():
        # extend bridges a little along the road so decks reach the shore
        grown = cv2.dilate(bridge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        bridge = grown & (roadlike | water)
        cls[bridge & roadlike] = CID["water_shallow"]
        water = WATER_SET[cls]

    # ocean = water connected to the border; everything else wet = lakes / rivers
    n, lab = cv2.connectedComponents(water.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    ocean = np.isin(lab, border)
    lake = water & ~ocean
    lake_id = np.where(lake, lab, 0).astype(np.int32)
    return Structure(classes=cls, bridge=bridge, ocean=ocean, lake_id=lake_id, land=~water)


# ------------------------------------------------------------------ trees
TREE_CLASSES = ("grass", "forest", "blossom", "magic", "snow", "desert")


def detect_trees(img: np.ndarray, cls: np.ndarray, min_strength: float = 2.5):
    """Find individual tree crowns.

    Green maps draw trees as darker round blobs; blossom trees as pink blobs.
    We use a multi-scale difference-of-Gaussians on lightness and keep local
    extrema inside vegetated classes.  Returns array of (u, v, radius_px, class_id)
    in source-pixel coordinates.
    """
    veg = np.isin(cls, [CID[k] for k in TREE_CLASSES])
    lab = _to_lab(img)
    L = lab[..., 0]
    a = lab[..., 1]
    out = []
    taken = np.zeros(cls.shape, bool)
    for sigma in (2.2, 1.5, 1.0):                     # large crowns first
        g1 = cv2.GaussianBlur(L, (0, 0), sigma)
        g2 = cv2.GaussianBlur(L, (0, 0), sigma * 2.2)
        dark = g2 - g1                                # >0 where darker than surroundings
        ga1 = cv2.GaussianBlur(a, (0, 0), sigma)
        ga2 = cv2.GaussianBlur(a, (0, 0), sigma * 2.2)
        pink = ga1 - ga2                              # >0 where more magenta than surroundings
        score = np.maximum(dark, pink * 0.8)
        k = max(3, int(round(sigma * 2)) * 2 + 1)
        peak = (score >= cv2.dilate(score, np.ones((k, k), np.uint8))) & (score > min_strength) & veg & ~taken
        vs, us = np.nonzero(peak)
        r = sigma * 1.6
        for u, v in zip(us, vs):
            rr = max(1, int(round(r * 0.7)))
            win = img[max(0, v - rr):v + rr + 1, max(0, u - rr):u + rr + 1].reshape(-1, 3)
            R, G, B = np.median(win, axis=0)
            out.append((u + 0.5, v + 0.5, r, int(cls[v, u]), R, G, B))
        if len(us):
            m = np.zeros(cls.shape, np.uint8)
            for u, v in zip(us, vs):
                cv2.circle(m, (int(u), int(v)), int(math.ceil(r)), 1, -1)
            taken |= m > 0
    if not out:
        return np.zeros((0, 7), np.float32)
    return np.array(out, np.float32)


# ------------------------------------------------------------------ buildings
@dataclass
class Building:
    id: int
    polygon: list                  # [[u,v],...] source-pixel coordinates
    area_px: float
    roof_rgb: list
    storeys: int = 1
    kind: str = "house"           # house | platform | none
    roof_block: str = ""
    wall_block: str = ""
    estimated: bool = True         # height not known from the map -> flagged
    name: str = ""

    def centroid(self):
        p = np.array(self.polygon, np.float32)
        return float(p[:, 0].mean()), float(p[:, 1].mean())


# Roof colours -> blocks (concrete/terracotta colours are close to map colours)
ROOF_BLOCKS = {
    "minecraft:white_concrete": (207, 213, 214), "minecraft:light_gray_concrete": (125, 125, 115),
    "minecraft:gray_concrete": (55, 58, 62), "minecraft:black_concrete": (8, 10, 15),
    "minecraft:red_concrete": (142, 33, 33), "minecraft:orange_concrete": (224, 97, 1),
    "minecraft:yellow_concrete": (241, 175, 21), "minecraft:lime_concrete": (94, 169, 24),
    "minecraft:green_concrete": (73, 91, 36), "minecraft:cyan_concrete": (21, 119, 136),
    "minecraft:light_blue_concrete": (36, 137, 199), "minecraft:blue_concrete": (45, 47, 143),
    "minecraft:purple_concrete": (100, 32, 156), "minecraft:pink_concrete": (214, 101, 143),
    "minecraft:brown_concrete": (96, 60, 32), "minecraft:terracotta": (152, 94, 68),
    "minecraft:white_terracotta": (210, 178, 161), "minecraft:light_gray_terracotta": (135, 107, 98),
    "minecraft:red_terracotta": (143, 61, 47), "minecraft:brown_terracotta": (77, 51, 36),
    "minecraft:smooth_stone": (159, 159, 159), "minecraft:stone_bricks": (122, 122, 122),
    "minecraft:dark_oak_planks": (67, 43, 20), "minecraft:spruce_planks": (115, 85, 49),
    "minecraft:oak_planks": (162, 131, 79), "minecraft:quartz_block": (236, 230, 223),
}
_roof_names = list(ROOF_BLOCKS)
_roof_lab = _to_lab(np.array([list(ROOF_BLOCKS.values())], np.uint8))[0]


def nearest_block_color(rgb) -> str:
    lab = _to_lab(np.array([[rgb]], np.uint8))[0, 0]
    d = ((_roof_lab - lab) ** 2).sum(1)
    return _roof_names[int(np.argmin(d))]


def detect_buildings(img: np.ndarray, cls: np.ndarray, water: np.ndarray, min_px: int,
                     default_storeys: int = 1, town_px: int = 150, roof_delta: float = 13.0) -> list[Building]:
    """Find individual roofs.

    * small isolated built-up patches (< town_px) are one building each,
    * in larger built-up areas the paving colour is estimated (median colour of
      the area) and only pixels clearly different from it are roofs; colour
      edges and roads additionally separate neighbouring roofs.
    Everything else in a town becomes paved ground.
    """
    urban = (cls == CID["urban"]).astype(np.uint8)
    if not urban.any():
        return []
    lab = _to_lab(cv2.medianBlur(img, 3))
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(urban, connectivity=8)
    roofs = np.zeros(urban.shape, bool)
    for i in range(1, n0):
        x, y, w, h, area = st0[i]
        sub = lab0[y:y + h, x:x + w] == i
        if area < town_px:
            roofs[y:y + h, x:x + w] |= sub
            continue
        pav = np.median(lab[y:y + h, x:x + w][sub], axis=0)
        d = np.linalg.norm(lab[y:y + h, x:x + w] - pav, axis=-1)
        roofs[y:y + h, x:x + w] |= sub & (d > roof_delta)
    # roofs are separated by paving (already excluded) and by roads; colour edges
    # inside a roof would only fragment it, so they are not used.
    sep = (cls == CID["road"]) | (cls == CID["path"])
    parts = roofs & ~sep
    parts = cv2.morphologyEx(parts.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labm, stats, _ = cv2.connectedComponentsWithStats(parts, connectivity=4)
    wf = cv2.blur(water.astype(np.float32), (15, 15))
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_px:
            continue
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(urban.shape[1], x + w + 1), min(urban.shape[0], y + h + 1)
        m = np.zeros(urban.shape, np.uint8)
        m[y0:y1, x0:x1] = cv2.dilate((labm[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((3, 3), np.uint8))
        m &= roofs.astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2).astype(float)
        if len(poly) < 3:
            continue
        ys, xs = np.nonzero(m)
        roof = np.median(img[ys, xs], axis=0).astype(int).tolist()
        cy, cx = int(ys.mean()), int(xs.mean())
        kind = "platform" if wf[cy, cx] > 0.5 else "house"
        out.append(Building(id=len(out) + 1, polygon=poly.tolist(), area_px=float(area), roof_rgb=roof,
                            storeys=default_storeys, kind=kind, roof_block=nearest_block_color(roof),
                            wall_block="", estimated=True))
    return out


# natural colours an isolated object must NOT resemble (grass, tree crowns incl. the
# teal crowns of stylised maps, blossoms, sand, desert)
_NATURAL_RGB = np.array([[121, 144, 59], [103, 135, 55], [86, 110, 49], [70, 95, 42], [90, 150, 130],
                         [70, 130, 120], [100, 160, 140], [60, 120, 110], [120, 170, 150], [96, 154, 91],
                         [150, 190, 110], [230, 150, 200], [205, 120, 170], [243, 216, 176], [235, 205, 160],
                         [212, 136, 95], [241, 167, 117]], np.uint8)


def detect_objects(img: np.ndarray, cls: np.ndarray, existing: list, min_px: int = 8, contrast: float = 24.0,
                   default_storeys: int = 1, max_px: int = 900) -> list:
    """Isolated houses outside towns (and ships/huts on water).

    A building on a stylised map is a compact blob that
      * clearly differs from its surroundings (Lab distance to a 21 px median),
      * does not look like vegetation, sand or desert,
      * is roughly rectangular (fill of its min-area rectangle >= 0.6, elongation <= 3),
      * has a fairly uniform colour (rocks are textured),
      * is not a small rock lying on the shoreline,
      * is not green-dominant (light bushes).
    """
    lab8 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = cv2.cvtColor(cv2.medianBlur(img, 3), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = np.stack([cv2.medianBlur(np.ascontiguousarray(lab8[..., i]), 21) for i in range(3)], -1).astype(np.float32)
    diff = np.linalg.norm(lab - bg, axis=-1)
    nat = cv2.cvtColor(_NATURAL_RGB[None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    dnat = np.full(diff.shape, np.inf, np.float32)
    for c in nat:
        np.minimum(dnat, np.linalg.norm(lab - c, axis=-1), out=dnat)
    water = WATER_SET[cls]
    shore = cv2.dilate((water | (cls == CID["sand"])).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cand = (diff > contrast) & (dnat > 18) & ~np.isin(cls, [CID["path"], CID["road"], CID["void"]])
    taken = np.zeros(cls.shape, np.uint8)
    for b in existing:
        cv2.fillPoly(taken, [np.array(b.polygon, np.int32)], 1)
    cand &= cv2.dilate(taken, np.ones((3, 3), np.uint8)) == 0
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    n, labm, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    wf = cv2.blur(water.astype(np.float32), (9, 9))
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_px or a > max_px:
            continue
        m = labm[y:y + h, x:x + w] == i
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if a / max(rw * rh, 1) < 0.6 or max(rw, rh) / max(1.0, min(rw, rh)) > 3:
            continue
        if shore[y:y + h, x:x + w][m].mean() > 0.25 and a < 40:
            continue
        if lab[y:y + h, x:x + w][m].std(0).mean() > 14:
            continue
        rgb = np.median(img[y:y + h, x:x + w][m], axis=0)
        if rgb[1] > rgb[0] + 4 and rgb[1] > rgb[2] + 4:
            continue
        poly = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2).astype(float) + [x, y]
        if len(poly) < 3:
            continue
        cy, cx = int(y + h / 2), int(x + w / 2)
        kind = "platform" if wf[cy, cx] > 0.5 else "house"
        roof = rgb.astype(int).tolist()
        out.append(Building(id=0, polygon=poly.tolist(), area_px=float(a), roof_rgb=roof, storeys=default_storeys,
                            kind=kind, roof_block=nearest_block_color(roof), wall_block="", estimated=True,
                            name="Einzelobjekt"))
    return out


def drop_tree_crowns(objects: list, trees: np.ndarray) -> list:
    """Remove single objects that are really coloured tree crowns.

    Fantasy maps draw purple / magenta / pink trees as compact uniform blobs that the
    object detector would take for houses.  An object is a tree when the tree detector
    found a crown inside it and its colour is in the purple-pink family."""
    if not len(trees) or not objects:
        return objects
    out = []
    for b in objects:
        r, g, bl = (int(v) for v in b.roof_rgb)
        if bl - g >= 15 and r - g >= 15:
            p = np.array(b.polygon, np.float32)
            if any(cv2.pointPolygonTest(p, (float(t[0]), float(t[1])), False) >= 0 for t in trees):
                continue
        out.append(b)
    return out


def class_preview(cls: np.ndarray) -> np.ndarray:
    from .config import CLASSES
    lut = np.array([c[2] for c in CLASSES], np.uint8)
    return lut[cls]
