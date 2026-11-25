"""Project folder: holds the source image, settings, manual edits and all
intermediate results, so work can be resumed at any time.

    <project>/project.json
    <project>/source.png                 copy of the map image
    <project>/settings.json palette.json prototypes.json
    <project>/edits/class_edit.png       manual class paint (255 = no edit)
    <project>/edits/height_delta.npy     raise/lower brush (blocks)
    <project>/edits/height_override.npy  imported height map (optional)
    <project>/edits/buildings.json       per-building edits (storeys, type, ...)
    <project>/cache/…                    classification checkpoint
    <project>/data/…                     final arrays read by the generator
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

from . import analysis, terrain
from .config import CID, CLASS_KEYS, CLASSES, Settings, default_palette, default_prototypes
from .generator import Geometry, compute_geometry, generate_world
from .imageio import load_map_image


def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


class Project:
    def __init__(self, folder: str):
        self.dir = os.path.abspath(folder)
        for sub in ("edits", "cache", "data"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.settings = Settings.from_dict(_load_json(self.p("settings.json"), {}))
        self.palette = default_palette()
        self.palette.update(_load_json(self.p("palette.json"), {}))
        self.prototypes = default_prototypes()
        self.prototypes.update(_load_json(self.p("prototypes.json"), {}))
        self.image = load_map_image(self.p("source.png")) if os.path.exists(self.p("source.png")) else None
        h, w = self.image.shape[:2] if self.image is not None else (0, 0)
        ce = self.p("edits", "class_edit.png")
        self.class_edit = (np.array(Image.open(ce)) if os.path.exists(ce) else np.full((h, w), 255, np.uint8))
        hd = self.p("edits", "height_delta.npy")
        self.height_delta = np.load(hd) if os.path.exists(hd) else np.zeros((h, w), np.float32)
        ho = self.p("edits", "height_override.npy")
        self.height_override = np.load(ho) if os.path.exists(ho) else None
        self.building_edits: dict = _load_json(self.p("edits", "buildings.json"), {})
        # results (filled by run_* methods or loaded from disk)
        self.class_auto = self.review = None
        self.classes = self.height = self.lake = self.bridge = self.land = None
        self.trees_px = None
        self.height_base = None
        self.buildings: list[analysis.Building] = []
        self._load_results()

    # ------------------------------------------------------------ paths / io
    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    @classmethod
    def create(cls, image_path: str, folder: str) -> "Project":
        img = load_map_image(image_path)            # validates
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img).save(os.path.join(folder, "source.png"))
        _save_json(os.path.join(folder, "project.json"), {"format": 1, "original": os.path.abspath(image_path)})
        prj = cls(folder)
        # default scale: island about 2.5 km wide (typical battle-royale island)
        try:
            cls_auto, _ = analysis.classify_colors(img[::4, ::4], prj.prototypes, prj.settings.max_color_distance)
            land = ~analysis.WATER_SET[analysis.remove_background_artifacts(cls_auto)]
            xs = np.nonzero(land.any(0))[0]
            if xs.size:
                prj.settings.blocks_per_pixel = round(max(1.0, min(8.0, 2500 / ((xs.max() - xs.min() + 1) * 4))), 2)
        except Exception:
            pass
        prj.save()
        return prj

    def save(self):
        self.settings.save(self.p("settings.json"))
        _save_json(self.p("palette.json"), self.palette)
        _save_json(self.p("prototypes.json"), self.prototypes)
        self.save_edits()

    def save_edits(self):
        if self.image is None:
            return
        Image.fromarray(self.class_edit).save(self.p("edits", "class_edit.png"))
        _save_npy(self.p("edits", "height_delta.npy"), self.height_delta.astype(np.float32))
        if self.height_override is not None:
            _save_npy(self.p("edits", "height_override.npy"), self.height_override.astype(np.float32))
        _save_json(self.p("edits", "buildings.json"), self.building_edits)

    def _load_results(self):
        c = self.p("cache", "class_auto.npy")
        if os.path.exists(c) and os.path.exists(self.p("cache", "review.npy")):
            self.class_auto = np.load(c)
            self.review = np.load(self.p("cache", "review.npy"))
        d = self.p("data")
        try:
            self.classes = np.load(os.path.join(d, "classes.npy"))
            self.height = np.load(os.path.join(d, "height.npy"))
            hb = os.path.join(d, "height_base.npy")
            self.height_base = np.load(hb) if os.path.exists(hb) else self.height.copy()
            self.lake = np.load(os.path.join(d, "lake.npy"))
            self.bridge = np.load(os.path.join(d, "bridge.npy")).astype(bool)
            self.land = np.load(os.path.join(d, "land.npy")).astype(bool)
            self.trees_px = np.load(os.path.join(d, "trees_px.npy"))
            self.buildings = [analysis.Building(**b) for b in _load_json(os.path.join(d, "buildings.json"), [])]
        except (OSError, ValueError, TypeError):
            # incomplete / missing analysis -> treat as not analysed
            self.classes = self.height = self.lake = self.bridge = self.land = None

    # ------------------------------------------------------------ pipeline
    def _class_key(self):
        return hashlib.sha1(json.dumps([self.prototypes, self.settings.max_color_distance],
                                       sort_keys=True).encode()).hexdigest()

    def run_classification(self, progress=None, force=False):
        key_path = self.p("cache", "class_key.txt")
        key = self._class_key()
        if (not force and self.class_auto is not None and os.path.exists(key_path)
                and open(key_path).read() == key):
            if progress:
                progress(1.0)
            return
        self.class_auto, self.review = analysis.classify_colors(
            self.image, self.prototypes, self.settings.max_color_distance, progress)
        _save_npy(self.p("cache", "class_auto.npy"), self.class_auto)
        _save_npy(self.p("cache", "review.npy"), self.review)
        with open(key_path, "w") as f:
            f.write(key)

    def edited_classes(self):
        cls = self.class_auto.copy()
        m = self.class_edit != 255
        cls[m] = self.class_edit[m]
        return cls

    def run_derive(self, progress=None, log=None):
        """Cleanup, structure, heights, trees and buildings (applies manual edits)."""
        log = log or (lambda s: None)
        cfg = self.settings
        if self.class_auto is None:
            self.run_classification()
        cls = self.edited_classes()
        manual = self.class_edit != 255
        clean = analysis.cleanup_classes(cls, cfg.min_region_px)
        clean[manual] = cls[manual]                    # manual paint always wins
        if progress:
            progress(0.25)
        st = analysis.derive_structure(clean)
        h, lake = terrain.compute_heights(self.image, st, cfg)
        if self.height_override is not None and self.height_override.shape != h.shape:
            self.height_override = cv2.resize(self.height_override, (h.shape[1], h.shape[0]))
        self.height_base = h
        _save_npy(self.p("data", "height_base.npy"), h.astype(np.float32))
        h = terrain.apply_edits(h, self.height_delta, self.height_override, st.land)
        if progress:
            progress(0.55)
        trees = analysis.detect_trees(self.image, st.classes) if cfg.detect_single_trees else np.zeros((0, 7), np.float32)
        if progress:
            progress(0.75)
        blds = analysis.detect_buildings(self.image, st.classes, ~st.land, cfg.min_building_px, cfg.default_storeys)
        if cfg.detect_single_houses:
            objs = analysis.detect_objects(self.image, st.classes, blds, cfg.object_min_px, cfg.object_contrast,
                                           cfg.default_storeys)
            blds += analysis.drop_tree_crowns(objs, trees)
        for i, b in enumerate(blds, start=1):
            b.id = i
        self._apply_building_edits(blds)
        self.classes, self.height, self.lake, self.bridge, self.land = st.classes, h, lake, st.bridge, st.land
        self.trees_px, self.buildings = trees, blds
        d = self.p("data")
        _save_npy(os.path.join(d, "classes.npy"), st.classes)
        _save_npy(os.path.join(d, "height.npy"), h.astype(np.float32))
        _save_npy(os.path.join(d, "lake.npy"), lake.astype(np.float32))
        _save_npy(os.path.join(d, "bridge.npy"), st.bridge.astype(np.uint8))
        _save_npy(os.path.join(d, "land.npy"), st.land.astype(np.uint8))
        _save_npy(os.path.join(d, "trees_px.npy"), trees)
        _save_json(os.path.join(d, "buildings.json"), [b.__dict__ for b in blds])
        log(f"Analyse fertig: {len(trees)} Einzelbäume, {len(blds)} Gebäude, "
            f"{int(st.bridge.sum())} Brücken-Pixel, {len(np.unique(st.lake_id)) - 1} Binnengewässer.")
        if progress:
            progress(1.0)

    def apply_height_edits(self, save=True):
        """Recompute the final height from the estimate + brush edits (no re-analysis needed)."""
        if self.height_base is None:
            return
        self.height = terrain.apply_edits(self.height_base, self.height_delta, self.height_override, self.land)
        if save:
            _save_npy(self.p("data", "height.npy"), self.height)
            self.save_edits()

    def import_heightmap(self, path: str, min_blocks: float, max_blocks: float):
        """Use a real height map (grayscale image, 8 or 16 bit) instead of the estimate on land."""
        im = np.array(Image.open(path))
        if im.ndim == 3:
            im = im[..., :3].mean(-1)
        im = im.astype(np.float32)
        im = (im - im.min()) / max(1e-6, float(im.max() - im.min()))
        H, W = self.image.shape[:2]
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
        self.height_override = min_blocks + im * (max_blocks - min_blocks)
        self.apply_height_edits()

    def clear_heightmap(self):
        self.height_override = None
        try:
            os.remove(self.p("edits", "height_override.npy"))
        except OSError:
            pass
        self.apply_height_edits()

    def building_at(self, u: float, v: float, tolerance: float = 3.0):
        """Building under (u, v) or the nearest one within `tolerance` source pixels."""
        best, best_d = None, -tolerance
        for b in self.buildings:
            d = cv2.pointPolygonTest(np.array(b.polygon, np.float32), (float(u), float(v)), True)
            if d >= best_d:          # inside -> positive, outside -> negative distance
                best, best_d = b, d
        return best

    # ------------------------------------------------------------ buildings
    @staticmethod
    def _bkey(b: analysis.Building):
        u, v = b.centroid()
        return f"{round(u)},{round(v)}"

    def _apply_building_edits(self, blds):
        """Apply per-building edits and append manually drawn buildings."""
        for a in self.building_edits.get("added", []):
            blds.append(analysis.Building(id=len(blds) + 1, polygon=a["polygon"], area_px=a["area_px"],
                                          roof_rgb=a.get("roof_rgb", [150, 150, 150]),
                                          storeys=a.get("storeys", self.settings.default_storeys),
                                          kind=a.get("kind", "house"), roof_block=a.get("roof_block", ""),
                                          wall_block=a.get("wall_block", ""), estimated=a.get("estimated", True),
                                          name=a.get("name", "manuell")))
        keys = []
        for k in self.building_edits:
            if k == "added":
                continue
            u, v = map(float, k.split(","))
            keys.append((u, v, k))
        for b in blds:
            u, v = b.centroid()
            if not keys or b.name == "manuell":
                continue
            best = min(keys, key=lambda t: (t[0] - u) ** 2 + (t[1] - v) ** 2)
            if (best[0] - u) ** 2 + (best[1] - v) ** 2 <= 9:
                for f, val in self.building_edits[best[2]].items():
                    setattr(b, f, val)
                b.estimated = False if "storeys" in self.building_edits[best[2]] else b.estimated

    def add_building(self, u0, v0, u1, v1, storeys=None) -> analysis.Building:
        """Manually draw a building (rectangle in source pixels). Survives re-analysis."""
        poly = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
        y0, y1 = max(0, int(v0)), max(int(v0) + 1, int(np.ceil(v1)))
        x0, x1 = max(0, int(u0)), max(int(u0) + 1, int(np.ceil(u1)))
        roof = np.median(self.image[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(int).tolist()
        rec = dict(polygon=poly, area_px=float((u1 - u0) * (v1 - v0)), roof_rgb=roof,
                   storeys=int(storeys or self.settings.default_storeys), kind="house",
                   roof_block=analysis.nearest_block_color(roof), wall_block="", estimated=True, name="manuell")
        self.building_edits.setdefault("added", []).append(rec)
        b = analysis.Building(id=len(self.buildings) + 1, **rec)
        self.buildings.append(b)
        _save_json(self.p("edits", "buildings.json"), self.building_edits)
        _save_json(self.p("data", "buildings.json"), [x.__dict__ for x in self.buildings])
        return b

    def edit_building(self, b: analysis.Building, **changes):
        if b.name == "manuell":                      # drawn by hand: edit the stored record
            for a in self.building_edits.get("added", []):
                if a["polygon"] == b.polygon:
                    a.update(changes)
                    if "storeys" in changes:
                        a["estimated"] = False
            for f, val in changes.items():
                setattr(b, f, val)
            if "storeys" in changes:
                b.estimated = False
            _save_json(self.p("edits", "buildings.json"), self.building_edits)
            _save_json(self.p("data", "buildings.json"), [x.__dict__ for x in self.buildings])
            return
        for f, val in changes.items():
            setattr(b, f, val)
        if "storeys" in changes:
            b.estimated = False
        self.building_edits.setdefault(self._bkey(b), {}).update(changes)
        _save_json(self.p("edits", "buildings.json"), self.building_edits)
        _save_json(self.p("data", "buildings.json"), [x.__dict__ for x in self.buildings])

    # ------------------------------------------------------------ world data
    def geometry(self) -> Geometry:
        return compute_geometry(self.land, self.settings)

    def prepare_world_data(self):
        """Convert trees/buildings to world coordinates and choose a spawn point."""
        cfg, geo = self.settings, self.geometry()
        s = cfg.blocks_per_pixel
        d = self.p("data")
        T = self.trees_px
        if T is not None and len(T):
            X, Z = geo.px_to_world(T[:, 0], T[:, 1])
            if T.shape[1] >= 7:
                cols = T[:, 4:7]
            else:   # older analysis without crown colours: sample the image at the tree centre
                iu = np.clip(T[:, 0].astype(int), 0, self.image.shape[1] - 1)
                iv = np.clip(T[:, 1].astype(int), 0, self.image.shape[0] - 1)
                cols = self.image[iv, iu].astype(np.float32)
            tw = np.concatenate([np.stack([np.floor(X), np.floor(Z), T[:, 2] * s, T[:, 3]], 1), cols], 1).astype(np.float32)
        else:
            tw = np.zeros((0, 7), np.float32)
        _save_npy(os.path.join(d, "image.npy"), self.image)
        _save_npy(os.path.join(d, "trees_world.npy"), tw)
        out = []
        H, W = self.height.shape
        for b in self.buildings:
            poly = np.array(b.polygon, np.float64) + 0.5
            X, Z = geo.px_to_world(poly[:, 0], poly[:, 1])
            pw = np.stack([X, Z], 1)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.array(b.polygon, np.int32)], 1)
            hv = self.height[m > 0]
            base_rel = float(np.median(hv)) if hv.size else 1.0
            base_y = cfg.water_level + max(1, int(round(base_rel)))
            if not cfg.generate_buildings:
                break
            out.append(dict(id=b.id, kind=b.kind, storeys=int(b.storeys), roof_block=b.roof_block,
                            wall_block=b.wall_block, base_y=base_y, polygon_world=pw.tolist(),
                            bbox=[float(X.min()), float(Z.min()), float(X.max()), float(Z.max())]))
        _save_json(os.path.join(d, "buildings_world.json"), out)
        _save_json(os.path.join(d, "geometry.json"), geo.to_dict())
        return geo, self.find_spawn(geo)

    def find_spawn(self, geo: Geometry):
        cls, h = self.classes, self.height
        good = self.land & np.isin(cls, [CID["grass"], CID["path"], CID["sand"], CID["forest"], CID["desert"]])
        if not good.any():
            good = self.land
        if not good.any():
            return (0, self.settings.water_level + 2, 0)
        ys, xs = np.nonzero(good)
        cy, cx = ys.mean(), xs.mean()
        i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        X, Z = geo.px_to_world(xs[i] + 0.5, ys[i] + 0.5)
        y = self.settings.water_level + int(round(h[ys[i], xs[i]])) + 3
        return int(X), y, int(Z)

    def report(self) -> dict:
        unk = float((self.class_auto == CID["unknown"]).mean() * 100) if self.class_auto is not None else 0
        rev = float(self.review.mean() * 100) if self.review is not None else 0
        est = sum(1 for b in self.buildings if b.estimated)
        counts = {CLASS_KEYS[i]: int((self.classes == i).sum()) for i in range(len(CLASSES))} if self.classes is not None else {}
        return {
            "hinweise": [
                "Höhen sind aus Küstenabstand, Geländeklasse und Kartenschattierung geschätzt – "
                "das Kartenbild enthält keine echten Höhendaten. Für exakte Höhen eine Höhenkarte importieren.",
                f"{est} Gebäudehöhen sind geschätzt (Standard {self.settings.default_storeys} Stockwerk(e)) – "
                "in der Gebäudeliste korrigierbar.",
                "Innenräume, Türen und Dachformen sind nicht aus dem Kartenbild ableitbar und werden schlicht gebaut.",
            ],
            "unbekannte_pixel_prozent": round(unk, 2),
            "pruefbereiche_prozent": round(rev, 2),
            "gebaeude": len(self.buildings),
            "gebaeude_hoehe_geschaetzt": est,
            "einzelbaeume": int(len(self.trees_px)) if self.trees_px is not None else 0,
            "klassen_pixel": counts,
            "einstellungen": self.settings.to_dict(),
        }

    def generate(self, world_dir, progress=None, log=None, cancel=None, only_regions=None):
        if self.classes is None:
            self.run_derive(log=log)
        geo, spawn = self.prepare_world_data()
        res = generate_world(self.p("data"), world_dir, self.settings, self.palette, geo, spawn,
                             progress=progress, log=log, cancel=cancel, only_regions=only_regions,
                             report=self.report())
        if not res.get("cancelled"):
            self.write_icon(world_dir)
        return res

    def write_icon(self, world_dir):
        prev = analysis.class_preview(self.classes)
        geo = self.geometry()
        crop = prev[geo.cv0:geo.cv1, geo.cu0:geo.cu1]
        Image.fromarray(crop).resize((64, 64), Image.LANCZOS).save(os.path.join(world_dir, "icon.png"))


def export_zip(world_dir: str, zip_path: str):
    """Zip the world folder so it can be imported (extract into .minecraft/saves)."""
    name = os.path.basename(os.path.normpath(world_dir))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(world_dir):
            for fn in files:
                if fn.endswith(".tmp") or fn == "map2mc_progress.json":
                    continue
                full = os.path.join(root, fn)
                z.write(full, os.path.join(name, os.path.relpath(full, world_dir)))


def minecraft_saves_dir() -> str:
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA", ""), ".minecraft", "saves")
    return os.path.expanduser("~/.minecraft/saves")


def copy_world(world_dir: str, dest_parent: str) -> str:
    dest = os.path.join(dest_parent, os.path.basename(os.path.normpath(world_dir)))
    if os.path.exists(dest):
        raise FileExistsError(dest)
    shutil.copytree(world_dir, dest, ignore=shutil.ignore_patterns("*.tmp", "map2mc_progress.json"))
    return dest
