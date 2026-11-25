"""Core tests: NBT, Anvil packing, image validation and a small end-to-end world.

    python -m unittest discover -s tests -v
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from map2mc import anvil, nbt  # noqa: E402
from map2mc.imageio import MapImageError, load_map_image  # noqa: E402


def top_block(chunk, x, z):
    best = None
    for s in chunk["sections"]:
        bs = s["block_states"]
        pal = bs["palette"]
        if "data" in bs:
            bits = max(4, int(np.ceil(np.log2(len(pal)))))
            idx = anvil.unpack_indices(bs["data"], bits, 4096).reshape(16, 16, 16)
        else:
            idx = np.zeros((16, 16, 16), int)
        for y in range(16):
            n = pal[idx[y, z, x]]["Name"]
            if n != "minecraft:air" and not n.endswith(("grass", "dandelion", "poppy", "azure_bluet",
                                                         "oxeye_daisy", "cornflower")):
                best = (s["Y"].value * 16 + y, n)
    return best


class TestNBT(unittest.TestCase):
    def test_roundtrip(self):
        root = {"a": nbt.Byte(3), "b": nbt.Short(-2), "c": nbt.Int(70000), "d": nbt.Long(-5),
                "e": nbt.Float(1.5), "f": nbt.Double(2.25), "g": "häh", "h": {"x": nbt.Int(1)},
                "i": nbt.List(nbt.TAG_INT, [nbt.Int(1), nbt.Int(2)]), "j": nbt.List(nbt.TAG_COMPOUND, []),
                "k": np.array([1, -2, 3], np.int64), "l": np.array([4, 5], np.int32), "m": np.array([1, 2], np.int8)}
        back = nbt.decode(nbt.encode(root))
        self.assertEqual(nbt.plain(back), nbt.plain(root))


class TestAnvil(unittest.TestCase):
    def test_pack_unpack(self):
        rng = np.random.default_rng(1)
        for bits in range(1, 16):
            vals = rng.integers(0, 2 ** bits, 4096)
            self.assertTrue((anvil.unpack_indices(anvil.pack_indices(vals, bits), bits, 4096) == vals).all())

    def test_chunk_region_roundtrip(self):
        reg = anvil.BlockRegistry()
        ids = [reg.id(s) for s in ("minecraft:stone", "minecraft:dirt", "minecraft:oak_log[axis=y]",
                                   "minecraft:water[level=0]")]
        rng = np.random.default_rng(2)
        blocks = np.array(ids)[rng.integers(0, 4, (64, 16, 16))]
        blocks[-16:] = reg.AIR
        data = anvil.encode_chunk(3, -2, blocks, -64, reg, np.zeros((4, 4), int), ["minecraft:plains"])
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "r.0.-1.mca")
            anvil.write_region(path, {(3, 30): data})
            ch = anvil.read_region_chunk(path, 3, 30)
            self.assertEqual(ch["DataVersion"].value, 4671)
            self.assertEqual(ch["Status"], "minecraft:full")
            sec = ch["sections"][1]
            pal = [p["Name"] for p in sec["block_states"]["palette"]]
            bits = max(4, int(np.ceil(np.log2(len(pal)))))
            idx = anvil.unpack_indices(sec["block_states"]["data"], bits, 4096).reshape(16, 16, 16)
            names = np.array(pal)[idx]
            expect = np.array([reg.states[i].split("[")[0] for i in range(len(reg.states))])[blocks[16:32]]
            self.assertTrue((names == expect).all())
            self.assertIsNone(anvil.read_region_chunk(path, 0, 0))
        finally:
            shutil.rmtree(d)


class TestLitematic(unittest.TestCase):
    def test_spanning_bits(self):
        from map2mc import litematic
        for bits in (2, 3, 5, 6, 7, 11):
            v = np.random.default_rng(bits).integers(0, 1 << bits, 4099)
            w = litematic.pack_spanning(v, bits)
            self.assertEqual(len(w), (len(v) * bits + 63) // 64)
            self.assertTrue((litematic.unpack_spanning(w, bits, len(v)) == v).all())
        # entry crossing a long border: 3 bits, index 21 occupies bits 63..65
        v = np.zeros(30, np.int64)
        v[21] = 0b101
        w = litematic.pack_spanning(v, 3).view(np.uint64)
        self.assertEqual(int(w[0]) >> 63, 1)
        self.assertEqual(int(w[1]) & 0b11, 0b10)


class TestTreePack(unittest.TestCase):
    def _schem(self, path, palette, blocks, w, h, l):
        """Write a tiny Sponge v2 schematic: blocks dict (x,y,z)->palette index."""
        data = np.zeros(w * h * l, np.int8)
        for (x, y, z), i in blocks.items():
            data[(y * l + z) * w + x] = i
        root = {"Version": nbt.Int(2), "DataVersion": nbt.Int(4189), "Width": nbt.Short(w), "Height": nbt.Short(h),
                "Length": nbt.Short(l), "PaletteMax": nbt.Int(len(palette)),
                "Palette": {n: nbt.Int(i) for i, n in enumerate(palette)}, "BlockData": data,
                "Offset": np.zeros(3, np.int32)}
        nbt.write_gzip_file(path, root)

    def test_import_and_place_states(self):
        from map2mc import treepack
        d = tempfile.mkdtemp()
        try:
            pal = ["minecraft:air", "minecraft:birch_log[axis=y]", "minecraft:birch_leaves[distance=3,persistent=false,waterlogged=false]"]
            blocks = {(2, y, 2): 1 for y in range(4)}
            for x in range(5):
                for z in range(5):
                    blocks.setdefault((x, 4, z), 2)
            os.makedirs(os.path.join(d, "pack", "Birch"))
            self._schem(os.path.join(d, "pack", "Birch", "t1.schem"), pal, blocks, 5, 5, 5)
            vol, names = treepack.read_schem(open(os.path.join(d, "pack", "Birch", "t1.schem"), "rb").read())
            self.assertEqual(vol.shape, (5, 5, 5))
            self.assertEqual(names[vol[0, 2, 2]], "minecraft:birch_log[axis=y]")
            info = treepack.import_pack(os.path.join(d, "pack"), os.path.join(d, "lib.m2trees"))
            self.assertEqual(info["trees"], 1)
            lib = treepack.Library(info["path"])
            t = lib.trees[0]
            self.assertEqual(t["kind"], "birch")
            self.assertTrue(all("persistent=true" in s for s in t["palette"] if "leaves" in s))
            dx, dy, dz, pid, states, leaf = lib.blocks_of(0, 1)
            self.assertEqual(int(dy.min()), 0)                       # trunk base sits at y 0
            self.assertEqual(lib.choose("birch", 2.0, 7), 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_rotate_and_swap(self):
        from map2mc import treepack
        self.assertEqual(treepack.rotate_state("minecraft:oak_log[axis=x]", 1), "minecraft:oak_log[axis=z]")
        self.assertEqual(treepack.rotate_state("minecraft:ladder[facing=north,waterlogged=false]", 1),
                         "minecraft:ladder[facing=east,waterlogged=false]")
        self.assertEqual(treepack.fix_state("minecraft:grass"), "minecraft:short_grass")
        self.assertEqual(treepack.swap_wood("minecraft:oak_leaves[distance=7,persistent=true]", "cherry"),
                         "minecraft:cherry_leaves[distance=7,persistent=true]")


class TestLeaves(unittest.TestCase):
    def test_kind_from_rgb(self):
        from map2mc.features import kind_from_rgb
        self.assertEqual(kind_from_rgb((230, 150, 200)), "cherry")
        self.assertEqual(kind_from_rgb((90, 150, 130)), "spruce")
        self.assertIn(kind_from_rgb((96, 154, 91)), ("oak", "birch"))


class TestTreeCrownObjects(unittest.TestCase):
    def test_purple_crown_is_not_a_house(self):
        from map2mc.analysis import Building, drop_tree_crowns
        sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
        purple = Building(id=1, polygon=sq, area_px=100, roof_rgb=[115, 78, 150])
        grey = Building(id=2, polygon=[[20, 0], [30, 0], [30, 10], [20, 10]], area_px=100, roof_rgb=[120, 118, 115])
        lone = Building(id=3, polygon=[[40, 0], [50, 0], [50, 10], [40, 10]], area_px=100, roof_rgb=[115, 78, 150])
        trees = np.array([[5, 5, 3, 0, 115, 78, 150], [25, 5, 3, 0, 0, 0, 0]], np.float32)
        kept = drop_tree_crowns([purple, grey, lone], trees)
        self.assertEqual([b.id for b in kept], [2, 3])   # grey roof stays, purple without a tree stays


class TestLanguage(unittest.TestCase):
    def test_translation(self):
        from map2mc import i18n
        try:
            i18n.set_language("en")
            self.assertEqual(i18n.tr("Generierung starten"), "Start generation")
            self.assertEqual(i18n.tr("FEHLER: Datei nicht gefunden: a.png"), "ERROR: File not found: a.png")
            self.assertEqual(i18n.tr("Gebäude 3: house, 2 Stockwerk(e)."), "Building 3: house, 2 storey(s).")
            i18n.set_language("de")
            self.assertEqual(i18n.tr("Generierung starten"), "Generierung starten")
        finally:
            i18n.set_language(i18n.DEFAULT)


class TestImages(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_bad_files(self):
        empty = os.path.join(self.d, "e.png")
        open(empty, "wb").close()
        with self.assertRaises(MapImageError):
            load_map_image(empty)
        good = os.path.join(self.d, "g.png")
        Image.fromarray((np.random.default_rng(0).random((100, 100, 3)) * 255).astype(np.uint8)).save(good)
        raw = open(good, "rb").read()
        trunc = os.path.join(self.d, "t.png")
        open(trunc, "wb").write(raw[: len(raw) // 2])
        with self.assertRaises(MapImageError):
            load_map_image(trunc)
        tiny = os.path.join(self.d, "s.png")
        Image.new("RGB", (10, 10)).save(tiny)
        with self.assertRaises(MapImageError):
            load_map_image(tiny)
        self.assertEqual(load_map_image(good).shape, (100, 100, 3))


class TestEndToEnd(unittest.TestCase):
    """Synthetic island: deep sea, shallow ring, beach, grass core with a road and a house."""

    def test_small_world(self):
        from map2mc.project import Project
        d = tempfile.mkdtemp()
        try:
            H = W = 160
            yy, xx = np.mgrid[:H, :W]
            r = np.hypot(xx - 80, yy - 80)
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = (35, 95, 114)                                  # deep water
            img[r < 60] = (68, 164, 159)                            # shallow
            img[r < 50] = (243, 216, 176)                           # sand
            img[r < 44] = (121, 144, 59)                            # grass
            img[78:82, 40:120] = (149, 133, 85)                     # path
            img[60:68, 90:100] = (180, 70, 70)                      # red roof
            img[56:72, 86:104][~((np.mgrid[56:72, 86:104][0] >= 60) & (np.mgrid[56:72, 86:104][0] < 68)
                                 & (np.mgrid[56:72, 86:104][1] >= 90) & (np.mgrid[56:72, 86:104][1] < 100))] = (166, 162, 161)
            src = os.path.join(d, "map.png")
            Image.fromarray(img).save(src)
            prj = Project.create(src, os.path.join(d, "prj"))
            prj.settings.blocks_per_pixel = 2.0
            prj.settings.workers = 2
            prj.run_classification()
            prj.run_derive()
            self.assertGreaterEqual(len(prj.buildings), 1)
            world = os.path.join(d, "world")
            res = prj.generate(world)
            self.assertFalse(res["cancelled"])
            lvl = nbt.plain(nbt.read_gzip_file(os.path.join(world, "level.dat")))["Data"]
            self.assertEqual(lvl["DataVersion"], 4671)
            self.assertEqual(lvl["Version"]["Name"], "1.21.11")
            self.assertIn("DragonFight", lvl)
            # island centre (pixel 80,80 is on the path; take grass at 80,100 -> world)
            geo = prj.geometry()
            X, Z = geo.px_to_world(60.5, 100.5)
            X, Z = int(X), int(Z)
            path = os.path.join(world, "region", f"r.{X >> 9}.{Z >> 9}.mca")
            ch = anvil.read_region_chunk(path, (X >> 4) & 31, (Z >> 4) & 31)
            y, name = top_block(ch, X & 15, Z & 15)
            self.assertIn(name, ("minecraft:grass_block", "minecraft:oak_leaves", "minecraft:oak_log"))
            self.assertGreater(y, prj.settings.water_level)
            # sea at a corner
            X, Z = geo.px_to_world(geo.cu0 + 1, geo.cv0 + 1)
            X, Z = int(X), int(Z)
            ch = anvil.read_region_chunk(os.path.join(world, "region", f"r.{X >> 9}.{Z >> 9}.mca"),
                                         (X >> 4) & 31, (Z >> 4) & 31)
            self.assertEqual(top_block(ch, X & 15, Z & 15), (prj.settings.water_level, "minecraft:water"))
            # resume: nothing left to do
            logs = []
            prj.generate(world, log=logs.append)
            self.assertTrue(any("0 zu erzeugen" in s for s in logs), logs)
            # read back a box, mesh it and export/import a .litematic
            from map2mc import blockmesh, litematic, worldread
            X, Z = geo.px_to_world(80.5, 80.5)
            vol, names, y0 = worldread.read_box(world, int(X) - 24, int(Z) - 24, 48, 48)
            self.assertGreater(np.count_nonzero(vol), 0)
            op, wat, atlas = blockmesh.build_mesh(vol, names, blockmesh.Atlas(None), blockmesh.surface_floor(vol, names))
            self.assertGreater(len(op), 0)
            self.assertEqual(op.shape[1], 8)
            lp = os.path.join(d, "a.litematic")
            info = litematic.export_area(world, lp, int(X) - 24, int(Z) - 24, 48, 48, below_surface=None)
            v2, pal, root = litematic.read_litematic(lp)
            self.assertEqual(nbt.plain(root["MinecraftDataVersion"]), anvil.DATA_VERSION)
            self.assertTrue((np.array(pal)[v2] == np.array(names)[vol]).all())
            self.assertEqual(info["blocks"], int(np.count_nonzero(vol)))
            # generator wrote the LOD overview, and the LOD scene builds
            from map2mc import scene3d
            self.assertTrue(os.listdir(os.path.join(world, "map2mc_lod")))
            sc = scene3d.build_scene(world, int(X), int(Z), 32, scene3d.LodCache(), None)
            self.assertGreater(sc["faces"], sc["detail_faces"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestHeights(unittest.TestCase):
    """Height estimate without a height map, on synthetic islands with known heights."""

    def test_estimate_matches_truth(self):
        from map2mc import heightbench
        r = heightbench.run(seeds=(7, 11), size=256)
        self.assertGreater(r["corr"], 0.85, r)        # shape of the terrain is recovered
        self.assertLess(r["streaks"], 0.3, r)         # no diagonal stripe artefacts
        self.assertLess(r["pits"], 0.05, r)           # tree dots do not dig holes
        self.assertLess(r["river"], 0.15, r)          # a river does not flatten the whole inland

    def test_town_stays_flat(self):
        from map2mc import heightbench, terrain
        from map2mc.config import Settings
        img, st, truth, info = heightbench.make_island(256, 7)
        h, _ = terrain.compute_heights(img, st, Settings())
        town = info["town"] & st.land
        self.assertLess(float(np.percentile(h[town], 90) - np.percentile(h[town], 10)), 4.0)


if __name__ == "__main__":
    unittest.main()
