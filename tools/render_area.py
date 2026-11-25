"""Render an area of a generated world from its region files (oblique view, 1 block = N px).

    python tools/render_area.py <world_dir> <x0> <z0> <size> <out.png> [--px 4]

Reads the real .mca files, so it shows exactly what Minecraft will load.
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from map2mc import anvil  # noqa: E402

COL = {
    "grass_block": (100, 160, 60), "short_grass": (90, 150, 55), "dirt": (134, 96, 67), "dirt_path": (148, 122, 65),
    "coarse_dirt": (119, 85, 59), "podzol": (91, 63, 24), "sand": (219, 207, 163), "sandstone": (216, 203, 155),
    "red_sand": (190, 102, 33), "terracotta": (152, 94, 68), "orange_terracotta": (161, 83, 37),
    "stone": (125, 125, 125), "andesite": (136, 136, 136), "cobblestone": (120, 120, 120), "gravel": (131, 127, 126),
    "blackstone": (42, 36, 41), "basalt": (80, 80, 85), "tuff": (108, 109, 102), "water": (50, 110, 200),
    "oak_log": (109, 85, 50), "oak_leaves": (60, 130, 40), "birch_leaves": (110, 150, 70), "spruce_leaves": (50, 90, 50),
    "dark_oak_leaves": (45, 95, 30), "cherry_leaves": (240, 170, 210), "cherry_log": (60, 35, 45),
    "dark_oak_log": (60, 45, 25), "gray_concrete": (55, 58, 62), "black_concrete": (8, 10, 15),
    "smooth_stone": (159, 159, 159), "polished_andesite": (132, 134, 133), "stone_bricks": (122, 122, 122),
    "white_terracotta": (210, 178, 161), "spruce_planks": (115, 85, 49), "spruce_fence": (100, 75, 45),
    "glass": (200, 230, 240), "mycelium": (111, 99, 105), "clay": (160, 166, 179), "pink_petals": (245, 180, 210),
    "red_sandstone": (186, 99, 29), "lantern": (250, 200, 90), "ladder": (120, 95, 55),
}


def color(name):
    n = name.split(":")[-1]
    if n in COL:
        return COL[n]
    if "concrete" in n or "terracotta" in n or "planks" in n or "_block" in n:
        h = zlib.crc32(n.encode())
        return (80 + h % 150, 80 + (h >> 8) % 150, 80 + (h >> 16) % 150)
    if "flower" in n or n in ("poppy", "dandelion", "azure_bluet", "oxeye_daisy", "cornflower"):
        return (220, 200, 60)
    return (150, 150, 150)


def load_columns(world, x0, z0, size):
    """Return (top_y, rgb, heights for every block) -> we keep a 3D array of colours for the area."""
    ymin, ymax = -64, 200
    vol = np.zeros((ymax - ymin, size, size), np.int32)        # palette index into names
    names = ["minecraft:air"]
    nidx = {"minecraft:air": 0}
    for cz in range(z0 >> 4, (z0 + size - 1 >> 4) + 1):
        for cx in range(x0 >> 4, (x0 + size - 1 >> 4) + 1):
            path = os.path.join(world, "region", f"r.{cx >> 5}.{cz >> 5}.mca")
            if not os.path.exists(path):
                continue
            ch = anvil.read_region_chunk(path, cx & 31, cz & 31)
            if ch is None:
                continue
            for s in ch["sections"]:
                bs = s["block_states"]
                pal = [p["Name"] for p in bs["palette"]]
                loc = np.array([nidx.setdefault(n, len(nidx)) for n in pal])
                for n in pal:
                    if n not in names:
                        names.append(n)
                if "data" in bs:
                    bits = max(4, int(np.ceil(np.log2(len(pal)))))
                    idx = anvil.unpack_indices(bs["data"], bits, 4096).reshape(16, 16, 16)
                else:
                    idx = np.zeros((16, 16, 16), int)
                sy = s["Y"].value * 16
                for ly in range(16):
                    y = sy + ly - ymin
                    if not (0 <= y < vol.shape[0]):
                        continue
                    for lz in range(16):
                        z = cz * 16 + lz - z0
                        if not (0 <= z < size):
                            continue
                        xs = cx * 16 + np.arange(16) - x0
                        ok = (xs >= 0) & (xs < size)
                        vol[y, z, xs[ok]] = loc[idx[ly, lz][ok]]
    inv = {v: k for k, v in nidx.items()}
    return vol, inv, ymin


def render(vol, inv, px=4):
    """Isometric-ish oblique render (painter's algorithm, back to front)."""
    Y, D, W = vol.shape
    lut = np.array([color(inv[i]) for i in range(len(inv))], np.float32)
    air = np.array([inv[i].endswith(":air") for i in range(len(inv))])
    occ = ~air[vol]
    ys = np.nonzero(occ.any((1, 2)))[0]
    y0, y1 = ys.min(), ys.max() + 1
    rise = px                                   # screen px per block height
    H_img = D * px // 2 + (y1 - y0) * rise + px * 2
    img = np.full((H_img, W * px, 3), 25, np.uint8)
    for y in range(y0, y1):
        for z in range(D):
            row = occ[y, z]
            if not row.any():
                continue
            xs = np.nonzero(row)[0]
            sy = int(z * px / 2 + (y1 - y) * rise)
            c = lut[vol[y, z, xs]]
            # top face visible if block above is air
            top_open = (y + 1 >= Y) | ~occ[min(y + 1, Y - 1), z, xs]
            for k, x in enumerate(xs):
                col = c[k] * (1.0 if top_open[k] else 0.8)
                img[sy:sy + px // 2, x * px:(x + 1) * px] = col
                img[sy + px // 2:sy + px // 2 + rise, x * px:(x + 1) * px] = col * 0.65
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("x0", type=int)
    ap.add_argument("z0", type=int)
    ap.add_argument("size", type=int)
    ap.add_argument("out")
    ap.add_argument("--px", type=int, default=4)
    a = ap.parse_args()
    vol, inv, _ = load_columns(a.world, a.x0, a.z0, a.size)
    img = render(vol, inv, a.px)
    cv2.imwrite(a.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("written", a.out, img.shape)


if __name__ == "__main__":
    main()
