""".litematic export (Litematica schematic format, version 7 / sub-version 1).

Layout of the gzip-compressed NBT file:
  Version 7, SubVersion 1, MinecraftDataVersion 4671 (1.21.11)
  Metadata { Name, Author, Description, RegionCount, TotalVolume, TotalBlocks,
             TimeCreated, TimeModified, EnclosingSize {x,y,z} }
  Regions { <name>: { Position {x,y,z}, Size {x,y,z}, BlockStatePalette [..],
                      BlockStates long[], TileEntities [], Entities [],
                      PendingBlockTicks [], PendingFluidTicks [] } }

BlockStates is Litematica's bit array: every entry uses max(2, ceil(log2(n)))
bits, entries are packed LSB-first and MAY span two longs (unlike chunk data).
Index = (y * sizeZ + z) * sizeX + x, palette entry 0 is air.
"""
from __future__ import annotations

import gzip
import os
import time

import numpy as np

from . import nbt, worldread
from .anvil import DATA_VERSION
from .nbt import Int, List, Long

SCHEMATIC_VERSION = 7
SCHEMATIC_SUBVERSION = 1
TAG_COMPOUND = 10


def bits_for(n: int) -> int:
    return max(2, int(np.ceil(np.log2(max(n, 2)))))


def pack_spanning(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack values (any int dtype) into int64 words, entries may cross word borders."""
    vals = values.astype(np.uint64).reshape(-1)
    n_words = (len(vals) * bits + 63) // 64
    out = np.zeros(n_words, np.uint64)
    # process in chunks to keep memory bounded
    step = 1 << 20
    for s in range(0, len(vals), step):
        v = vals[s:s + step]
        stream = ((v[:, None] >> np.arange(bits, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8).reshape(-1)
        bit0 = s * bits
        # align the chunk to a byte border of the global stream
        lead = bit0 % 8
        if lead:
            stream = np.concatenate([np.zeros(lead, np.uint8), stream])
        packed = np.packbits(stream, bitorder="little")
        byte0 = bit0 // 8
        buf = out.view(np.uint8)
        end = min(len(buf), byte0 + len(packed))
        buf[byte0:end] |= packed[:end - byte0]
    return out.view(np.int64)


def unpack_spanning(words: np.ndarray, bits: int, n: int) -> np.ndarray:
    stream = np.unpackbits(words.astype(np.int64).view(np.uint8), bitorder="little")[:n * bits]
    w = (np.uint64(1) << np.arange(bits, dtype=np.uint64))
    return (stream.reshape(n, bits).astype(np.uint64) * w).sum(1).astype(np.int64)


def _palette_entry(state: str) -> dict:
    name = state.split("[", 1)[0]
    e = {"Name": name}
    if "[" in state:
        e["Properties"] = dict(p.split("=", 1) for p in state[state.index("[") + 1:-1].split(",") if "=" in p)
    return e


def write_litematic(path: str, vol: np.ndarray, names: list[str], name: str = "Map2MC", author: str = "Map2MC",
                    description: str = "") -> dict:
    """vol [y,z,x] indices into names (names[0] must be air)."""
    assert names[0].startswith("minecraft:air")
    used = np.unique(vol)
    if used[0] != 0:
        used = np.concatenate([[0], used])
    remap = np.zeros(len(names), np.int64)
    remap[used] = np.arange(len(used))
    pal_names = [names[i] for i in used]
    sy, sz, sx = vol.shape
    bits = bits_for(len(pal_names))
    states = pack_spanning(remap[vol.reshape(-1)], bits)
    total_blocks = int(np.count_nonzero(vol))
    now = int(time.time() * 1000)
    size = {"x": Int(sx), "y": Int(sy), "z": Int(sz)}
    region = {
        "Position": {"x": Int(0), "y": Int(0), "z": Int(0)},
        "Size": dict(size),
        "BlockStatePalette": List(TAG_COMPOUND, [_palette_entry(s) for s in pal_names]),
        "BlockStates": states,
        "TileEntities": List(TAG_COMPOUND),
        "Entities": List(TAG_COMPOUND),
        "PendingBlockTicks": List(TAG_COMPOUND),
        "PendingFluidTicks": List(TAG_COMPOUND),
    }
    root = {
        "Version": Int(SCHEMATIC_VERSION),
        "SubVersion": Int(SCHEMATIC_SUBVERSION),
        "MinecraftDataVersion": Int(DATA_VERSION),
        "Metadata": {
            "Name": name, "Author": author, "Description": description,
            "RegionCount": Int(1), "TotalVolume": Int(sx * sy * sz), "TotalBlocks": Int(total_blocks),
            "TimeCreated": Long(now), "TimeModified": Long(now),
            "EnclosingSize": dict(size),
        },
        "Regions": {name: region},
    }
    tmp = path + ".tmp"
    nbt.write_gzip_file(tmp, root)
    os.replace(tmp, path)
    return {"path": path, "size": (sx, sy, sz), "blocks": total_blocks, "palette": len(pal_names)}


def read_litematic(path: str):
    """Read back the first region -> (vol [y,z,x], names).  Used for verification."""
    root = nbt.read_gzip_file(path)
    reg = next(iter(root["Regions"].values()))
    s = nbt.plain(reg["Size"])
    sx, sy, sz = abs(s["x"]), abs(s["y"]), abs(s["z"])
    pal = [worldread.state_string(p) for p in reg["BlockStatePalette"]]
    idx = unpack_spanning(np.asarray(reg["BlockStates"]), bits_for(len(pal)), sx * sy * sz)
    return idx.reshape(sy, sz, sx), pal, root


def export_area(world: str, path: str, x0: int, z0: int, sx: int, sz: int, below_surface: int | None = 8,
                name: str | None = None, progress=None) -> dict:
    """Export a box of the generated world.

    below_surface: keep this many layers under the lowest surface column (None = down to bedrock)."""
    vol, names, y0 = worldread.read_box(world, x0, z0, sx, sz, progress=progress)
    if below_surface is not None:
        from .blockmesh import surface_floor
        fl = surface_floor(vol, names, below_surface)
        vol, y0 = vol[fl:], y0 + fl
    nm = name or os.path.splitext(os.path.basename(path))[0]
    info = write_litematic(path, np.ascontiguousarray(vol), names, nm,
                           description=f"Map2MC export, world origin x={x0} y={y0} z={z0}")
    info["origin"] = (x0, y0, z0)
    return info


def export_tiles(world: str, folder: str, x0: int, z0: int, x1: int, z1: int, tile: int = 256,
                 below_surface: int | None = 8, progress=None, cancel=None) -> list[dict]:
    """Split a large area into tile x tile schematics (file names carry the world origin)."""
    os.makedirs(folder, exist_ok=True)
    jobs = [(x, z) for z in range(z0, z1, tile) for x in range(x0, x1, tile)]
    out = []
    for k, (x, z) in enumerate(jobs):
        if cancel is not None and cancel.is_set():
            break
        sx, sz = min(tile, x1 - x), min(tile, z1 - z)
        p = os.path.join(folder, f"map2mc_x{x}_z{z}.litematic")
        info = export_area(world, p, x, z, sx, sz, below_surface)
        if info["blocks"] == 0:
            os.remove(p)
        else:
            out.append(info)
        if progress:
            progress((k + 1) / len(jobs))
    return out
