"""Anvil region / chunk writer for Minecraft Java 1.21.x (DataVersion 4671 = 1.21.11).

Chunk format (since 1.18, no "Level" wrapper):
    DataVersion, xPos, yPos, zPos, Status="minecraft:full",
    sections: [{Y, block_states:{palette,data}, biomes:{palette,data}}],
    block_entities, isLightOn=0 (the game computes light itself on load)
Heightmaps are omitted on purpose: the game primes missing heightmaps when a
chunk is loaded, which is safer than writing possibly wrong ones.

Region file (.mca): 4 KiB location table + 4 KiB timestamp table, then
chunks padded to 4 KiB sectors, each as <len:int><compression=2 zlib><data>.
"""
from __future__ import annotations

import os
import struct
import time
import zlib

import numpy as np

from . import nbt
from .nbt import Byte, Int, List, Long

DATA_VERSION = 4671          # Minecraft Java 1.21.11 ("world_version" in version.json)
MIN_Y = -64                  # overworld bottom
MAX_Y = 319                  # overworld top (inclusive)
MIN_SECTION = MIN_Y >> 4     # -4


def parse_state(state: str):
    """'minecraft:oak_log[axis=y]' -> ('minecraft:oak_log', {'axis': 'y'})."""
    state = state.strip()
    if "[" in state:
        name, props = state[:-1].split("[", 1)
        p = {}
        for kv in props.split(","):
            if kv:
                k, v = kv.split("=", 1)
                p[k.strip()] = v.strip()
    else:
        name, p = state, {}
    if ":" not in name:
        name = "minecraft:" + name
    return name, p


class BlockRegistry:
    """Maps block-state strings to small integer ids (local to one process)."""

    def __init__(self):
        self.states: list[str] = []
        self.index: dict[str, int] = {}
        self.AIR = self.id("minecraft:air")

    def id(self, state: str) -> int:
        i = self.index.get(state)
        if i is None:
            name, props = parse_state(state)
            canon = name + ("[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]" if props else "")
            i = self.index.get(canon)
            if i is None:
                i = len(self.states)
                self.states.append(canon)
                self.index[canon] = i
            self.index[state] = i
        return i

    def nbt_entry(self, i: int) -> dict:
        name, props = parse_state(self.states[i])
        e = {"Name": name}
        if props:
            e["Properties"] = dict(props)
        return e


def pack_indices(idx: np.ndarray, bits: int) -> np.ndarray:
    """Pack palette indices into longs (1.16+ layout: entries never span longs)."""
    vpl = 64 // bits
    n = idx.size
    longs = -(-n // vpl)
    padded = np.zeros(longs * vpl, dtype=np.uint64)
    padded[:n] = idx.astype(np.uint64)
    padded = padded.reshape(longs, vpl)
    shifts = (np.arange(vpl, dtype=np.uint64) * np.uint64(bits))
    packed = np.bitwise_or.reduce(padded << shifts, axis=1)
    return packed.view(np.int64)


def unpack_indices(data: np.ndarray, bits: int, n: int) -> np.ndarray:
    vpl = 64 // bits
    u = data.view(np.uint64)
    shifts = (np.arange(vpl, dtype=np.uint64) * np.uint64(bits))
    mask = np.uint64((1 << bits) - 1)
    vals = (u[:, None] >> shifts[None, :]) & mask
    return vals.reshape(-1)[:n].astype(np.int64)


def encode_section_blocks(sec: np.ndarray, reg: BlockRegistry) -> dict:
    """sec: (16,16,16) array [y,z,x] of registry ids."""
    flat = sec.reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    palette = List(nbt.TAG_COMPOUND, [reg.nbt_entry(int(u)) for u in uniq])
    if len(uniq) == 1:
        return {"palette": palette}
    bits = max(4, int(np.ceil(np.log2(len(uniq)))))
    return {"palette": palette, "data": pack_indices(inv, bits)}


def encode_section_biomes(bio: np.ndarray, biome_names: list[str]) -> dict:
    """bio: (4,4,4) array [y,z,x] of indices into biome_names."""
    flat = bio.reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    palette = List(nbt.TAG_STRING, [biome_names[int(u)] for u in uniq])
    if len(uniq) == 1:
        return {"palette": palette}
    bits = max(1, int(np.ceil(np.log2(len(uniq)))))
    return {"palette": palette, "data": pack_indices(inv, bits)}


def encode_chunk(cx: int, cz: int, blocks: np.ndarray, y0: int, reg: BlockRegistry,
                 biomes2d: np.ndarray, biome_names: list[str],
                 block_entities: list | None = None) -> bytes:
    """Build the zlib-compressed NBT of one chunk.

    blocks:   (H,16,16) [y,z,x] ids, starting at world y = y0 (multiple of 16, >= MIN_Y)
    biomes2d: (4,4) [z,x] biome indices (applied to every section)
    """
    H = blocks.shape[0]
    assert y0 % 16 == 0 and H % 16 == 0
    sections = []
    bio3 = np.broadcast_to(biomes2d[None, :, :], (4, 4, 4))
    bio_nbt = encode_section_biomes(bio3, biome_names)
    first = y0 >> 4
    for s in range(H // 16):
        sec = blocks[s * 16:(s + 1) * 16]
        sections.append({
            "Y": Byte(first + s),
            "block_states": encode_section_blocks(sec, reg),
            "biomes": bio_nbt,
        })
    root = {
        "DataVersion": Int(DATA_VERSION),
        "xPos": Int(cx),
        "yPos": Int(MIN_SECTION),
        "zPos": Int(cz),
        "Status": "minecraft:full",
        "LastUpdate": Long(0),
        "InhabitedTime": Long(0),
        "isLightOn": Byte(0),
        "sections": List(nbt.TAG_COMPOUND, sections),
        "block_entities": List(nbt.TAG_COMPOUND, block_entities or []),
        "block_ticks": List(nbt.TAG_COMPOUND, []),
        "fluid_ticks": List(nbt.TAG_COMPOUND, []),
        "PostProcessing": List(nbt.TAG_LIST, [List(nbt.TAG_SHORT, []) for _ in range(24)]),
        "structures": {"References": {}, "starts": {}},
    }
    return zlib.compress(nbt.encode(root), 6)


def write_region(path: str, chunks: dict[tuple[int, int], bytes]) -> None:
    """Write a region file atomically. chunks maps local (lx,lz) 0..31 -> compressed NBT."""
    locations = bytearray(4096)
    timestamps = bytearray(4096)
    body = bytearray()
    sector = 2
    now = int(time.time())
    for (lx, lz), data in sorted(chunks.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        payload = struct.pack(">ib", len(data) + 1, 2) + data
        pad = (-len(payload)) % 4096
        payload += b"\x00" * pad
        nsec = len(payload) // 4096
        if nsec > 255:
            raise ValueError(f"chunk {lx},{lz} too large for region format ({nsec} sectors)")
        i = 4 * (lx + lz * 32)
        locations[i:i + 4] = struct.pack(">I", (sector << 8) | nsec)
        timestamps[i:i + 4] = struct.pack(">I", now)
        body += payload
        sector += nsec
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(locations)
        f.write(timestamps)
        f.write(body)
    os.replace(tmp, path)


def read_region_chunk(path: str, lx: int, lz: int) -> dict | None:
    """Read one chunk back (used by tests and the validator)."""
    with open(path, "rb") as f:
        head = f.read(4096)
        loc = struct.unpack(">I", head[4 * (lx + lz * 32):4 * (lx + lz * 32) + 4])[0]
        if loc == 0:
            return None
        f.seek((loc >> 8) * 4096)
        length, comp = struct.unpack(">ib", f.read(5))
        data = f.read(length - 1)
    if comp != 2:
        raise ValueError("unsupported compression")
    return nbt.decode(zlib.decompress(data))
