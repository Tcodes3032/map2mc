"""level.dat writer for Minecraft Java 1.21.11.

Verified against a level.dat saved by the vanilla 1.21.11 server:
  * the world spawn is a `spawn` compound {dimension, pos:[I;x,y,z], yaw, pitch}
    (SpawnX/Y/Z are no longer used),
  * the world border lives in data/world_border.dat (default = no border),
  * game rules live in `game_rules`; we leave them at their defaults.

The overworld uses a *flat* generator configured as endless ocean, so any
chunk outside the generated map area (beyond the world border padding) is
plain sea instead of random vanilla terrain.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import nbt
from .anvil import DATA_VERSION
from .nbt import Byte, Double, Float, Int, List, Long

VERSION_NAME = "1.21.11"


def flat_ocean_generator(water_level: int, biome: str = "minecraft:ocean") -> dict:
    """Flat layers: bedrock, stone, sand, water so that the water surface is at water_level."""
    # Layers are stacked upward from y = -64:
    #   bedrock y=-64, stone y=-63..S, sand S+1..S+2, water S+3..water_level (20 blocks)
    S = water_level - 22
    layers = [
        {"block": "minecraft:bedrock", "height": Int(1)},
        {"block": "minecraft:stone", "height": Int(S + 64)},
        {"block": "minecraft:sand", "height": Int(2)},
        {"block": "minecraft:water", "height": Int(water_level - S - 2)},
    ]
    return {
        "type": "minecraft:flat",
        "settings": {
            "biome": biome,
            "features": Byte(0),
            "lakes": Byte(0),
            "layers": List(nbt.TAG_COMPOUND, layers),
            "structure_overrides": List(nbt.TAG_STRING, []),
        },
    }


def write_level_dat(world_dir: str, name: str, spawn: tuple[int, int, int], seed: int,
                    water_level: int, game_type: int = 1) -> None:
    sx, sy, sz = spawn
    now_ms = int(time.time() * 1000)
    data = {
        "DataVersion": Int(DATA_VERSION),
        "version": Int(19133),            # anvil level format version, unchanged since 1.9
        "Version": {"Id": Int(DATA_VERSION), "Name": VERSION_NAME, "Series": "main", "Snapshot": Byte(0)},
        "LevelName": name,
        "GameType": Int(game_type),       # 1 = creative (useful for exploring/editing)
        "Difficulty": Byte(1),
        "DifficultyLocked": Byte(0),
        "hardcore": Byte(0),
        "allowCommands": Byte(1),
        "initialized": Byte(1),
        "LastPlayed": Long(now_ms),
        "Time": Long(0),
        "DayTime": Long(6000),
        "spawn": {
            "dimension": "minecraft:overworld",
            "pos": np.array([sx, sy, sz], dtype=np.int32),
            "yaw": Float(0.0),
            "pitch": Float(0.0),
        },
        "clearWeatherTime": Int(0), "raining": Byte(0), "rainTime": Int(0),
        "thundering": Byte(0), "thunderTime": Int(0),
        "WasModded": Byte(0),
        "ServerBrands": List(nbt.TAG_STRING, ["map2mc"]),
        "DataPacks": {"Enabled": List(nbt.TAG_STRING, ["vanilla"]), "Disabled": List(nbt.TAG_STRING, [])},
        # Same shape the 1.21.11 server writes itself (it logs "key missing: DragonFight" otherwise).
        "DragonFight": {"NeedsStateScanning": Byte(1), "DragonKilled": Byte(0), "PreviouslyKilled": Byte(0),
                        "Gateways": List(nbt.TAG_INT, [Int(i) for i in range(20)])},
        "CustomBossEvents": {},
        "ScheduledEvents": List(nbt.TAG_COMPOUND, []),
        "WanderingTraderSpawnChance": Int(25),
        "WanderingTraderSpawnDelay": Int(24000),
        "WorldGenSettings": {
            "seed": Long(seed),
            "generate_features": Byte(0),
            "bonus_chest": Byte(0),
            "dimensions": {
                "minecraft:overworld": {
                    "type": "minecraft:overworld",
                    "generator": flat_ocean_generator(water_level),
                },
                "minecraft:the_nether": {
                    "type": "minecraft:the_nether",
                    "generator": {"type": "minecraft:noise", "settings": "minecraft:nether",
                                  "biome_source": {"type": "minecraft:multi_noise", "preset": "minecraft:nether"}},
                },
                "minecraft:the_end": {
                    "type": "minecraft:the_end",
                    "generator": {"type": "minecraft:noise", "settings": "minecraft:end",
                                  "biome_source": {"type": "minecraft:the_end"}},
                },
            },
        },
    }
    os.makedirs(world_dir, exist_ok=True)
    tmp = os.path.join(world_dir, "level.dat.tmp")
    nbt.write_gzip_file(tmp, {"Data": data})
    os.replace(tmp, os.path.join(world_dir, "level.dat"))
