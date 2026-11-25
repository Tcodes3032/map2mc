"""Settings, terrain classes and the default block palette.

Every class has
  * colour prototypes (RGB) used by the image classifier; you can add more in
    the GUI with the eyedropper,
  * a block assignment (surface, subsurface, biome, trees, vegetation) that is
    fully editable in the GUI ("Blockzuordnung") and stored as palette.json.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields

# ----------------------------------------------------------------- classes
# id: (key, German label, preview colour)
CLASSES = [
    ("void", "Hintergrund (außerhalb der Karte)", (40, 40, 48)),
    ("water_deep", "Tiefes Wasser", (25, 70, 140)),
    ("water_shallow", "Flaches Wasser", (60, 170, 200)),
    ("sand", "Sand / Strand", (235, 215, 160)),
    ("grass", "Gras / Wiese", (110, 160, 60)),
    ("forest", "Wald", (40, 100, 35)),
    ("desert", "Wüste / roter Sand", (215, 130, 80)),
    ("rock", "Fels / Berg", (125, 120, 110)),
    ("dark_rock", "Dunkler Fels / Vulkan", (45, 40, 40)),
    ("path", "Weg (Erde)", (170, 140, 90)),
    ("road", "Straße (Asphalt)", (90, 90, 95)),
    ("urban", "Bebauung / Gebäude", (200, 200, 205)),
    ("snow", "Schnee", (245, 248, 255)),
    ("blossom", "Blütenbäume (rosa)", (240, 150, 200)),
    ("magic", "Lila Gebiet", (150, 100, 170)),
    ("unknown", "Unbekannt – prüfen!", (255, 0, 255)),
]
CLASS_KEYS = [c[0] for c in CLASSES]
CID = {k: i for i, k in enumerate(CLASS_KEYS)}
N_CLASSES = len(CLASSES)
WATER_CLASSES = (CID["void"], CID["water_deep"], CID["water_shallow"])

# Default colour prototypes, measured on a typical top-down island map.
DEFAULT_PROTOTYPES: dict[str, list[list[int]]] = {
    "void": [[47, 48, 55], [43, 45, 52]],
    "water_deep": [[35, 95, 114], [38, 77, 93], [43, 61, 72], [34, 116, 135]],
    "water_shallow": [[34, 139, 152], [68, 164, 159], [112, 188, 175], [160, 213, 192], [99, 135, 124]],
    "sand": [[243, 216, 176], [235, 205, 160], [204, 214, 191]],
    "grass": [[121, 144, 59], [103, 135, 55], [114, 142, 85], [135, 155, 70]],
    "forest": [[86, 110, 49], [70, 95, 42], [58, 82, 40]],
    "desert": [[212, 136, 95], [241, 167, 117], [208, 124, 91], [225, 150, 105]],
    "rock": [[104, 97, 79], [120, 112, 95], [140, 128, 110]],
    "dark_rock": [[51, 46, 46], [65, 60, 58], [80, 74, 70]],
    "path": [[149, 133, 85], [176, 155, 107], [193, 169, 145]],
    "road": [],                     # assigned from "path" near built-up areas
    "urban": [[152, 141, 131], [166, 162, 161], [200, 200, 200], [235, 235, 235], [180, 70, 70], [90, 110, 150]],
    "snow": [[245, 248, 255]],
    "blossom": [[230, 150, 200], [205, 120, 170], [215, 140, 185]],
    "magic": [[121, 103, 122], [131, 104, 145], [150, 110, 190], [110, 90, 160]],
    "unknown": [],
}

# Block assignment per class.  Blocks can be a weighted mix: "a*5,b*2,c".
DEFAULT_PALETTE: dict[str, dict] = {
    "void":          {"surface": "minecraft:gravel", "subsurface": "minecraft:gravel", "sub_depth": 3, "biome": "minecraft:deep_ocean", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "water_deep":    {"surface": "minecraft:gravel*3,minecraft:sand*2,minecraft:clay", "subsurface": "minecraft:sand", "sub_depth": 3, "biome": "minecraft:warm_ocean", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "water_shallow": {"surface": "minecraft:sand", "subsurface": "minecraft:sand", "sub_depth": 3, "biome": "minecraft:warm_ocean", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "sand":          {"surface": "minecraft:sand", "subsurface": "minecraft:sandstone", "sub_depth": 4, "biome": "minecraft:beach", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "grass":         {"surface": "minecraft:grass_block", "subsurface": "minecraft:dirt", "sub_depth": 3, "biome": "minecraft:plains", "tree": "auto", "tree_density": 0.0, "vegetation": 0.18},
    "forest":        {"surface": "minecraft:grass_block*6,minecraft:podzol,minecraft:coarse_dirt", "subsurface": "minecraft:dirt", "sub_depth": 3, "biome": "minecraft:forest", "tree": "auto", "tree_density": 1.0, "vegetation": 0.25},
    "desert":        {"surface": "minecraft:red_sand*5,minecraft:terracotta*2,minecraft:orange_terracotta", "subsurface": "minecraft:red_sandstone", "sub_depth": 4, "biome": "minecraft:badlands", "tree": "none", "tree_density": 0.0, "vegetation": 0.01},
    "rock":          {"surface": "minecraft:stone*4,minecraft:andesite*3,minecraft:cobblestone,minecraft:gravel", "subsurface": "minecraft:stone", "sub_depth": 3, "biome": "minecraft:windswept_hills", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "dark_rock":     {"surface": "minecraft:blackstone*4,minecraft:basalt[axis=y]*2,minecraft:tuff", "subsurface": "minecraft:blackstone", "sub_depth": 4, "biome": "minecraft:stony_peaks", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "path":          {"surface": "minecraft:dirt_path*6,minecraft:coarse_dirt*2,minecraft:gravel", "subsurface": "minecraft:dirt", "sub_depth": 3, "biome": "minecraft:plains", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "road":          {"surface": "minecraft:gray_concrete*6,minecraft:andesite*2,minecraft:black_concrete", "subsurface": "minecraft:stone", "sub_depth": 3, "biome": "minecraft:plains", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "urban":         {"surface": "minecraft:smooth_stone*3,minecraft:polished_andesite*2,minecraft:stone_bricks", "subsurface": "minecraft:stone", "sub_depth": 3, "biome": "minecraft:plains", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
    "snow":          {"surface": "minecraft:snow_block", "subsurface": "minecraft:packed_ice*1,minecraft:dirt*3", "sub_depth": 3, "biome": "minecraft:snowy_plains", "tree": "spruce", "tree_density": 0.0, "vegetation": 0.0},
    "blossom":       {"surface": "minecraft:grass_block", "subsurface": "minecraft:dirt", "sub_depth": 3, "biome": "minecraft:cherry_grove", "tree": "cherry", "tree_density": 0.6, "vegetation": 0.2},
    "magic":         {"surface": "minecraft:grass_block*3,minecraft:mycelium", "subsurface": "minecraft:dirt", "sub_depth": 3, "biome": "minecraft:cherry_grove", "tree": "auto", "tree_density": 0.3, "vegetation": 0.1},
    "unknown":       {"surface": "minecraft:grass_block", "subsurface": "minecraft:dirt", "sub_depth": 3, "biome": "minecraft:plains", "tree": "none", "tree_density": 0.0, "vegetation": 0.0},
}

# "auto" = leaf type chosen from the crown colour in the map image
TREE_TYPES = ["auto", "none", "oak", "birch", "spruce", "dark_oak", "cherry", "acacia", "jungle", "pale_oak",
              "azalea", "mangrove", "palm", "bush"]


@dataclass
class Settings:
    # --- scale and geometry
    blocks_per_pixel: float = 3.0       # world blocks per source pixel
    crop_to_island: bool = True         # cut away background/ocean far from land
    ocean_margin: int = 96              # blocks of sea kept around the island when cropping
    # --- heights (all in blocks)
    water_level: int = 62               # y of the highest water block
    max_land_height: int = 48           # max height of normal land above water
    mountain_height: int = 40           # extra height for rock / dark rock areas
    beach_height: float = 2.0           # beach rises to this many blocks above water
    max_ocean_depth: int = 24
    relief_strength: float = 0.6        # use map shading as relief (0 = off)
    detail_noise: float = 1.2           # small natural surface variation (0 = off)
    smoothing: float = 1.5              # gaussian smoothing of heights (source pixels)
    # --- classification
    max_color_distance: float = 30.0    # Lab distance above which a pixel is "unknown"
    min_region_px: int = 6              # specks smaller than this are merged into neighbours
    # --- features
    tree_density: float = 1.0           # global multiplier
    tree_spacing: int = 7               # blocks between filler trees in forests
    detect_single_trees: bool = True
    filler_trees: bool = False          # extra random trees in forest areas (off: only trees seen on the map)
    tree_pack: str = ""                 # imported tree library (.m2trees); empty = built-in tree shapes
    use_tree_pack: bool = True
    storey_height: int = 4
    default_storeys: int = 1
    min_building_px: int = 6            # smaller built-up specks become paved ground
    generate_buildings: bool = True     # False = no buildings at all (towns become paved squares)
    detect_single_houses: bool = True   # isolated houses outside towns
    object_contrast: float = 24.0       # sensitivity of the single-house detector (lower = more)
    object_min_px: int = 8
    building_lights: bool = True
    wall_block: str = "minecraft:white_terracotta"
    floor_block: str = "minecraft:spruce_planks"
    window_block: str = "minecraft:glass"
    bridge_deck_block: str = "minecraft:spruce_planks"
    bridge_fence_block: str = "minecraft:spruce_fence"
    bridge_pillar_block: str = "minecraft:spruce_log[axis=y]"
    # --- world
    world_name: str = "Map2MC Island"
    seed: int = 20261123
    game_mode: int = 1                  # 0 survival, 1 creative
    workers: int = 0                    # 0 = CPU count - 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "Settings":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def default_palette() -> dict:
    return copy.deepcopy(DEFAULT_PALETTE)


def default_prototypes() -> dict:
    return copy.deepcopy(DEFAULT_PROTOTYPES)


def parse_mix(spec: str) -> list[tuple[str, int]]:
    """'minecraft:stone*4,minecraft:andesite' -> [('minecraft:stone',4),('minecraft:andesite',1)].
    Block states may contain commas inside [...], so we split carefully."""
    out, depth, cur = [], 0, ""
    for ch in spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    res = []
    for item in out:
        item = item.strip()
        w = 1
        if "*" in item.rsplit("]", 1)[-1]:
            item, ws = item.rsplit("*", 1)
            w = max(1, int(ws))
        res.append((item.strip(), w))
    return res
