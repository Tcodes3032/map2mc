# Map2MC – map image → Minecraft Java world (1.21.11)

Turns a top-down map image (for example a battle-royale island) into a **real, walkable
Minecraft Java world**: region files (`.mca`) plus `level.dat`, ready to load in the game.

## Quick start

```powershell
cd C:\Users\tobias\Downloads\map2mc
.\start_gui.bat
```

Or from the command line:

```powershell
.venv\Scripts\python -m map2mc new samples\island.webp projects\island
.venv\Scripts\python -m map2mc analyse projects\island
.venv\Scripts\python -m map2mc generate projects\island worlds\Island --zip worlds\Island.zip
```

Copy the finished world into `%APPDATA%\.minecraft\saves\` (GUI: “Copy to Minecraft saves”).
Fresh setup: `uv venv --python 3.12 .venv` and `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`
(PySide6 needs 64-bit Python ≤ 3.13).

`samples/` is not in the repository (third-party map images); put your own map image there.

## Workflow in the GUI

1. **Upload map image** – the image is checked (empty, corrupt, truncated, too small, single colour).
2. **Size & height** – scale (blocks per pixel), sea level, land/mountain height, ocean depth.
   The world size in blocks/km is shown live. 1 block = 1 m.
3. **Analyse** – classification, terrain, water, trees, buildings.
4. **Check & correct** (views: original, classes, height, shaded preview, areas to check,
   3D preview):
   - *Paint class* / *Class: restore automatic* – brush on the class map
   - *Height + / Height − / Flatten* – height brushes (take effect immediately)
   - *Pick colour for class* – the eyedropper adds a colour prototype (analyse again afterwards)
   - *Edit building* (click) / *Add building* (drag rectangle) / *Remove building* (click,
     click again = undo) / building list
   - Checkboxes *Generate buildings* (off = no buildings, towns become paved areas) and
     *Detect single houses outside towns* (sensitivity and minimum size adjustable)
   - *Import heightmap* – real height data (greyscale image) replaces the estimate
   - *Import tree pack* – .7z/.zip with WorldEdit `.schem` trees (e.g. “5K Trees”). Natural
     trees are sorted by leaf type and crown size, **all leaves are set to `persistent=true`**
     (otherwise trees without a trunk decay), old block names are updated, glass/fantasy/crystal
     objects and oversized trees are filtered out. During generation each tree gets a pack tree of
     similar size (deterministic, rotated); missing species such as cherry get oak shapes with matching
     leaves. Crowns that would grow into houses are left out.
   - *Extra filler trees in forests* (default: off) – off means only the trees visible on the map;
     on also fills forest areas with random trees (old behaviour).
   - *Block mapping* – surface, subsurface, biome, tree species, density per class; mixes
     such as `minecraft:stone*4,minecraft:andesite`
5. **Start generation** – runs in parallel per region (512×512 blocks), with progress and cancel.
   Finished regions are saved immediately; starting again resumes.
6. **3D blocks / hologram** – shows the *generated* world like the Voxy mod: real blocks with the
   Minecraft textures around the focus (from the installed 1.21.11 client, otherwise flat colours),
   2×2 columns up to 384 blocks further out, and the whole map as coarse columns behind that
   (4–32 blocks depending on map size).
   Set the focus with the *Pick 3D area* tool on the map or enter centre X/Z;
   with *Detail follows camera* the detail area moves with you.
   **Player (WASD)** controls (default, like Minecraft): clicking into the view captures the mouse,
   mouse = look around, W/A/S/D move, Space/Shift up/down, Ctrl faster, mouse wheel flight speed,
   **F** toggles flying ↔ walking (walking: gravity, jump with Space, walls of 1 block and higher
   stop you, in water you stand chest-deep), **R** back to start, **Esc** releases the mouse.
   **Orbit (mouse)** controls: left mouse rotates, right mouse pans, wheel zooms,
   double-click resets.
   The generator writes the overview data as it goes (`map2mc_lod/`); for older worlds it is
   computed once (≈ 30 s).
   *Hologram* shows the blocks translucent like Litematica, *Layers up to* hides everything above
   a given height.
7. **Export** – copy into the Minecraft saves, as ZIP, or as **.litematic**
   (an area or the whole world in 256×256 tiles; the file name contains the world coordinate of the corner).
   Put the files into `.minecraft/schematics` and load them in Litematica.

**Language:** English is the default; switch to German in the *Language / Sprache* menu
(the window is rebuilt, the project stays open). Translations live in `map2mc/i18n.py`.

All inputs and intermediate results live in the project folder (`settings.json`, `palette.json`,
`prototypes.json`, `edits/`, `cache/`, `data/`) and are saved continuously.

## What is detected automatically – and what is not

| Element | Method | Limits |
|---|---|---|
| Water, coast, beach, meadow, forest, desert, rock, volcano, paths, roads, built-up areas, blossom trees, purple areas | colour prototypes in CIE-Lab, cleanup, unknown colours → area to check | new map styles may need extra colour prototypes (eyedropper) |
| Lakes/rivers vs. sea | connection to the map border; inland water gets its own water level | – |
| Bridges | path pixels mostly surrounded by water → wooden bridge with railings and pillars | – |
| Single trees | dark/pink crown blobs (multi-scale DoG), crown size from blob size; tree species “auto” picks the leaves from the crown hue (pink → cherry, teal → spruce, yellow-green → birch/acacia, dark → dark oak, otherwise oak) | the map shows no species, only the colour; can be fixed in the block mapping |
| Colourful crowns vs. houses | purple/pink/magenta single objects in which the tree detector finds a crown are trees, not houses | a purple roof with a tree right on top would also count as a tree |
| Buildings | roofs = areas whose colour stands out from the town paving; single houses outside = compact, single-coloured blobs with contrast to their surroundings; roof block from roof colour | **height/storeys are not in the image** → estimated and flagged; simple interiors (floor, windows, door, ladder, lantern) |
| Heights | distance to the sea + valleys at rivers/lakes + terrain class (mountains in the middle of rock areas) + map shading (trees, icons, towns and class colours are removed first) + towns as level plateaus | **estimate** – for exact heights import a heightmap or correct with the brush. Measure the quality: `python -m map2mc.heightbench` |

`map2mc_report.json` in the world folder lists estimated areas and areas to check.

## World format (verified, not assumed)

* DataVersion **4671** = Minecraft 1.21.11 (read from `version.json` of the official 1.21.11 jar).
* Chunks in the format used since 1.18 (`sections`, palettes, packed long arrays, biomes 4×4×4), `Status: minecraft:full`;
  the game computes light and heightmaps itself on load.
* `level.dat` with the structure the 1.21.11 server writes itself (`spawn` compound, `DragonFight` …);
  outside the map a flat generator produces endless sea.
* Own NBT/Anvil implementation instead of Amulet-Core, because Amulet does not reliably support 1.21.11.
* `.litematic`: version 7 / sub-version 1 with `MinecraftDataVersion` 4671; bit array with entries spanning
  long boundaries, index `(y·Z + z)·X + x`. Cross-checked with the independent reader *litemapy*.
* `tools/validate_with_server.py` loads the world in a real vanilla server, loads every chunk
  (checked with `execute if loaded`) and compares sample blocks with `execute if block`.
  You must accept the Minecraft EULA yourself for this (`eula.txt`); the script never accepts it.

## Project structure

```
map2mc/
  imageio.py      load & validate images
  analysis.py     classification, cleanup, water, bridges, trees, buildings
  terrain.py      height model, lakes, height edits
  heightbench.py  measures the height estimate on synthetic islands with known heights
  features.py     deterministic noise, tree shapes, houses, platforms
  generator.py    generate regions in parallel, resume
  anvil.py        chunk/region writer (DataVersion 4671)
  nbt.py          read/write NBT
  level.py        level.dat
  project.py      project folder, checkpoints, edits, export
  preview.py      shaded top view, 3D preview
  worldread.py    read blocks from the generated region files
  blockmesh.py    textured block mesh (textures from the client jar)
  litematic.py    write/read .litematic
  treepack.py     import and place tree packs (.schem)
  scene3d.py      3D scene with levels of detail (LOD)
  gui/            PySide6 GUI
  cli.py          command line
tools/            server validation, area render
tests/            unit and end-to-end tests (python -m unittest discover -s tests)
```

Note: this program is a general converter for any map image. Third-party map images
(e.g. from games) are subject to their owners' rights – use the generated world privately only or clear the rights.
