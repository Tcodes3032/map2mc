"""Language support (English / German).

The program texts are written in German; for English every text that reaches the user
goes through tr(): exact matches from EN, templates with {} placeholders from EN_PATTERNS
(placeholders are translated recursively), multi-line texts line by line.

The GUI calls install() once: it routes QMessageBox / QFileDialog / setText-like calls
through tr() and translate_tree() translates widgets that were built with German texts.
"""
from __future__ import annotations

import re

LANGUAGES = {"en": "English", "de": "Deutsch"}
DEFAULT = "en"
LANG = DEFAULT


def set_language(code: str):
    global LANG
    LANG = code if code in LANGUAGES else DEFAULT


EN = {
    # --- window, menu, groups
    "Map2MC – Kartenbild → Minecraft Java 1.21.11": "Map2MC – map image → Minecraft Java 1.21.11",
    "&Datei": "&File",
    "Kartenbild hochladen (neues Projekt)…": "Upload map image (new project)…",
    "Projekt öffnen…": "Open project…",
    "Speichern": "Save",
    "1 · Kartenbild": "1 · Map image",
    "Kartenbild hochladen…": "Upload map image…",
    "Kein Projekt geladen.": "No project loaded.",
    "2 · Größe, Höhe && Erkennung": "2 · Size, height && detection",
    "Auf Insel zuschneiden": "Crop to island",
    "Gebäude erzeugen": "Generate buildings",
    "Aus: keine Gebäude in der Welt – Städte werden gepflasterte Flächen.":
        "Off: no buildings in the world – towns become paved areas.",
    "Einzelhäuser außerhalb von Städten erkennen": "Detect single houses outside towns",
    "Zusätzliche Füllbäume in Wäldern (nicht auf der Karte)": "Extra filler trees in forests (not on the map)",
    "Aus: nur Bäume, die auf der Karte zu sehen sind. An: Wälder werden zusätzlich mit zufälligen Bäumen aufgefüllt.":
        "Off: only trees visible on the map. On: forests are additionally filled with random trees.",
    "Bäume aus Baum-Paket (Standard)": "Trees from tree pack (default)",
    "Bäume aus einem importierten Schematic-Paket statt der eingebauten Formen. Blätter werden beim Import auf persistent=true gesetzt.":
        "Trees from an imported schematic pack instead of the built-in shapes. Leaves are set to persistent=true on import.",
    "Baum-Paket importieren…": "Import tree pack…",
    "kein Baum-Paket": "no tree pack",
    "kein Baum-Paket (eingebaute Baumformen)": "no tree pack (built-in tree shapes)",
    "Weltgröße:": "World size:",
    "3 · Analyse": "3 · Analysis",
    "Karte analysieren / neu berechnen": "Analyse map / recalculate",
    "Blockzuordnung…": "Block mapping…",
    "Gebäude…": "Buildings…",
    "Höhenkarte importieren…": "Import heightmap…",
    "4 · Manuelle Bearbeitung": "4 · Manual editing",
    "Werkzeug": "Tool",
    "Klasse": "Class",
    "Pinsel (px)": "Brush (px)",
    "Höhen-Stärke": "Height strength",
    "5 · Minecraft-Welt": "5 · Minecraft world",
    "Weltname": "World name",
    "Spielmodus": "Game mode",
    "Überleben": "Survival",
    "Kreativ": "Creative",
    "Ausgabeordner": "Output folder",
    "Generierung starten": "Start generation",
    "Abbrechen": "Cancel",
    "In Minecraft-Saves kopieren": "Copy to Minecraft saves",
    "Als ZIP exportieren…": "Export as ZIP…",
    "Weltordner öffnen": "Open world folder",
    "Ansicht:": "View:",
    "Klassen überlagern": "Overlay classes",
    "Gebäude anzeigen": "Show buildings",
    "Karte": "Map",
    "Terrain-Vorschau 3D": "Terrain preview 3D",
    "Drehen:": "Rotate:",
    "Überhöhung:": "Exaggeration:",
    "Sprache": "Language",
    "Sprache / Language": "Language / Sprache",
    # --- views and tools
    "Original": "Original",
    "Klassen": "Classes",
    "Höhe": "Height",
    "Vorschau (schattiert)": "Preview (shaded)",
    "Original + Prüfbereiche": "Original + areas to check",
    "Ansehen / Verschieben": "View / pan",
    "Klasse malen": "Paint class",
    "Klasse: Automatik wiederherstellen": "Class: restore automatic",
    "Höhe +": "Height +",
    "Höhe −": "Height −",
    "Ebnen": "Flatten",
    "Farbe für Klasse aufnehmen": "Pick colour for class",
    "Gebäude bearbeiten": "Edit building",
    "Gebäude hinzufügen (Rechteck ziehen)": "Add building (drag rectangle)",
    "Gebäude entfernen (Klick)": "Remove building (click)",
    "3D-Ausschnitt wählen (Klick)": "Pick 3D area (click)",
    "Ansehen": "View",
    "3D-Ausschnitt": "3D area",
    "Farbe": "Colour",
    "Gebäude hinzufügen": "Add building",
    "Gebäude entfernen": "Remove building",
    "Klasse: Automatik": "Class: automatic",
    "Vorschau": "Preview",
    "Prüfbereiche": "Areas to check",
    "Gebäude": "Building",
    "Höhenkarte": "Heightmap",
    # --- settings fields
    "Maßstab (Blöcke pro Pixel)": "Scale (blocks per pixel)",
    "Wie viele Minecraft-Blöcke ein Bildpixel wird. 1 Block = 1 m.":
        "How many Minecraft blocks one image pixel becomes. 1 block = 1 m.",
    "Meeresspiegel (Y)": "Sea level (Y)",
    "Y-Koordinate der obersten Wasserschicht.": "Y coordinate of the top water layer.",
    "Max. Landhöhe (Blöcke)": "Max. land height (blocks)",
    "Höhe normaler Landflächen über dem Meer.": "Height of normal land above the sea.",
    "Berg-Zusatzhöhe": "Extra mountain height",
    "Zusätzliche Höhe für Fels/Vulkan.": "Additional height for rock/volcano.",
    "Strandhöhe": "Beach height",
    "Max. Meerestiefe": "Max. ocean depth",
    "Relief aus Schattierung": "Relief from shading",
    "Nutzt die Schattierung des Kartenbilds als Relief. 0 = aus.": "Uses the shading of the map image as relief. 0 = off.",
    "Oberflächendetail": "Surface detail",
    "Kleine Unebenheiten (verändert keine Struktur).": "Small bumps (does not change the structure).",
    "Glättung": "Smoothing",
    "Farbtoleranz (Lab)": "Colour tolerance (Lab)",
    "Pixel weiter entfernt von allen Farbprototypen werden als 'unbekannt' markiert.":
        "Pixels further away from all colour prototypes are marked as 'unknown'.",
    "Min. Flächengröße (px)": "Min. area size (px)",
    "Baumdichte (gesamt)": "Tree density (overall)",
    "Baumabstand im Wald": "Tree spacing in forests",
    "Standard-Stockwerke": "Default storeys",
    "Gilt für alle Gebäude ohne bekannte Höhe.": "Applies to all buildings without a known height.",
    "Stockwerkshöhe": "Storey height",
    "Min. Gebäudegröße (px)": "Min. building size (px)",
    "Einzelhaus-Empfindlichkeit": "Single-house sensitivity",
    "Kontrast, ab dem ein kompakter Fleck als Haus gilt. Kleiner = mehr Häuser (auch mehr Fehltreffer).":
        "Contrast from which a compact spot counts as a house. Lower = more houses (also more false hits).",
    "Einzelhaus Mindestgröße (px)": "Single house min. size (px)",
    "Meeresrand (Blöcke)": "Ocean margin (blocks)",
    "Prozesse (0 = automatisch)": "Processes (0 = automatic)",
    # --- 3D tab
    "3D-Blöcke / Hologramm": "3D blocks / hologram",
    " Blöcke": " blocks",
    "Mitte X:": "Centre X:",
    "Detailbereich:": "Detail area:",
    "Laden": "Load",
    "ganze Karte (LOD)": "whole map (LOD)",
    "Wie Voxy: um die Kamera echte Blöcke, weiter weg vereinfachte Säulen – so ist die ganze Karte sichtbar.":
        "Like Voxy: real blocks around the camera, simplified columns further away – so the whole map is visible.",
    "Detail folgt Kamera": "Detail follows camera",
    "Wenn du dich bewegst, wird der Detailbereich an die neue Stelle gelegt.":
        "When you move, the detail area is moved to the new place.",
    "Spieler (WASD)": "Player (WASD)",
    "Orbit (Maus)": "Orbit (mouse)",
    "Spieler: ins Bild klicken, Maus umsehen, WASD bewegen, Leertaste/Shift hoch/runter, F Fliegen/Laufen, Esc Maus freigeben.\nOrbit: Maus ziehen zum Drehen/Verschieben.":
        "Player: click into the view, mouse to look around, WASD to move, Space/Shift up/down, F fly/walk, Esc releases the mouse.\nOrbit: drag the mouse to rotate/pan.",
    "Hologramm": "Hologram",
    "Ebenen bis:": "Layers up to:",
    "alle": "all",
    "Ausschnitt als .litematic…": "Area as .litematic…",
    "Ganze Welt als .litematic-Kacheln…": "Whole world as .litematic tiles…",
    "Linke Maus: drehen · Rechte Maus: verschieben · Rad: Zoom · Doppelklick: zurücksetzen. Ausschnitt auf der Karte mit „3D-Ausschnitt wählen“ anklicken.":
        "Left mouse: rotate · Right mouse: pan · Wheel: zoom · Double click: reset. Pick the area on the map with “Pick 3D area”.",
    "Orbit: linke Maus drehen · rechte Maus verschieben · Rad zoomen · Doppelklick zurücksetzen":
        "Orbit: left mouse rotate · right mouse pan · wheel zoom · double click reset",
    "Laufen": "Walking",
    "Esc: Maus freigeben": "Esc: release mouse",
    "Klick ins Bild: Maus fangen": "Click into the view: capture mouse",
    "WASD bewegen · Leertaste/Shift hoch/runter (Laufen: springen) · Strg sprinten":
        "WASD move · Space/Shift up/down (walking: jump) · Ctrl sprint",
    "OpenGL nicht verfügbar": "OpenGL not available",
    # --- file dialogs
    "Kartenbild wählen": "Choose map image",
    "Bilder (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)": "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
    "Projektordner wählen (wird angelegt)": "Choose project folder (will be created)",
    "Projektordner öffnen": "Open project folder",
    "Ausgabeordner (Weltordner wird darin angelegt)": "Output folder (the world folder is created inside)",
    "Höhenkarte (Graustufen, hell = hoch)": "Heightmap (greyscale, bright = high)",
    "Bilder (*.png *.tif *.tiff *.jpg *.webp)": "Images (*.png *.tif *.tiff *.jpg *.webp)",
    ".litematic speichern": "Save .litematic",
    "Ordner für .litematic-Kacheln (z. B. .minecraft/schematics)": "Folder for .litematic tiles (e.g. .minecraft/schematics)",
    "Baum-Paket wählen (.7z / .zip mit .schem-Dateien)": "Choose tree pack (.7z / .zip with .schem files)",
    "Baum-Pakete (*.7z *.zip)": "Tree packs (*.7z *.zip)",
    "Minecraft-Saves-Ordner": "Minecraft saves folder",
    "ZIP speichern": "Save ZIP",
    # --- messages
    "Fehler": "Error",
    "Map2MC – Fehler": "Map2MC – error",
    "Bild nicht verwendbar": "Image not usable",
    "Bitte zuerst ein Kartenbild hochladen oder ein Projekt öffnen.": "Please upload a map image or open a project first.",
    "Bitte zuerst die Karte analysieren.": "Please analyse the map first.",
    "Es läuft bereits ein Vorgang.": "A task is already running.",
    "Abbruch angefordert …": "Cancel requested …",
    "Kein Map2MC-Projektordner (source.png fehlt).": "Not a Map2MC project folder (source.png missing).",
    "Analyse gestartet …": "Analysis started …",
    "Farben klassifizieren …": "Classifying colours …",
    "Gelände, Gewässer, Bäume, Gebäude ableiten …": "Deriving terrain, water, trees, buildings …",
    "Blockzuordnung gespeichert.": "Block mapping saved.",
    "Klassen geändert – „Karte analysieren / neu berechnen“ übernimmt die Änderung in Höhe, Bäume und Gebäude.":
        "Classes changed – “Analyse map / recalculate” applies the change to heights, trees and buildings.",
    "Kein Gebäude an dieser Stelle.": "No building at this spot.",
    "Im Ausgabeordner liegt bereits eine fremde Welt. Bitte anderen Ordner wählen.":
        "The output folder already contains another world. Please choose a different folder.",
    "Klassen wurden bemalt, aber noch nicht neu berechnet. Trotzdem mit dem alten Stand generieren?":
        "Classes were painted but not recalculated yet. Generate with the old state anyway?",
    "Fertig. Die Welt kann jetzt in die Minecraft-Saves kopiert oder als ZIP exportiert werden.":
        "Done. The world can now be copied to the Minecraft saves or exported as ZIP.",
    "Bitte zuerst die Welt generieren – die 3D-Blockansicht und der .litematic-Export lesen die erzeugten Regionsdateien.":
        "Please generate the world first – the 3D block view and the .litematic export read the generated region files.",
    "Kein Minecraft-1.21-Client gefunden – Blöcke werden einfarbig dargestellt.":
        "No Minecraft 1.21 client found – blocks are shown in plain colours.",
    "Lade 3D-Ansicht …": "Loading 3D view …",
    "Dieser Ausschnitt ist leer (liegt außerhalb der erzeugten Welt).": "This area is empty (outside the generated world).",
    "Exportiere .litematic …": "Exporting .litematic …",
    "Exportiere .litematic-Kacheln …": "Exporting .litematic tiles …",
    "Bitte zuerst die Welt generieren.": "Please generate the world first.",
    "Ein Vorgang läuft noch. Abbrechen und beenden?": "A task is still running. Cancel and quit?",
    ". Fehlende Arten (z. B. Kirsche) nutzen Eichen-Formen mit passendem Laub. Welt neu generieren, um die Bäume zu verwenden.":
        ". Missing kinds (e.g. cherry) use oak shapes with matching leaves. Regenerate the world to use the trees.",
    "Die Sprache wird umgestellt …": "Switching language …",
    "Bitte warten, bis der laufende Vorgang fertig ist.": "Please wait until the running task has finished.",
    # --- core modules (logs / errors)
    "Abgebrochen – bereits fertige Regionen bleiben erhalten (Fortsetzen möglich).":
        "Cancelled – finished regions are kept (can be resumed).",
    "Bild enthält praktisch nur eine Farbe – keine Karte erkennbar.": "The image contains practically one colour – no map recognisable.",
    "Höhen sind aus Küstenabstand, Geländeklasse und Kartenschattierung geschätzt – das Kartenbild enthält keine echten Höhendaten. Für exakte Höhen eine Höhenkarte importieren.":
        "Heights are estimated from coast distance, terrain class and map shading – the map image has no real height data. Import a heightmap for exact heights.",
    "Innenräume, Türen und Dachformen sind nicht aus dem Kartenbild ableitbar und werden schlicht gebaut.":
        "Interiors, doors and roof shapes cannot be derived from the map image and are built plainly.",
    "Im Paket wurden keine verwendbaren Bäume gefunden.": "No usable trees were found in the pack.",
    "Baum-Paket: bitte einen Ordner, eine .zip- oder .7z-Datei wählen": "Tree pack: please choose a folder, a .zip or a .7z file",
    "Erzeuge Übersichtsdaten (einmalig für diese Welt) …": "Creating overview data (once for this world) …",
    # --- classes
    "Hintergrund (außerhalb der Karte)": "Background (outside the map)",
    "Tiefes Wasser": "Deep water",
    "Flaches Wasser": "Shallow water",
    "Sand / Strand": "Sand / beach",
    "Gras / Wiese": "Grass / meadow",
    "Wald": "Forest",
    "Wüste / roter Sand": "Desert / red sand",
    "Fels / Berg": "Rock / mountain",
    "Dunkler Fels / Vulkan": "Dark rock / volcano",
    "Weg (Erde)": "Path (dirt)",
    "Straße (Asphalt)": "Road (asphalt)",
    "Bebauung / Gebäude": "Built-up / buildings",
    "Schnee": "Snow",
    "Blütenbäume (rosa)": "Blossom trees (pink)",
    "Lila Gebiet": "Purple area",
    "Unbekannt – prüfen!": "Unknown – check!",
    # --- dialogs
    "Oberfläche": "Surface",
    "Untergrund": "Subsurface",
    "Tiefe": "Depth",
    "Biom": "Biome",
    "Baumart": "Tree kind",
    "Baumdichte": "Tree density",
    "Pflanzen": "Plants",
    "Blockzuordnung bearbeiten": "Edit block mapping",
    "Standard wiederherstellen": "Restore defaults",
    "Mischungen: <code>minecraft:stone*4,minecraft:andesite</code> (Gewicht nach *). Blockzustände z. B. <code>minecraft:basalt[axis=y]</code>. Bei Wasserklassen ist „Oberfläche“ der Gewässerboden.":
        "Mixes: <code>minecraft:stone*4,minecraft:andesite</code> (weight after *). Block states e.g. <code>minecraft:basalt[axis=y]</code>. For water classes “Surface” is the water bed.",
    "Gelb = Höhe geschätzt (nicht aus der Karte ablesbar). Doppelklick zeigt das Gebäude. Typ „none“ entfernt ein Gebäude, „platform“ baut ein Deck (z. B. Schiffe/Stege).":
        "Yellow = height estimated (not readable from the map). Double click shows the building. Type “none” removes a building, “platform” builds a deck (e.g. ships/jetties).",
    "Fläche (Blöcke²)": "Area (blocks²)",
    "Typ": "Type",
    "Stockwerke": "Storeys",
    "Dach-Block": "Roof block",
    "Wand-Block (leer = Standard)": "Wall block (empty = default)",
    "Name": "Name",
    "Höhe ist geschätzt.": "Height is estimated.",
    "Einzelobjekt": "Single object",
}

EN_PATTERNS = {
    "Neues Projekt: {}": "New project: {}",
    "{}\nBild: {}×{} px": "{}\nImage: {}×{} px",
    "Höhenkarte importiert (Land: 1 … {} Blöcke über dem Meer).": "Heightmap imported (land: 1 … {} blocks above sea level).",
    "Gebäude {} hinzugefügt ({} × {} Blöcke, Dach {}). Mit „Gebäude bearbeiten“ Stockwerke/Typ ändern.":
        "Building {} added ({} × {} blocks, roof {}). Change storeys/type with “Edit building”.",
    "Generiere Welt nach {} …": "Generating world into {} …",
    "Importiere Baum-Paket {} …": "Importing tree pack {} …",
    "Welt kopiert nach {}": "World copied to {}",
    "{} × {} Blöcke  (≈ {} × {} km)": "{} × {} blocks  (≈ {} × {} km)",
    "{} × {} Blöcke (ganzes Bild)": "{} × {} blocks (whole image)",
    "Unbekannte Pixel: {} %, Prüfbereiche: {} % (Ansicht „Original + Prüfbereiche“).":
        "Unknown pixels: {} %, areas to check: {} % (view “Original + areas to check”).",
    "Farbe {} als Prototyp für „{}“ hinzugefügt – bitte neu analysieren.":
        "Colour {} added as prototype for “{}” – please analyse again.",
    "3D: Detailbereich um x {}, z {} ({}×{} Blöcke).": "3D: detail area around x {}, z {} ({}×{} blocks).",
    "3D: Detailbereich um x {}, z {} ({}×{} Blöcke), Rest der Karte vereinfacht (Zellen 2 / {} Blöcke).":
        "3D: detail area around x {}, z {} ({}×{} blocks), rest of the map simplified (cells 2 / {} blocks).",
    "Y {}": "Y {}",
    "{} .litematic-Kacheln (256×256) gespeichert in {}. Der Dateiname enthält die Weltkoordinate der Kachel-Ecke.":
        "{} .litematic tiles (256×256) saved in {}. The file name contains the world coordinate of the tile corner.",
    "ZIP gespeichert: {}": "ZIP saved: {}",
    "{}\n\nDetails stehen in:\n{}\n\n{}": "{}\n\nDetails are in:\n{}\n\n{}",
    "Gebäude {}: {} (erneut klicken = rückgängig).": "Building {}: {} (click again = undo).",
    "Gebäude {}: {}, {} Stockwerk(e).": "Building {}: {}, {} storey(s).",
    "Standard-Baumpaket (kuratiert, {} Bäume)": "Default tree pack (curated, {} trees)",
    "Baum-Paket: {}": "Tree pack: {}",
    "Dort existiert bereits eine Welt mit diesem Namen:\n{}": "A world with this name already exists there:\n{}",
    "Letztes Projekt konnte nicht geladen werden: {}": "Could not load the last project: {}",
    ".litematic gespeichert: {} – {} Blöcke, Ursprung in der Welt x={} y={} z={}":
        ".litematic saved: {} – {} blocks, origin in the world x={} y={} z={}",
    "Baum-Paket importiert: {} Bäume ({}). Nicht übernommen: {}. Fehlende Arten (z. B. Kirsche) nutzen Eichen-Formen mit passendem Laub. Welt neu generieren, um die Bäume zu verwenden.":
        "Tree pack imported: {} trees ({}). Not taken: {}. Missing kinds (e.g. cherry) use oak shapes with matching leaves. Regenerate the world to use the trees.",
    "X {}  Y {}  Z {}   {}": "X {}  Y {}  Z {}   {}",
    "FEHLER: {}": "ERROR: {}",
    "FEHLER (3D): {}": "ERROR (3D): {}",
    "Hinweis: {}": "Note: {}",
    "Job-Fehler:\n{}": "Job error:\n{}",
    "3D-Ansicht: {}": "3D view: {}",
    "Fliegen ({} Blöcke/s)": "Flying ({} blocks/s)",
    "F Fliegen/Laufen · Mausrad Tempo · R Start · {}": "F fly/walk · Wheel speed · R start · {}",
    "Gebäude {}": "Building {}",
    "Dachfarbe aus Karte: RGB {} → {}": "Roof colour from map: RGB {} → {}",
    "Projekt angelegt: {} (Maßstab {} Blöcke/Pixel)": "Project created: {} (scale {} blocks/pixel)",
    "fertig in {} s": "done in {} s",
    "Welt {}×{} Blöcke, {} Regionen ({} zu erzeugen).": "World {}×{} blocks, {} regions ({} to generate).",
    "Welt fertig: {}": "World done: {}",
    "Fortsetzen: {} Regionen bereits fertig.": "Resuming: {} regions already done.",
    "Datei nicht gefunden: {}": "File not found: {}",
    "Dateityp {} wird nicht unterstützt. Erlaubt: {}": "File type {} is not supported. Allowed: {}",
    "Bild ist zu klein ({}×{}); mindestens {}×{} Pixel.": "Image is too small ({}×{}); at least {}×{} pixels.",
    "Bild ist zu groß ({}).": "Image is too large ({}).",
    "Bild ist beschädigt oder unlesbar: {}": "Image is damaged or unreadable: {}",
    "Analyse fertig: {} Einzelbäume, {} Gebäude, {} Brücken-Pixel, {} Binnengewässer.":
        "Analysis done: {} single trees, {} buildings, {} bridge pixels, {} inland waters.",
    "{} Gebäudehöhen sind geschätzt (Standard {} Stockwerk(e)) – in der Gebäudeliste korrigierbar.":
        "{} building heights are estimated (default {} storey(s)) – can be corrected in the building list.",
    "  übersprungen (nicht lesbar): {}": "  skipped (unreadable): {}",
}


def _compile(t: str):
    return re.compile("^" + re.escape(t).replace(r"\{\}", "(.*?)") + "$", re.S)


_PATTERNS = [(_compile(de), en) for de, en in sorted(EN_PATTERNS.items(), key=lambda kv: -len(kv[0]))]


def tr(s):
    """Text in the current language (German texts are the keys)."""
    if LANG == "de" or not isinstance(s, str) or not s.strip():
        return s
    hit = EN.get(s)
    if hit is not None:
        return hit
    for rx, en in _PATTERNS:
        m = rx.match(s)
        if m:
            parts = iter(tr(g) for g in m.groups())
            return re.sub(r"\{\}", lambda _: next(parts), en)
    if "\n" in s:
        return "\n".join(tr(line) for line in s.split("\n"))
    st = s.strip()
    if st != s and st in EN:
        return s.replace(st, EN[st])
    return s


# ------------------------------------------------------------------ Qt integration
_installed = False


def install():
    """Route Qt texts through tr() (no effect in German). Safe to call more than once."""
    global _installed
    if _installed:
        return
    _installed = True
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QFileDialog, QGroupBox, QLabel, QLineEdit,
                                   QMessageBox, QPlainTextEdit, QStatusBar, QWidget)

    def wrap(cls, name, idx=(0,)):
        orig = getattr(cls, name)

        def f(*a, **k):
            a = list(a)
            for i in idx:
                if i < len(a):
                    a[i] = tr(a[i])
            return orig(*a, **k)
        setattr(cls, name, f)

    for cls in (QLabel, QAbstractButton, QLineEdit):
        wrap(cls, "setText", (1,))
    wrap(QPlainTextEdit, "appendPlainText", (1,))
    wrap(QWidget, "setToolTip", (1,))
    wrap(QWidget, "setWindowTitle", (1,))
    wrap(QGroupBox, "setTitle", (1,))
    wrap(QStatusBar, "showMessage", (1,))
    wrap(QComboBox, "addItem", (1,))
    orig_items = QComboBox.addItems
    QComboBox.addItems = lambda self, items: orig_items(self, [tr(t) for t in items])
    for name in ("information", "warning", "critical", "question"):
        orig = getattr(QMessageBox, name)

        def mb(*a, _o=orig, **k):
            a = list(a)
            for i in (1, 2):
                if i < len(a):
                    a[i] = tr(a[i])
            return _o(*a, **k)
        setattr(QMessageBox, name, staticmethod(mb))
    from PySide6.QtWidgets import QDialog
    orig_exec = QDialog.exec

    def dexec(self, *a, **k):
        translate_tree(self)
        return orig_exec(self, *a, **k)
    QDialog.exec = dexec
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        orig = getattr(QFileDialog, name)

        def fd(*a, _o=orig, **k):
            a = list(a)
            if len(a) > 1:
                a[1] = tr(a[1])
            if len(a) > 3:
                a[3] = tr(a[3])
            for key in ("caption", "filter"):
                if key in k:
                    k[key] = tr(k[key])
            return _o(*a, **k)
        setattr(QFileDialog, name, staticmethod(fd))


def translate_tree(root):
    """Translate texts of a widget built with German strings and all of its children."""
    if LANG == "de":
        return
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QComboBox, QGroupBox, QLabel, QMenu,
                                   QTableWidget, QTabWidget, QWidget)
    for w in [root] + root.findChildren(QWidget):
        if w.toolTip():
            w.setToolTip(w.toolTip())
        if w.isWindow() and w.windowTitle():
            w.setWindowTitle(w.windowTitle())
        if isinstance(w, (QLabel, QAbstractButton)) and w.text():
            w.setText(w.text())
        elif isinstance(w, QGroupBox):
            w.setTitle(w.title())
        elif isinstance(w, QComboBox):
            for i in range(w.count()):
                w.setItemText(i, tr(w.itemText(i)))
        elif isinstance(w, QTabWidget):
            for i in range(w.count()):
                w.setTabText(i, tr(w.tabText(i)))
        elif isinstance(w, QMenu):
            w.setTitle(tr(w.title()))
            for act in w.actions():
                act.setText(tr(act.text()))
        elif isinstance(w, QTableWidget):
            for i in range(w.columnCount()):
                it = w.horizontalHeaderItem(i)
                if it is not None:
                    it.setText(tr(it.text()))
        if isinstance(w, QAbstractSpinBox) and hasattr(w, "suffix") and w.suffix():
            w.setSuffix(tr(w.suffix()))
    for act in root.actions() if hasattr(root, "actions") else []:
        act.setText(tr(act.text()))
