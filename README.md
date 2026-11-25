# Map2MC – Kartenbild → Minecraft-Java-Welt (1.21.11)

Wandelt ein Top-Down-Kartenbild (z. B. eine Battle-Royale-Insel) in eine **echte, begehbare
Minecraft-Java-Welt** um: Region-Dateien (`.mca`) plus `level.dat`, direkt im Spiel ladbar.

## Schnellstart

```powershell
cd C:\Users\tobias\Downloads\map2mc
.\start_gui.bat
```

Oder per Kommandozeile:

```powershell
.venv\Scripts\python -m map2mc new samples\island.webp projekte\insel
.venv\Scripts\python -m map2mc analyse projekte\insel
.venv\Scripts\python -m map2mc generate projekte\insel welten\Insel --zip welten\Insel.zip
```

Die fertige Welt in `%APPDATA%\.minecraft\saves\` kopieren (GUI: „In Minecraft-Saves kopieren“).
Neu einrichten: `uv venv --python 3.12 .venv` und `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`
(PySide6 braucht 64-bit-Python ≤ 3.13).

## Ablauf in der Oberfläche

1. **Kartenbild hochladen** – Bild wird geprüft (leer, beschädigt, abgeschnitten, zu klein, einfarbig).
2. **Größe & Höhe** – Maßstab (Blöcke pro Pixel), Meeresspiegel, Land-/Berghöhe, Meerestiefe.
   Die Weltgröße in Blöcken/km wird live angezeigt. 1 Block = 1 m.
3. **Analysieren** – Klassifizierung, Gelände, Gewässer, Bäume, Gebäude.
4. **Prüfen & korrigieren** (Ansichten: Original, Klassen, Höhe, schattierte Vorschau, Prüfbereiche,
   3D-Vorschau):
   - *Klasse malen* / *Automatik wiederherstellen* – Pinsel auf der Klassenkarte
   - *Höhe + / Höhe − / Ebnen* – Höhen-Pinsel (sofort wirksam)
   - *Farbe für Klasse aufnehmen* – Pipette fügt einen Farbprototyp hinzu (danach neu analysieren)
   - *Gebäude bearbeiten* (Klick) / *Gebäude hinzufügen* (Rechteck ziehen) / *Gebäude entfernen* (Klick,
     erneut klicken = rückgängig) / Gebäudeliste
   - Häkchen *Gebäude erzeugen* (aus = keine Gebäude, Städte werden gepflasterte Flächen) und
     *Einzelhäuser außerhalb von Städten erkennen* (Empfindlichkeit und Mindestgröße einstellbar)
   - *Höhenkarte importieren* – echte Höhendaten (Graustufenbild) ersetzen die Schätzung
   - *Baum-Paket importieren* – .7z/.zip mit WorldEdit-`.schem`-Bäumen (z. B. „5K Trees“). Natürliche
     Bäume werden nach Blattart und Kronengröße einsortiert, **alle Blätter auf `persistent=true` gesetzt**
     (sonst zerfallen Bäume ohne Stamm), alte Blocknamen angepasst, Glas-/Fantasy-/Kristall-Objekte und
     zu große Bäume aussortiert. Beim Generieren wird je Baum ein Paketbaum ähnlicher Größe gewählt
     (deterministisch, gedreht); fehlende Arten wie Kirsche bekommen Eichen-Formen mit passendem Laub.
     Kronen, die in Häuser wachsen würden, werden ausgelassen.
   - *Zusätzliche Füllbäume in Wäldern* (Standard: aus) – aus heißt: nur die Bäume, die auf der Karte
     zu sehen sind; an füllt Waldflächen zusätzlich mit zufälligen Bäumen (alter Stand).
   - *Blockzuordnung* – Oberfläche, Untergrund, Biom, Baumart, Dichte je Klasse; Mischungen
     wie `minecraft:stone*4,minecraft:andesite`
5. **Generierung starten** – parallel in Regionen (512×512 Blöcke), mit Fortschritt und Abbruch.
   Fertige Regionen werden sofort gespeichert; ein erneuter Start setzt fort.
6. **3D-Blöcke / Hologramm** – zeigt die *erzeugte* Welt wie die Mod Voxy: um den Fokus echte Blöcke mit
   den Minecraft-Texturen (aus dem installierten 1.21.11-Client, sonst einfarbig), bis 384 Blöcke weiter
   2×2-Säulen, dahinter die ganze Karte als grobe Säulen (4–32 Blöcke je nach Kartengröße).
   Fokus mit dem Werkzeug *3D-Ausschnitt wählen* auf der Karte anklicken oder Mitte X/Z eingeben;
   mit *Detail folgt Kamera* wandert der Detailbereich mit, wenn du dich bewegst.
   Steuerung **Spieler (WASD)** (Standard, wie in Minecraft): ins Bild klicken fängt die Maus,
   Maus = umsehen, W/A/S/D bewegen, Leertaste/Shift hoch/runter, Strg schneller, Mausrad Fluggeschwindigkeit,
   **F** wechselt Fliegen ↔ Laufen (Laufen: Schwerkraft, Springen mit Leertaste, Wände ab 1 Block Höhe
   halten auf, im Wasser steht man brusttief), **R** zurück zum Start, **Esc** gibt die Maus frei.
   Steuerung **Orbit (Maus)**: linke Maus drehen, rechte Maus verschieben, Mausrad zoomen,
   Doppelklick zurücksetzen.
   Die Übersichtsdaten schreibt der Generator mit (`map2mc_lod/`); bei älteren Welten werden sie
   einmalig nachberechnet (≈ 30 s).
   *Hologramm* zeigt die Blöcke durchscheinend wie Litematica, *Ebenen bis* blendet alles oberhalb
   einer Höhe aus.
7. **Exportieren** – in die Minecraft-Saves kopieren, als ZIP, oder als **.litematic**
   (Ausschnitt oder ganze Welt in 256×256-Kacheln; der Dateiname enthält die Weltkoordinate der Ecke).
   Die Dateien in `.minecraft/schematics` legen und in Litematica laden.

**Sprache / Language:** Englisch ist Standard; im Menü *Sprache / Language* auf Deutsch umschaltbar
(das Fenster wird neu aufgebaut, das Projekt bleibt offen). Übersetzungen stehen in `map2mc/i18n.py`.

Alle Eingaben und Zwischenergebnisse liegen im Projektordner (`settings.json`, `palette.json`,
`prototypes.json`, `edits/`, `cache/`, `data/`) und werden laufend gespeichert.

## Was automatisch erkannt wird – und was nicht

| Element | Methode | Grenzen |
|---|---|---|
| Wasser, Küste, Strand, Wiese, Wald, Wüste, Fels, Vulkan, Wege, Straßen, Bebauung, Blütenbäume, lila Gebiete | Farb-Prototypen in CIE-Lab, Bereinigung, unbekannte Farben → Prüfbereich | neue Kartenstile brauchen evtl. zusätzliche Farbprototypen (Pipette) |
| Seen/Flüsse vs. Meer | Zusammenhang mit dem Kartenrand; Binnengewässer bekommen eigenen Wasserspiegel | – |
| Brücken | Weg-Pixel, die überwiegend von Wasser umgeben sind → Holzbrücke mit Geländer und Pfeilern | – |
| Einzelbäume | dunkle/rosa Kronen-Flecken (Multiskalen-DoG), Kronengröße aus der Fleckgröße; Baumart „auto“ wählt das Laub nach dem Farbton der Krone (rosa → Kirsche, türkis → Fichte, gelbgrün → Birke/Akazie, dunkel → Schwarzeiche, sonst Eiche) | die Karte zeigt keine Baumart, nur die Farbe; in der Blockzuordnung fest einstellbar |
| Bunte Baumkronen vs. Häuser | lila/pinke/magenta Einzelobjekte, in denen der Baumdetektor eine Krone findet, sind Bäume und keine Häuser | ein lila Dach mit Baum direkt darauf würde ebenfalls als Baum gelten |
| Gebäude | Dächer = Bereiche, die sich farblich vom Pflaster der Stadt abheben; einzelne Häuser außerhalb = kompakte, einfarbige Flecken mit Kontrast zur Umgebung; Dachblock aus Dachfarbe | **Höhe/Stockwerke stehen nicht im Bild** → geschätzt und markiert; Innenräume schlicht (Boden, Fenster, Tür, Leiter, Laterne) |
| Höhen | Abstand zum Meer + Täler an Flüssen/Seen + Geländeklasse (Berge in der Mitte von Felsflächen) + Schattierung der Karte (Bäume, Icons, Städte und Klassenfarben werden vorher herausgerechnet) + Städte als ebene Plateaus | **Schätzung** – für exakte Höhen eine Höhenkarte importieren oder mit dem Pinsel korrigieren. Qualität messen: `python -m map2mc.heightbench` |

`map2mc_report.json` im Weltordner listet geschätzte und zu prüfende Bereiche.

## Weltformat (geprüft, nicht angenommen)

* DataVersion **4671** = Minecraft 1.21.11 (aus `version.json` des offiziellen 1.21.11-Jars gelesen).
* Chunks im Format seit 1.18 (`sections`, Paletten, gepackte Long-Arrays, Biome 4×4×4), `Status: minecraft:full`,
  Licht und Heightmaps berechnet das Spiel beim Laden selbst.
* `level.dat` im Aufbau, den der 1.21.11-Server selbst schreibt (`spawn`-Compound, `DragonFight` …);
  außerhalb der Karte erzeugt ein flacher Generator endloses Meer.
* Eigene NBT/Anvil-Implementierung statt Amulet-Core, weil Amulet 1.21.11 nicht zuverlässig unterstützt.
* `.litematic`: Version 7 / SubVersion 1 mit `MinecraftDataVersion` 4671; Bit-Array mit über Long-Grenzen
  reichenden Einträgen, Index `(y·Z + z)·X + x`. Gegengeprüft mit dem unabhängigen Leser *litemapy*.
* `tools/validate_with_server.py` lädt die Welt in einem echten Vanilla-Server, lädt jeden Chunk
  (geprüft mit `execute if loaded`) und vergleicht Stichproben-Blöcke mit `execute if block`.
  Die Minecraft-EULA muss dafür von dir selbst akzeptiert sein (`eula.txt`); das Skript akzeptiert sie nie.

## Projektstruktur

```
map2mc/
  imageio.py    Bild laden & prüfen
  analysis.py   Klassifizierung, Bereinigung, Gewässer, Brücken, Bäume, Gebäude
  terrain.py    Höhenmodell, Seen, Höhen-Edits
  heightbench.py  Messung der Höhenschätzung an künstlichen Inseln mit bekannten Höhen
  features.py   deterministisches Rauschen, Baumformen, Häuser, Plattformen
  generator.py  Regionen parallel erzeugen, Fortsetzen
  anvil.py      Chunk-/Region-Writer (DataVersion 4671)
  nbt.py        NBT lesen/schreiben
  level.py      level.dat
  project.py    Projektordner, Checkpoints, Edits, Export
  preview.py    schattierte Draufsicht, 3D-Vorschau
  worldread.py  Blöcke aus den erzeugten Regionsdateien lesen
  blockmesh.py  texturiertes Block-Mesh (Texturen aus dem Client-Jar)
  litematic.py  .litematic schreiben/lesen
  treepack.py   Baum-Pakete (.schem) importieren und platzieren
  scene3d.py    3D-Szene mit Detailstufen (LOD)
  gui/          PySide6-Oberfläche
  cli.py        Kommandozeile
tools/          Server-Validierung, Gebiets-Render
tests/          Unit- und End-to-End-Tests (python -m unittest discover -s tests)
```

Hinweis: Das Programm ist ein allgemeiner Konverter für beliebige Kartenbilder. Kartenbilder Dritter
(z. B. von Spielen) unterliegen deren Rechten – die erzeugte Welt nur privat nutzen bzw. die Rechte klären.
