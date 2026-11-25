"""Main window of the Map → Minecraft generator."""
from __future__ import annotations

import os
import sys
import threading
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import analysis, i18n, preview
from ..config import CID, CLASS_KEYS, CLASSES, Settings
from ..imageio import MapImageError
from ..project import Project, copy_world, export_zip, minecraft_saves_dir
from .blockview import BlockView
from .canvas import MapCanvas, to_qimage
from .dialogs import BuildingDialog, PaletteDialog, SingleBuildingDialog


# ------------------------------------------------------------------ background jobs
class Job(QObject):
    progress = Signal(float)
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            res = self.fn(self)
            self.done.emit(res)
        except MapImageError as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc())


# setting name -> (label, kind, min, max, step, tooltip)
SETTING_FIELDS = [
    ("blocks_per_pixel", "Maßstab (Blöcke pro Pixel)", float, 0.25, 16, 0.25,
     "Wie viele Minecraft-Blöcke ein Bildpixel wird. 1 Block = 1 m."),
    ("water_level", "Meeresspiegel (Y)", int, -50, 200, 1, "Y-Koordinate der obersten Wasserschicht."),
    ("max_land_height", "Max. Landhöhe (Blöcke)", int, 4, 200, 1, "Höhe normaler Landflächen über dem Meer."),
    ("mountain_height", "Berg-Zusatzhöhe", int, 0, 200, 1, "Zusätzliche Höhe für Fels/Vulkan."),
    ("beach_height", "Strandhöhe", float, 0.5, 10, 0.5, ""),
    ("max_ocean_depth", "Max. Meerestiefe", int, 2, 100, 1, ""),
    ("relief_strength", "Relief aus Schattierung", float, 0, 3, 0.1,
     "Nutzt die Schattierung des Kartenbilds als Relief. 0 = aus."),
    ("detail_noise", "Oberflächendetail", float, 0, 5, 0.1, "Kleine Unebenheiten (verändert keine Struktur)."),
    ("smoothing", "Glättung", float, 0, 8, 0.25, ""),
    ("max_color_distance", "Farbtoleranz (Lab)", float, 5, 80, 1,
     "Pixel weiter entfernt von allen Farbprototypen werden als 'unbekannt' markiert."),
    ("min_region_px", "Min. Flächengröße (px)", int, 1, 500, 1, ""),
    ("tree_density", "Baumdichte (gesamt)", float, 0, 3, 0.1, ""),
    ("tree_spacing", "Baumabstand im Wald", int, 3, 30, 1, ""),
    ("default_storeys", "Standard-Stockwerke", int, 1, 20, 1, "Gilt für alle Gebäude ohne bekannte Höhe."),
    ("storey_height", "Stockwerkshöhe", int, 3, 8, 1, ""),
    ("min_building_px", "Min. Gebäudegröße (px)", int, 1, 500, 1, ""),
    ("object_contrast", "Einzelhaus-Empfindlichkeit", float, 8, 80, 1,
     "Kontrast, ab dem ein kompakter Fleck als Haus gilt. Kleiner = mehr Häuser (auch mehr Fehltreffer)."),
    ("object_min_px", "Einzelhaus Mindestgröße (px)", int, 3, 200, 1, ""),
    ("ocean_margin", "Meeresrand (Blöcke)", int, 0, 2000, 16, ""),
    ("workers", "Prozesse (0 = automatisch)", int, 0, 64, 1, ""),
    ("seed", "Seed", int, 0, 2_000_000_000, 1, ""),
]

VIEWS = ["Original", "Klassen", "Höhe", "Vorschau (schattiert)", "Original + Prüfbereiche"]
TOOLS = ["Ansehen / Verschieben", "Klasse malen", "Klasse: Automatik wiederherstellen", "Höhe +", "Höhe −",
         "Ebnen", "Farbe für Klasse aufnehmen", "Gebäude bearbeiten", "Gebäude hinzufügen (Rechteck ziehen)",
         "Gebäude entfernen (Klick)", "3D-Ausschnitt wählen (Klick)"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map2MC – Kartenbild → Minecraft Java 1.21.11")
        self.resize(1500, 950)
        self.prj: Project | None = None
        self.thread: QThread | None = None
        self.job: Job | None = None
        self.qs = QSettings("map2mc", "map2mc")
        i18n.set_language(str(self.qs.value("language", i18n.DEFAULT)))
        i18n.install()
        self._dirty_classes = False
        self._build_ui()
        i18n.translate_tree(self)
        last = self.qs.value("last_project")
        if last and os.path.isdir(last):
            try:
                self.open_project(last)
            except Exception as e:  # noqa: BLE001
                self.log(f"Letztes Projekt konnte nicht geladen werden: {e}")

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        m = self.menuBar().addMenu("&Datei")
        for text, fn, sc in (("Kartenbild hochladen (neues Projekt)…", self.new_project, "Ctrl+N"),
                             ("Projekt öffnen…", self.open_project_dialog, "Ctrl+O"),
                             ("Speichern", self.save_project, "Ctrl+S")):
            a = QAction(text, self)
            a.setShortcut(sc)
            a.triggered.connect(fn)
            m.addAction(a)
        lm = self.menuBar().addMenu("Sprache / Language")
        for code, name in i18n.LANGUAGES.items():
            a = QAction(name, self)
            a.setCheckable(True)
            a.setChecked(code == i18n.LANG)
            a.triggered.connect(lambda _=False, c=code: self.set_ui_language(c))
            lm.addAction(a)

        left = QWidget()
        L = QVBoxLayout(left)

        g1 = QGroupBox("1 · Kartenbild")
        l1 = QVBoxLayout(g1)
        b = QPushButton("Kartenbild hochladen…")
        b.clicked.connect(self.new_project)
        l1.addWidget(b)
        b = QPushButton("Projekt öffnen…")
        b.clicked.connect(self.open_project_dialog)
        l1.addWidget(b)
        self.info = QLabel("Kein Projekt geladen.")
        self.info.setWordWrap(True)
        l1.addWidget(self.info)
        L.addWidget(g1)

        g2 = QGroupBox("2 · Größe, Höhe && Erkennung")
        f2 = QFormLayout(g2)
        self.fields = {}
        for name, label, kind, lo, hi, step, tip in SETTING_FIELDS:
            w = QDoubleSpinBox() if kind is float else QSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            if kind is float:
                w.setDecimals(2)
            w.setToolTip(tip)
            w.valueChanged.connect(self._settings_changed)
            self.fields[name] = w
            f2.addRow(label, w)
        self.crop = QCheckBox("Auf Insel zuschneiden")
        self.crop.toggled.connect(self._settings_changed)
        f2.addRow(self.crop)
        self.gen_buildings = QCheckBox("Gebäude erzeugen")
        self.gen_buildings.setToolTip("Aus: keine Gebäude in der Welt – Städte werden gepflasterte Flächen.")
        self.gen_buildings.toggled.connect(self._settings_changed)
        f2.addRow(self.gen_buildings)
        self.single_houses = QCheckBox("Einzelhäuser außerhalb von Städten erkennen")
        self.single_houses.toggled.connect(self._settings_changed)
        f2.addRow(self.single_houses)
        self.filler = QCheckBox("Zusätzliche Füllbäume in Wäldern (nicht auf der Karte)")
        self.filler.setToolTip("Aus: nur Bäume, die auf der Karte zu sehen sind. An: Wälder werden zusätzlich mit zufälligen Bäumen aufgefüllt.")
        self.filler.toggled.connect(self._settings_changed)
        f2.addRow(self.filler)
        row = QHBoxLayout()
        self.use_pack = QCheckBox("Bäume aus Baum-Paket (Standard)")
        self.use_pack.setToolTip("Bäume aus einem importierten Schematic-Paket statt der eingebauten Formen. "
                                 "Blätter werden beim Import auf persistent=true gesetzt.")
        self.use_pack.toggled.connect(self._settings_changed)
        row.addWidget(self.use_pack)
        b = QPushButton("Baum-Paket importieren…")
        b.clicked.connect(self.import_tree_pack)
        row.addWidget(b)
        f2.addRow(row)
        self.pack_label = QLabel("kein Baum-Paket")
        self.pack_label.setWordWrap(True)
        f2.addRow(self.pack_label)
        self.size_label = QLabel("")
        f2.addRow("Weltgröße:", self.size_label)
        L.addWidget(g2)

        g3 = QGroupBox("3 · Analyse")
        l3 = QVBoxLayout(g3)
        self.btn_analyse = QPushButton("Karte analysieren / neu berechnen")
        self.btn_analyse.clicked.connect(self.run_analysis)
        l3.addWidget(self.btn_analyse)
        row = QHBoxLayout()
        b = QPushButton("Blockzuordnung…")
        b.clicked.connect(self.edit_palette)
        row.addWidget(b)
        b = QPushButton("Gebäude…")
        b.clicked.connect(self.edit_buildings)
        row.addWidget(b)
        l3.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Höhenkarte importieren…")
        b.clicked.connect(self.import_heightmap)
        row.addWidget(b)
        b = QPushButton("entfernen")
        b.clicked.connect(lambda: self.prj and (self.prj.clear_heightmap(), self.refresh_views()))
        row.addWidget(b)
        l3.addLayout(row)
        L.addWidget(g3)

        g4 = QGroupBox("4 · Manuelle Bearbeitung")
        f4 = QFormLayout(g4)
        self.tool = QComboBox()
        self.tool.addItems(TOOLS)
        self.tool.currentIndexChanged.connect(self._tool_changed)
        f4.addRow("Werkzeug", self.tool)
        self.cls_box = QComboBox()
        for key, label, color in CLASSES:
            pm = QPixmap(14, 14)
            pm.fill(QColor(*color))
            self.cls_box.addItem(QIcon(pm), label, key)
        f4.addRow("Klasse", self.cls_box)
        self.brush = QSpinBox()
        self.brush.setRange(1, 200)
        self.brush.setValue(6)
        self.brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush", float(v)))
        f4.addRow("Pinsel (px)", self.brush)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.1, 20)
        self.strength.setValue(1.0)
        f4.addRow("Höhen-Stärke", self.strength)
        L.addWidget(g4)

        g5 = QGroupBox("5 · Minecraft-Welt")
        f5 = QFormLayout(g5)
        self.world_name = QLineEdit()
        f5.addRow("Weltname", self.world_name)
        self.game_mode = QComboBox()
        self.game_mode.addItems(["Überleben", "Kreativ"])
        f5.addRow("Spielmodus", self.game_mode)
        row = QHBoxLayout()
        self.out_dir = QLineEdit()
        row.addWidget(self.out_dir)
        b = QPushButton("…")
        b.setFixedWidth(28)
        b.clicked.connect(self._choose_out)
        row.addWidget(b)
        f5.addRow("Ausgabeordner", row)
        row = QHBoxLayout()
        self.btn_gen = QPushButton("Generierung starten")
        self.btn_gen.clicked.connect(self.run_generate)
        row.addWidget(self.btn_gen)
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_job)
        row.addWidget(self.btn_cancel)
        f5.addRow(row)
        row = QHBoxLayout()
        b = QPushButton("In Minecraft-Saves kopieren")
        b.clicked.connect(self.copy_to_saves)
        row.addWidget(b)
        b = QPushButton("Als ZIP exportieren…")
        b.clicked.connect(self.export_zip)
        row.addWidget(b)
        f5.addRow(row)
        b = QPushButton("Weltordner öffnen")
        b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir.text())))
        f5.addRow(b)
        L.addWidget(g5)
        L.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(390)

        # right side: views
        self.tabs = QTabWidget()
        mapw = QWidget()
        ml = QVBoxLayout(mapw)
        top = QHBoxLayout()
        self.view_box = QComboBox()
        self.view_box.addItems(VIEWS)
        self.view_box.currentIndexChanged.connect(self.refresh_views)
        top.addWidget(QLabel("Ansicht:"))
        top.addWidget(self.view_box)
        self.overlay_classes = QCheckBox("Klassen überlagern")
        self.overlay_classes.toggled.connect(self.refresh_views)
        top.addWidget(self.overlay_classes)
        self.show_buildings = QCheckBox("Gebäude anzeigen")
        self.show_buildings.toggled.connect(self.refresh_views)
        top.addWidget(self.show_buildings)
        top.addStretch(1)
        self.hover_label = QLabel("")
        top.addWidget(self.hover_label)
        ml.addLayout(top)
        self.canvas = MapCanvas()
        self.canvas.stroke.connect(self.on_stroke)
        self.canvas.stroke_end.connect(self.on_stroke_end)
        self.canvas.picked.connect(self.on_pick)
        self.canvas.rect_drawn.connect(self.on_rect)
        self.canvas.hover.connect(self.on_hover)
        ml.addWidget(self.canvas)
        self.tabs.addTab(mapw, "Karte")

        w3d = QWidget()
        l3d = QVBoxLayout(w3d)
        row = QHBoxLayout()
        row.addWidget(QLabel("Drehen:"))
        self.rot = QSlider(Qt.Horizontal)
        self.rot.setRange(0, 3)
        self.rot.valueChanged.connect(self.render_3d)
        row.addWidget(self.rot)
        row.addWidget(QLabel("Überhöhung:"))
        self.exag = QSlider(Qt.Horizontal)
        self.exag.setRange(5, 60)
        self.exag.setValue(30)
        self.exag.valueChanged.connect(self.render_3d)
        row.addWidget(self.exag)
        l3d.addLayout(row)
        self.view3d = MapCanvas()
        l3d.addWidget(self.view3d)
        self.tabs.addTab(w3d, "Terrain-Vorschau 3D")
        self.tabs.currentChanged.connect(lambda i: i == 1 and self.render_3d())

        # --- textured block view of the generated world (Litematica-like)
        wb = QWidget()
        lb = QVBoxLayout(wb)
        row = QHBoxLayout()
        self.b3_x = QSpinBox()
        self.b3_z = QSpinBox()
        for sp in (self.b3_x, self.b3_z):
            sp.setRange(-29_000_000, 29_000_000)
        self.b3_size = QSpinBox()
        self.b3_size.setRange(32, 384)
        self.b3_size.setSingleStep(16)
        self.b3_size.setValue(160)
        self.b3_size.setSuffix(" Blöcke")
        for lab, wdg in (("Mitte X:", self.b3_x), ("Z:", self.b3_z), ("Detailbereich:", self.b3_size)):
            row.addWidget(QLabel(lab))
            row.addWidget(wdg)
        b = QPushButton("Laden")
        b.clicked.connect(lambda: self.load_blocks_3d())
        row.addWidget(b)
        self.b3_lod = QCheckBox("ganze Karte (LOD)")
        self.b3_lod.setChecked(True)
        self.b3_lod.setToolTip("Wie Voxy: um die Kamera echte Blöcke, weiter weg vereinfachte Säulen – "
                               "so ist die ganze Karte sichtbar.")
        row.addWidget(self.b3_lod)
        self.b3_follow = QCheckBox("Detail folgt Kamera")
        self.b3_follow.setChecked(True)
        self.b3_follow.setToolTip("Wenn du dich bewegst, wird der Detailbereich an die neue Stelle gelegt.")
        row.addWidget(self.b3_follow)
        self.b3_mode = QComboBox()
        self.b3_mode.addItems(["Spieler (WASD)", "Orbit (Maus)"])
        self.b3_mode.setToolTip("Spieler: ins Bild klicken, Maus umsehen, WASD bewegen, Leertaste/Shift hoch/runter, "
                                "F Fliegen/Laufen, Esc Maus freigeben.\nOrbit: Maus ziehen zum Drehen/Verschieben.")
        self.b3_mode.currentIndexChanged.connect(
            lambda i: self.blockview.set_mode("player" if i == 0 else "orbit"))
        row.addWidget(self.b3_mode)
        self.b3_holo = QCheckBox("Hologramm")
        self.b3_holo.toggled.connect(lambda on: self.blockview.set_hologram(on))
        row.addWidget(self.b3_holo)
        row.addWidget(QLabel("Ebenen bis:"))
        self.b3_clip = QSlider(Qt.Horizontal)
        self.b3_clip.setRange(0, 1)
        self.b3_clip.setValue(1)
        self.b3_clip.setEnabled(False)
        self.b3_clip.valueChanged.connect(self._clip_changed)
        row.addWidget(self.b3_clip, 1)
        self.b3_clip_label = QLabel("–")
        row.addWidget(self.b3_clip_label)
        lb.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Ausschnitt als .litematic…")
        b.clicked.connect(self.export_litematic_area)
        row.addWidget(b)
        b = QPushButton("Ganze Welt als .litematic-Kacheln…")
        b.clicked.connect(self.export_litematic_tiles)
        row.addWidget(b)
        row.addWidget(QLabel("Linke Maus: drehen · Rechte Maus: verschieben · Rad: Zoom · Doppelklick: zurücksetzen. "
                             "Ausschnitt auf der Karte mit „3D-Ausschnitt wählen“ anklicken."))
        row.addStretch(1)
        lb.addLayout(row)
        self.blockview = BlockView()
        self.blockview.status.connect(self.log)
        self.blockview.moved.connect(self._view_moved)
        from ..scene3d import LodCache
        self._lod_cache = LodCache()
        lb.addWidget(self.blockview, 1)
        self.tabs.addTab(wb, "3D-Blöcke / Hologramm")
        self._b3_info = None
        self._b3_thread: QThread | None = None
        self._b3_job = None
        self._b3_pending = None

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        bl.addWidget(self.bar)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(5000)
        bl.addWidget(self.logbox)

        vs = QSplitter(Qt.Vertical)
        vs.addWidget(self.tabs)
        vs.addWidget(bottom)
        vs.setSizes([760, 170])
        hs = QSplitter(Qt.Horizontal)
        hs.addWidget(scroll)
        hs.addWidget(vs)
        hs.setSizes([400, 1100])
        self.setCentralWidget(hs)

    # ------------------------------------------------------------ helpers
    def log(self, s: str):
        self.logbox.appendPlainText(s)

    def _require(self, derived=False) -> bool:
        if self.prj is None:
            QMessageBox.information(self, "Map2MC", "Bitte zuerst ein Kartenbild hochladen oder ein Projekt öffnen.")
            return False
        if derived and self.prj.classes is None:
            QMessageBox.information(self, "Map2MC", "Bitte zuerst die Karte analysieren.")
            return False
        return True

    def _busy(self) -> bool:
        if self.thread is not None:
            QMessageBox.information(self, "Map2MC", "Es läuft bereits ein Vorgang.")
            return True
        return False

    def start_job(self, fn, on_done=None, label=""):
        if self._busy():
            return
        self.save_project()
        self.thread = QThread()
        self.job = Job(fn)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.run)
        # IMPORTANT: connect to slots of this window (a QObject living in the GUI
        # thread).  Plain lambdas/closures would be executed in the *worker*
        # thread, which touches widgets from the wrong thread and makes the
        # thread wait on itself -> Qt aborts the whole process.
        self._on_done = on_done
        self.job.progress.connect(self._job_progress, Qt.QueuedConnection)
        self.job.log.connect(self._job_log, Qt.QueuedConnection)
        self.job.done.connect(self._job_done, Qt.QueuedConnection)
        self.job.failed.connect(self._job_failed, Qt.QueuedConnection)
        self.btn_cancel.setEnabled(True)
        self.bar.setValue(0)
        if label:
            self.log(label)
        self.thread.start()

    def _stop_thread(self):
        th = self.thread
        self.thread = None
        self.job = None
        if th is not None:
            th.quit()
            th.wait()
        self.btn_cancel.setEnabled(False)

    @Slot(float)
    def _job_progress(self, f):
        self.bar.setValue(int(f * 1000))

    @Slot(str)
    def _job_log(self, s):
        self.log(s)

    @Slot(object)
    def _job_done(self, res):
        self._stop_thread()
        cb, self._on_done = self._on_done, None
        if cb:
            cb(res)

    @Slot(str)
    def _job_failed(self, msg):
        self._stop_thread()
        self._on_done = None
        self.log("FEHLER: " + msg)
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("Job-Fehler:\n" + msg + "\n")
        except OSError:
            pass
        QMessageBox.critical(self, "Fehler", msg[-2000:])

    def cancel_job(self):
        if self.job:
            self.job.cancel.set()
            self.log("Abbruch angefordert …")

    # ------------------------------------------------------------ project
    def new_project(self):
        if self._busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Kartenbild wählen", self.qs.value("last_dir", ""),
                                              "Bilder (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)")
        if not path:
            return
        self.qs.setValue("last_dir", os.path.dirname(path))
        default = os.path.splitext(path)[0] + "_map2mc"
        folder = QFileDialog.getExistingDirectory(self, "Projektordner wählen (wird angelegt)", os.path.dirname(path))
        folder = os.path.join(folder, os.path.basename(default)) if folder else default
        try:
            prj = Project.create(path, folder)
        except MapImageError as e:
            QMessageBox.critical(self, "Bild nicht verwendbar", str(e))
            return
        self.open_project(prj.dir)
        self.log(f"Neues Projekt: {prj.dir}")
        self.run_analysis()

    def open_project_dialog(self):
        if self._busy():
            return
        d = QFileDialog.getExistingDirectory(self, "Projektordner öffnen", self.qs.value("last_dir", ""))
        if d:
            self.open_project(d)

    def open_project(self, folder):
        if not os.path.exists(os.path.join(folder, "source.png")):
            raise MapImageError("Kein Map2MC-Projektordner (source.png fehlt).")
        self.prj = Project(folder)
        self.qs.setValue("last_project", self.prj.dir)
        s = self.prj.settings
        for name, w in self.fields.items():
            w.blockSignals(True)
            w.setValue(getattr(s, name))
            w.blockSignals(False)
        for box, val in ((self.crop, s.crop_to_island), (self.gen_buildings, s.generate_buildings),
                         (self.single_houses, s.detect_single_houses), (self.use_pack, s.use_tree_pack),
                         (self.filler, s.filler_trees)):
            box.blockSignals(True)
            box.setChecked(val)
            box.blockSignals(False)
        self.world_name.setText(s.world_name)
        self.game_mode.setCurrentIndex(1 if s.game_mode == 1 else 0)
        safe = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in s.world_name).strip() or "world"
        self.out_dir.setText(os.path.join(self.prj.dir, "worlds", safe))
        h, w = self.prj.image.shape[:2]
        self.info.setText(f"{os.path.basename(self.prj.dir)}\nBild: {w}×{h} px")
        self._update_size()
        self._show_pack()
        self.refresh_views(fit=True)

    def _sync_settings(self):
        if not self.prj:
            return
        s = self.prj.settings
        for name, w in self.fields.items():
            setattr(s, name, type(getattr(s, name))(w.value()))
        s.crop_to_island = self.crop.isChecked()
        s.generate_buildings = self.gen_buildings.isChecked()
        s.detect_single_houses = self.single_houses.isChecked()
        s.use_tree_pack = self.use_pack.isChecked()
        s.filler_trees = self.filler.isChecked()
        s.world_name = self.world_name.text().strip() or "Map2MC Island"
        s.game_mode = 1 if self.game_mode.currentIndex() == 1 else 0

    def _settings_changed(self, *_):
        self._sync_settings()
        self._update_size()

    def _update_size(self):
        if not self.prj:
            return
        s = self.prj.settings
        if self.prj.land is not None:
            g = self.prj.geometry()
            self.size_label.setText(f"{g.width} × {g.depth} Blöcke  (≈ {g.width / 1000:.2f} × {g.depth / 1000:.2f} km)")
        else:
            h, w = self.prj.image.shape[:2]
            self.size_label.setText(f"{w * s.blocks_per_pixel:.0f} × {h * s.blocks_per_pixel:.0f} Blöcke (ganzes Bild)")

    def save_project(self):
        if self.prj:
            self._sync_settings()
            self.prj.save()

    def _choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Ausgabeordner (Weltordner wird darin angelegt)")
        if d:
            self.out_dir.setText(os.path.join(d, os.path.basename(self.out_dir.text()) or "world"))

    # ------------------------------------------------------------ analysis
    def run_analysis(self):
        if not self._require():
            return
        prj = self.prj

        def work(job: Job):
            job.log.emit("Farben klassifizieren …")
            prj.run_classification(lambda f: job.progress.emit(f * 0.4))
            job.log.emit("Gelände, Gewässer, Bäume, Gebäude ableiten …")
            prj.run_derive(lambda f: job.progress.emit(0.4 + f * 0.6), log=job.log.emit)
            return prj.report()

        def done(rep):
            self._dirty_classes = False
            self.log(f"Unbekannte Pixel: {rep['unbekannte_pixel_prozent']} %, Prüfbereiche: "
                     f"{rep['pruefbereiche_prozent']} % (Ansicht „Original + Prüfbereiche“).")
            for h in rep["hinweise"]:
                self.log("Hinweis: " + h)
            self._update_size()
            self.refresh_views()
            if self.tabs.currentIndex() == 1:
                self.render_3d()

        self.start_job(work, done, "Analyse gestartet …")

    def edit_palette(self):
        if not self._require():
            return
        dlg = PaletteDialog(self.prj.palette, self)
        if dlg.exec():
            self.prj.palette = dlg.palette
            self.save_project()
            self.log("Blockzuordnung gespeichert.")

    def edit_buildings(self):
        if not self._require(derived=True):
            return
        dlg = BuildingDialog(self.prj, self)
        dlg.goto.connect(lambda u, v: (self.tabs.setCurrentIndex(0), self.canvas.center_on_px(u, v, 6)))
        dlg.exec()

    def import_heightmap(self):
        if not self._require(derived=True):
            return
        path, _ = QFileDialog.getOpenFileName(self, "Höhenkarte (Graustufen, hell = hoch)", "",
                                              "Bilder (*.png *.tif *.tiff *.jpg *.webp)")
        if not path:
            return
        mx = float(self.prj.settings.max_land_height + self.prj.settings.mountain_height)
        try:
            self.prj.import_heightmap(path, 1.0, mx)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Höhenkarte", str(e))
            return
        self.log(f"Höhenkarte importiert (Land: 1 … {mx:.0f} Blöcke über dem Meer).")
        self.refresh_views()

    # ------------------------------------------------------------ views
    def _class_display(self):
        if self.prj.classes is not None and not self._dirty_classes:
            cls = self.prj.classes.copy()
        elif self.prj.class_auto is not None:
            cls = self.prj.class_auto.copy()
        else:
            return None
        m = self.prj.class_edit != 255
        cls[m] = self.prj.class_edit[m]
        return cls

    def refresh_views(self, *_, fit=False):
        if not self.prj:
            return
        p = self.prj
        v = VIEWS[max(0, self.view_box.currentIndex())]
        img = p.image
        overlay = None
        if v == "Klassen":
            cls = self._class_display()
            if cls is not None:
                img = analysis.class_preview(cls)
        elif v == "Höhe" and p.classes is not None:
            img = preview.height_view(p.height)
        elif v.startswith("Vorschau") and p.classes is not None:
            img = preview.shaded(p.classes, p.height, p.land)
        elif v.endswith("Prüfbereiche") and p.review is not None:
            ov = np.zeros(p.review.shape + (4,), np.uint8)
            ov[p.review] = (255, 0, 255, 255)
            overlay = ov
        if self.overlay_classes.isChecked() and v != "Klassen":
            cls = self._class_display()
            if cls is not None:
                ov = np.concatenate([analysis.class_preview(cls), np.full(cls.shape + (1,), 255, np.uint8)], -1)
                overlay = ov
        if self.show_buildings.isChecked() and p.buildings:
            overlay = np.zeros(img.shape[:2] + (4,), np.uint8) if overlay is None else overlay.copy()
            colors = {"house": (255, 40, 40, 255), "platform": (40, 80, 255, 255), "none": (130, 130, 130, 255)}
            for b in p.buildings:
                cv2.polylines(overlay, [np.array(b.polygon, np.int32)], True, colors.get(b.kind, colors["none"]), 1)
        self.canvas.set_image(img, fit=fit)
        self.canvas.set_overlay(overlay)

    def render_3d(self, *_):
        p = self.prj
        if p is None or p.classes is None:
            return
        g = p.geometry()
        sl = (slice(g.cv0, g.cv1), slice(g.cu0, g.cu1))
        img = preview.oblique(p.classes[sl], p.height[sl] / max(p.settings.blocks_per_pixel, 1e-3), p.land[sl],
                              rotation=self.rot.value(), size=900, exaggeration=self.exag.value() / 10.0)
        self.view3d.set_image(img, fit=True)

    # ------------------------------------------------------------ editing
    def _tool_changed(self, i):
        t = TOOLS[i]
        if t.startswith("Ansehen"):
            self.canvas.mode = "view"
        elif t.startswith("Gebäude hinzufügen"):
            self.canvas.mode = "rect"
            self.show_buildings.setChecked(True)
        elif t.startswith("3D-Ausschnitt"):
            self.canvas.mode = "pick"
        elif t.startswith("Farbe") or t.startswith("Gebäude"):
            self.canvas.mode = "pick"
            if t.startswith("Gebäude"):
                self.show_buildings.setChecked(True)
        else:
            self.canvas.mode = "paint"
        if t.startswith("Klasse") and VIEWS[self.view_box.currentIndex()] != "Klassen":
            self.view_box.setCurrentIndex(VIEWS.index("Klassen"))
        if t.startswith("Höhe") or t == "Ebnen":
            self.view_box.setCurrentIndex(VIEWS.index("Vorschau (schattiert)"))

    def _disk(self, u, v):
        h, w = self.prj.image.shape[:2]
        r = self.brush.value()
        y0, y1 = max(0, int(v - r)), min(h, int(v + r) + 1)
        x0, x1 = max(0, int(u - r)), min(w, int(u + r) + 1)
        if y1 <= y0 or x1 <= x0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx + 0.5 - u) ** 2 + (yy + 0.5 - v) ** 2) / (r * r)
        return (slice(y0, y1), slice(x0, x1)), d2

    def on_stroke(self, u, v, first):
        if not self.prj:
            return
        t = TOOLS[self.tool.currentIndex()]
        disk = self._disk(u, v)
        if disk is None:
            return
        sl, d2 = disk
        inside = d2 <= 1
        if t == "Klasse malen":
            self.prj.class_edit[sl][inside] = CID[self.cls_box.currentData()]
            self._dirty_classes = True
        elif t.startswith("Klasse: Automatik"):
            self.prj.class_edit[sl][inside] = 255
            self._dirty_classes = True
        elif t in ("Höhe +", "Höhe −", "Ebnen"):
            if self.prj.height is None:
                return
            fall = np.clip(1 - d2, 0, 1) ** 1.5
            if t == "Ebnen":
                target = float(self.prj.height[sl][inside].mean()) if inside.any() else 0.0
                self.prj.height_delta[sl] += (target - self.prj.height[sl]) * fall * 0.5
            else:
                sgn = 1 if t == "Höhe +" else -1
                self.prj.height_delta[sl] += sgn * self.strength.value() * fall
            self.prj.apply_height_edits(save=False)
        else:
            return
        self.refresh_views()

    def on_stroke_end(self):
        if not self.prj:
            return
        self.prj.save_edits()
        t = TOOLS[self.tool.currentIndex()]
        if t.startswith("Höhe") or t == "Ebnen":
            self.prj.apply_height_edits(save=True)
        if self._dirty_classes:
            self.log("Klassen geändert – „Karte analysieren / neu berechnen“ übernimmt die Änderung in Höhe, "
                     "Bäume und Gebäude.")

    def on_pick(self, u, v):
        if not self.prj:
            return
        t = TOOLS[self.tool.currentIndex()]
        h, w = self.prj.image.shape[:2]
        x, y = int(u), int(v)
        if not (0 <= x < w and 0 <= y < h):
            return
        if t.startswith("3D-Ausschnitt"):
            X, Z = self.prj.geometry().px_to_world(np.array([u]), np.array([v]))
            self.b3_x.setValue(int(X[0]))
            self.b3_z.setValue(int(Z[0]))
            self.tabs.setCurrentIndex(2)
            self.load_blocks_3d()
            return
        if t.startswith("Farbe"):
            key = self.cls_box.currentData()
            rgb = [int(c) for c in cv2.medianBlur(self.prj.image[max(0, y - 2):y + 3, max(0, x - 2):x + 3], 3)
                   .reshape(-1, 3).mean(0)]
            self.prj.prototypes.setdefault(key, []).append(rgb)
            self.save_project()
            self.log(f"Farbe {rgb} als Prototyp für „{dict((c[0], c[1]) for c in CLASSES)[key]}“ hinzugefügt – "
                     "bitte neu analysieren.")
        elif t.startswith("Gebäude entfernen"):
            b = self.prj.building_at(u, v)
            if b is None:
                self.log("Kein Gebäude an dieser Stelle.")
                return
            new = "house" if b.kind == "none" else "none"
            self.prj.edit_building(b, kind=new)
            self.log(f"Gebäude {b.id}: {'wiederhergestellt' if new == 'house' else 'entfernt'} (erneut klicken = rückgängig).")
            self.refresh_views()
        elif t.startswith("Gebäude"):
            b = self.prj.building_at(u, v)
            if b is None:
                self.log("Kein Gebäude an dieser Stelle.")
                return
            dlg = SingleBuildingDialog(b, self)
            if dlg.exec():
                self.prj.edit_building(b, kind=dlg.kind.currentText(), storeys=dlg.storeys.value())
                self.log(f"Gebäude {b.id}: {b.kind}, {b.storeys} Stockwerk(e).")

    def on_rect(self, u0, v0, u1, v1):
        if not self._require(derived=True):
            return
        b = self.prj.add_building(u0, v0, u1, v1)
        s = self.prj.settings.blocks_per_pixel
        self.log(f"Gebäude {b.id} hinzugefügt ({(u1 - u0) * s:.0f} × {(v1 - v0) * s:.0f} Blöcke, Dach {b.roof_block}). "
                 "Mit „Gebäude bearbeiten“ Stockwerke/Typ ändern.")
        self.refresh_views()

    def on_hover(self, u, v):
        if not self.prj:
            return
        h, w = self.prj.image.shape[:2]
        x, y = int(u), int(v)
        if 0 <= x < w and 0 <= y < h:
            txt = f"px {x},{y}"
            if self.prj.classes is not None:
                txt += f" · {CLASSES[int(self.prj.classes[y, x])][1]}"
            if self.prj.height is not None:
                txt += f" · Y {self.prj.settings.water_level + int(round(float(self.prj.height[y, x])))}"
            if self.prj.land is not None:
                g = self.prj.geometry()
                X, Z = g.px_to_world(x + 0.5, y + 0.5)
                txt += f" · Welt X {int(X)} Z {int(Z)}"
            self.hover_label.setText(txt)

    # ------------------------------------------------------------ generation / export
    def run_generate(self):
        if not self._require(derived=True):
            return
        if self._dirty_classes:
            if QMessageBox.question(self, "Map2MC", "Klassen wurden bemalt, aber noch nicht neu berechnet. "
                                    "Trotzdem mit dem alten Stand generieren?") != QMessageBox.Yes:
                return
        self._sync_settings()
        out = self.out_dir.text().strip()
        if not out:
            return
        if os.path.exists(os.path.join(out, "level.dat")) and not os.path.exists(os.path.join(out, "map2mc_progress.json")):
            QMessageBox.warning(self, "Map2MC", "Im Ausgabeordner liegt bereits eine fremde Welt. Bitte anderen Ordner wählen.")
            return
        prj = self.prj

        def work(job: Job):
            return prj.generate(out, progress=job.progress.emit, log=job.log.emit, cancel=job.cancel)

        def done(res):
            if res and not res.get("cancelled"):
                self.log("Fertig. Die Welt kann jetzt in die Minecraft-Saves kopiert oder als ZIP exportiert werden.")

        self.start_job(work, done, f"Generiere Welt nach {out} …")

    # ------------------------------------------------------------ 3D block view / litematic
    def _world_dir(self, need=True):
        w = self.out_dir.text().strip()
        if not os.path.isdir(os.path.join(w, "region")):
            if need:
                QMessageBox.information(self, "Map2MC", "Bitte zuerst die Welt generieren – die 3D-Blockansicht und "
                                        "der .litematic-Export lesen die erzeugten Regionsdateien.")
            return None
        return w

    def _b3_box(self):
        s = self.b3_size.value()
        return self.b3_x.value() - s // 2, self.b3_z.value() - s // 2, s

    def load_blocks_3d(self, keep_camera=False):
        """Load the 3D scene in the 3D view's own worker thread (independent of analysis/generation).

        While a load is running, further requests are not refused: only the newest one is
        remembered and started as soon as the running one is finished (fast flying never blocks)."""
        world = self._world_dir()
        if world is None:
            return
        req = dict(world=world, xc=self.b3_x.value(), zc=self.b3_z.value(), size=self.b3_size.value(),
                   lod=self.b3_lod.isChecked(), keep_camera=keep_camera)
        if self._b3_thread is not None:
            if self._b3_pending is not None and not self._b3_pending["keep_camera"]:
                req["keep_camera"] = False          # an explicit "Laden" must not turn into a follow-up
            self._b3_pending = req
            return
        self._b3_start(req)

    def _b3_start(self, req):
        cache = self._lod_cache

        def work(job: Job):
            from .. import blockmesh, scene3d
            jar = blockmesh.find_client_jar()
            if jar is None:
                job.log.emit("Kein Minecraft-1.21-Client gefunden – Blöcke werden einfarbig dargestellt.")
            res = scene3d.build_scene(req["world"], req["xc"], req["zc"], req["size"], cache, jar,
                                      progress=job.progress.emit, log=job.log.emit, lod=req["lod"])
            return req, res

        self._b3_thread = QThread()
        self._b3_job = Job(work)
        self._b3_job.moveToThread(self._b3_thread)
        self._b3_thread.started.connect(self._b3_job.run)
        # slots of this window (GUI thread), see start_job
        if not req["keep_camera"]:
            self._b3_job.progress.connect(self._job_progress, Qt.QueuedConnection)
            self.log("Lade 3D-Ansicht …")
        self._b3_job.log.connect(self._job_log, Qt.QueuedConnection)
        self._b3_job.done.connect(self._b3_done, Qt.QueuedConnection)
        self._b3_job.failed.connect(self._b3_failed, Qt.QueuedConnection)
        self._b3_thread.start()

    def _b3_stop_thread(self):
        th, self._b3_thread, self._b3_job = self._b3_thread, None, None
        if th is not None:
            th.quit()
            th.wait()

    def _b3_next(self):
        nxt, self._b3_pending = self._b3_pending, None
        if nxt is not None:
            self._b3_start(nxt)

    @Slot(str)
    def _b3_failed(self, msg):
        self._b3_stop_thread()
        self.log("FEHLER (3D): " + msg)
        self._b3_next()

    @Slot(object)
    def _b3_done(self, out):
        self._b3_stop_thread()
        req, res = out
        keep_camera = req["keep_camera"]
        old = self._b3_info["focus"] if (keep_camera and self._b3_info) else None
        if res["detail_faces"] == 0 and not keep_camera:
            self.log("Dieser Ausschnitt ist leer (liegt außerhalb der erzeugten Welt).")
        shift = (res["focus"][0] - old[0], res["focus"][1] - old[1]) if old else (0.0, 0.0)
        self.blockview.set_scene(res["opaque"], res["water"], res["atlas"], res["center"], res["size"],
                                 res["world_size"], keep_camera=old is not None, shift=shift)
        self.blockview.world_origin = res["focus"]
        self.blockview.set_ground(*res["ground"])
        if old is None:
            self.blockview.reset_player()
        self._b3_info = res
        self.b3_clip.blockSignals(True)
        self.b3_clip.setRange(res["y_min"], res["y_max"] + 1)
        self.b3_clip.setValue(res["y_max"] + 1)
        self.b3_clip.blockSignals(False)
        self.b3_clip.setEnabled(True)
        self._clip_changed(res["y_max"] + 1)
        if not keep_camera:
            self.bar.setValue(1000)
            fx, fz = res["focus"]
            extra = (f", Rest der Karte vereinfacht (Zellen 2 / {res['far_cell']} Blöcke)" if req["lod"] else "")
            self.log(f"3D: Detailbereich um x {fx}, z {fz} ({res['size']}×{res['size']} Blöcke){extra}.")
        self._b3_next()

    def _view_moved(self, x, z):
        """Camera was dragged: move the detailed area along (like Voxy)."""
        if not self._b3_info or not self.b3_follow.isChecked():
            return
        fx, fz = self._b3_info["focus"]
        if max(abs(x), abs(z)) < self._b3_info["size"] * 0.35:
            return
        self.b3_x.setValue(int(fx + x))
        self.b3_z.setValue(int(fz + z))
        self.load_blocks_3d(keep_camera=True)

    def _clip_changed(self, val):
        if not self._b3_info:
            return
        top = self._b3_info["y_max"] + 1
        self.blockview.set_clip(val if val < top else 1e9)
        self.b3_clip_label.setText(f"Y {val - 1}" if val < top else "alle")

    def export_litematic_area(self):
        world = self._world_dir()
        if world is None:
            return
        x0, z0, s = self._b3_box()
        path, _ = QFileDialog.getSaveFileName(self, ".litematic speichern", f"map2mc_x{x0}_z{z0}.litematic",
                                              "Litematica (*.litematic)")
        if not path:
            return

        def work(job: Job):
            from .. import litematic
            return litematic.export_area(world, path, x0, z0, s, s, progress=job.progress.emit)

        def done(info):
            x, y, z = info["origin"]
            self.log(f".litematic gespeichert: {info['path']} – {info['blocks']:,} Blöcke, "
                     f"Ursprung in der Welt x={x} y={y} z={z}".replace(",", "."))

        self.start_job(work, done, "Exportiere .litematic …")

    def export_litematic_tiles(self):
        world = self._world_dir()
        if world is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Ordner für .litematic-Kacheln (z. B. .minecraft/schematics)")
        if not folder:
            return
        from .. import worldread
        x0, z0, x1, z1 = worldread.world_bounds(world)

        def work(job: Job):
            from .. import litematic
            return litematic.export_tiles(world, folder, x0, z0, x1, z1, 256, progress=job.progress.emit,
                                          cancel=job.cancel)

        def done(infos):
            self.log(f"{len(infos)} .litematic-Kacheln (256×256) gespeichert in {folder}. "
                     "Der Dateiname enthält die Weltkoordinate der Kachel-Ecke.")

        self.start_job(work, done, "Exportiere .litematic-Kacheln …")

    # ------------------------------------------------------------ tree pack
    def _show_pack(self):
        from .. import treepack
        p = treepack.resolve(self.prj.settings.tree_pack if self.prj else "")
        if p is None:
            self.pack_label.setText("kein Baum-Paket (eingebaute Baumformen)")
        elif os.path.normcase(p) == os.path.normcase(treepack.DEFAULT_PACK):
            n = len(treepack.load_library(p).trees)
            self.pack_label.setText(f"Standard-Baumpaket (kuratiert, {n} Bäume)")
        else:
            self.pack_label.setText("Baum-Paket: " + os.path.basename(p))

    def import_tree_pack(self):
        if not self._require():
            return
        src, _ = QFileDialog.getOpenFileName(self, "Baum-Paket wählen (.7z / .zip mit .schem-Dateien)", "",
                                             "Baum-Pakete (*.7z *.zip)")
        if not src:
            return
        out = os.path.join(TREEPACK_DIR, os.path.splitext(os.path.basename(src))[0] + ".m2trees")

        def work(job: Job):
            from .. import treepack
            return treepack.import_pack(src, out, progress=job.progress.emit, log=job.log.emit, cancel=job.cancel)

        def done(summary):
            self.prj.settings.tree_pack = summary["path"]
            self.prj.settings.use_tree_pack = True
            self.use_pack.blockSignals(True)
            self.use_pack.setChecked(True)
            self.use_pack.blockSignals(False)
            self.prj.save()
            self._show_pack()
            kinds = ", ".join(f"{k} {v}" for k, v in summary["by_kind"].items())
            self.log(f"Baum-Paket importiert: {summary['trees']} Bäume ({kinds}). Nicht übernommen: "
                     + ", ".join(f"{k} {v}" for k, v in summary["skipped"].items())
                     + ". Fehlende Arten (z. B. Kirsche) nutzen Eichen-Formen mit passendem Laub. "
                     "Welt neu generieren, um die Bäume zu verwenden.")

        self.start_job(work, done, f"Importiere Baum-Paket {os.path.basename(src)} …")

    def copy_to_saves(self):
        src = self.out_dir.text().strip()
        if not os.path.exists(os.path.join(src, "level.dat")):
            QMessageBox.information(self, "Map2MC", "Bitte zuerst die Welt generieren.")
            return
        dest = QFileDialog.getExistingDirectory(self, "Minecraft-Saves-Ordner", minecraft_saves_dir())
        if not dest:
            return
        try:
            p = copy_world(src, dest)
        except FileExistsError as e:
            QMessageBox.warning(self, "Map2MC", f"Dort existiert bereits eine Welt mit diesem Namen:\n{e}")
            return
        self.log(f"Welt kopiert nach {p}")

    def export_zip(self):
        src = self.out_dir.text().strip()
        if not os.path.exists(os.path.join(src, "level.dat")):
            QMessageBox.information(self, "Map2MC", "Bitte zuerst die Welt generieren.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "ZIP speichern", os.path.basename(src) + ".zip", "ZIP (*.zip)")
        if path:
            export_zip(src, path)
            self.log(f"ZIP gespeichert: {path}")

    def set_ui_language(self, code):
        """Switch the language: the window is rebuilt (the project stays open)."""
        if code == i18n.LANG:
            return
        if self.thread is not None or self._b3_thread is not None:
            QMessageBox.information(self, "Map2MC", "Bitte warten, bis der laufende Vorgang fertig ist.")
            for a in self.menuBar().findChildren(QAction):
                if a.isCheckable():
                    a.setChecked(a.text() == i18n.LANGUAGES[i18n.LANG])
            return
        self.qs.setValue("language", code)
        if self.prj:
            self.save_project()
        i18n.set_language(code)
        w = MainWindow()
        w.setGeometry(self.geometry())
        w.show()
        QApplication.instance()._map2mc_window = w
        self.prj = None
        self.close()
        self.deleteLater()

    def closeEvent(self, e):
        if self.thread is not None:
            if QMessageBox.question(self, "Map2MC", "Ein Vorgang läuft noch. Abbrechen und beenden?") != QMessageBox.Yes:
                e.ignore()
                return
            self.cancel_job()
            self.thread.quit()
            self.thread.wait(5000)
        self._b3_pending = None
        self._b3_stop_thread()
        self.save_project()
        e.accept()


TREEPACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "treepacks")
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "map2mc_log.txt")


def _install_crash_logging():
    """pythonw.exe has no console: without this, any Python error would close the
    window silently.  Errors go to map2mc_log.txt and are shown in a dialog;
    hard crashes (segfaults) are written there by faulthandler."""
    import datetime
    import faulthandler
    log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    log.write(f"\n===== Start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    faulthandler.enable(log)

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        log.write(text)
        log.flush()
        try:
            QMessageBox.critical(None, "Map2MC – Fehler",
                                 f"{value}\n\nDetails stehen in:\n{LOG_PATH}\n\n{text[-1500:]}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def run():
    _install_crash_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Map2MC")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    return app.exec()
